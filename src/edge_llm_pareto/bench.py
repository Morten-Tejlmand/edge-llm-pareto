"""Concurrency sweep against an OpenAI-compatible endpoint.

Deliberately speaks only the OpenAI API so the same harness measures vLLM and
llama.cpp's llama-server without a second code path -- engine comparison is only
meaningful if both sides are measured by the same instrument.

Metrics, and why each one is here:
  ttft_s        prefill latency -- what an agent waits before it sees anything
  itl_s         inter-token latency -- per-agent decode speed, the thing that
                degrades as agents are added
  output_tok_s  aggregate decode throughput -- the system-level number that
                *improves* with concurrency; the tension between this and itl_s
                is the pareto front
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field

from .gpumon import GpuMonitor


@dataclass
class RequestResult:
    ok: bool
    ttft_s: float | None = None
    latency_s: float | None = None
    output_tokens: int = 0
    itls_s: list[float] = field(default_factory=list)
    error: str | None = None


@dataclass
class Workload:
    """Shaped like agent traffic: long prompt, moderate output.

    Agents carry big system prompts and tool schemas, so prefill dominates far
    more than in chat benchmarks. Defaults reflect that rather than a short-chat
    workload, which would understate prefill cost.
    """

    prompt_tokens: int = 2000
    max_output_tokens: int = 256
    temperature: float = 0.0  # deterministic: we are measuring speed, not samples

    # Calibrated against the Qwen3 tokenizer: the filler sentence is 10.0 tokens
    # and the head+tail scaffolding is ~18. Lands within 1% of the target, so
    # `prompt_tokens` in the results means what it says.
    _TOKENS_PER_SENTENCE = 10.0
    _SCAFFOLD_TOKENS = 18

    def prompt(self, seed: int) -> str:
        # Distinct prefix per request so prefix caching does not silently turn
        # the concurrency sweep into a cache-hit benchmark.
        head = f"[request {seed:06d}] "
        body_tokens = max(self.prompt_tokens - self._SCAFFOLD_TOKENS, 0)
        repeats = max(round(body_tokens / self._TOKENS_PER_SENTENCE), 1)
        filler = "The quick brown fox jumps over the lazy dog. " * repeats
        return head + filler + "\n\nSummarize the text above in one paragraph."


async def _one_request(client, model: str, wl: Workload, seed: int) -> RequestResult:
    t0 = time.perf_counter()
    ttft: float | None = None
    last = t0
    itls: list[float] = []
    n = 0
    try:
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": wl.prompt(seed)}],
            max_tokens=wl.max_output_tokens,
            temperature=wl.temperature,
            stream=True,
        )
        try:
            # Server-reported token counts are authoritative; llama-server has
            # not always accepted this option, so degrade to counting chunks
            # rather than failing the request.
            stream = await client.chat.completions.create(
                **kwargs, stream_options={"include_usage": True}
            )
        except Exception:
            stream = await client.chat.completions.create(**kwargs)
        usage_tokens = None
        async for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                usage_tokens = chunk.usage.completion_tokens
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = delta.content if delta else None
            if not text:
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            else:
                itls.append(now - last)
            last = now
            n += 1
        total = time.perf_counter() - t0
        return RequestResult(
            ok=True,
            ttft_s=ttft,
            latency_s=total,
            # Prefer server-reported usage; chunk count is a fallback and can
            # differ when a chunk carries multiple tokens.
            output_tokens=usage_tokens if usage_tokens is not None else n,
            itls_s=itls,
        )
    except Exception as e:
        return RequestResult(ok=False, error=f"{type(e).__name__}: {e}", latency_s=time.perf_counter() - t0)


async def run_concurrency_level(
    base_url: str,
    model: str,
    concurrency: int,
    wl: Workload,
    requests_per_agent: int = 3,
    api_key: str = "EMPTY",
    monitor_gpu: bool = True,
) -> dict:
    """Hold `concurrency` requests in flight and measure steady-state behaviour."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=600.0, max_retries=0)
    total_requests = concurrency * requests_per_agent
    counter = iter(range(total_requests))
    lock = asyncio.Lock()
    results: list[RequestResult] = []

    async def agent() -> None:
        """One simulated agent: issues requests back-to-back, keeping exactly
        one in flight, so `concurrency` is the true in-flight request count."""
        while True:
            async with lock:
                seed = next(counter, None)
            if seed is None:
                return
            results.append(await _one_request(client, model, wl, seed))

    mon = GpuMonitor() if monitor_gpu else None
    ctx = mon if mon else _NullCtx()
    with ctx:
        t0 = time.perf_counter()
        await asyncio.gather(*(agent() for _ in range(concurrency)))
        wall_s = time.perf_counter() - t0
    await client.close()

    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    out_tokens = sum(r.output_tokens for r in ok)
    all_itls = [x for r in ok for x in r.itls_s]

    record = {
        "concurrency": concurrency,
        "requests": len(results),
        "failed": len(failed),
        "wall_s": round(wall_s, 3),
        "output_tokens": out_tokens,
        # System-level: rises with concurrency until the GPU saturates.
        "output_tok_s": round(out_tokens / wall_s, 2) if wall_s else 0.0,
        # Per-agent experience: degrades with concurrency.
        "ttft_s_mean": _r(_mean([r.ttft_s for r in ok if r.ttft_s])),
        "ttft_s_p95": _r(_pct([r.ttft_s for r in ok if r.ttft_s], 95)),
        "itl_s_mean": _r(_mean(all_itls), 5),
        "itl_s_p95": _r(_pct(all_itls, 95), 5),
        "per_agent_tok_s": _r(1.0 / _mean(all_itls)) if all_itls else None,
        "latency_s_mean": _r(_mean([r.latency_s for r in ok if r.latency_s])),
        "workload": asdict(wl),
    }
    if mon:
        record["gpu"] = mon.samples.summary()
        if mon.error:
            record["gpu_error"] = mon.error
    if failed:
        record["first_error"] = failed[0].error
    return record


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return None


def _mean(xs: list) -> float:
    return statistics.fmean(xs) if xs else 0.0


def _pct(xs: list, p: int) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


def _r(x: float | None, nd: int = 3) -> float | None:
    return None if x is None else round(x, nd)


async def sweep(
    base_url: str,
    model: str,
    concurrency_levels: list[int],
    wl: Workload,
    label: str,
    out_path: str | None = None,
    requests_per_agent: int = 3,
    metadata: dict | None = None,
) -> list[dict]:
    """Walk the concurrency axis for one quantization rung."""
    rows = []
    for c in concurrency_levels:
        row = await run_concurrency_level(base_url, model, c, wl, requests_per_agent)
        row.update(metadata or {})
        row["label"] = label
        row["model"] = model
        rows.append(row)
        print(
            f"  c={c:<3} {row['output_tok_s']:>8.1f} tok/s total | "
            f"{row['per_agent_tok_s'] or 0:>6.1f} tok/s per-agent | "
            f"ttft {row['ttft_s_mean']}s | "
            f"sm {row.get('gpu', {}).get('sm_util_pct_mean', '?')}% "
            f"mem {row.get('gpu', {}).get('mem_util_pct_mean', '?')}%"
            + (f" | {row['failed']} FAILED" if row["failed"] else ""),
            flush=True,
        )
        if out_path:
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
    return rows
