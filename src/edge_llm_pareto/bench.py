"""Client-observed latency and throughput, raw requests and synchronized energy.

Legacy itl_s/per_agent_tok_s fields describe content chunks, not exact tokens.
Finite request pools include startup and drain effects; TTFT includes queueing.
"""

from __future__ import annotations

import asyncio
import json
import hashlib
import random
from pathlib import Path
import statistics
import time
from contextlib import nullcontext

from .telemetry import PowerMonitor
from .gsmm import power_record, measure_reference, adjusted_energy
from .results import read_records


def make_workload(prompt_tokens=2000, max_output_tokens=256, temperature=0.0):
    """Build the settings dictionary used by requests and saved results."""
    if max_output_tokens is not None and (not isinstance(max_output_tokens, int) or
                                          isinstance(max_output_tokens, bool) or max_output_tokens <= 0):
        raise ValueError("max_output_tokens must be a positive integer or null")
    return dict(prompt_tokens=prompt_tokens, max_output_tokens=max_output_tokens,
                output_limit_policy="explicit_cap_or_eos_or_context" if max_output_tokens is not None else "eos_or_context",
                temperature=temperature)


def make_prompt(workload, seed):
    """Approximate the requested length using the Qwen3 filler calibration."""
    if workload.get("prompts"):
        return workload["prompts"][seed % len(workload["prompts"])]["prompt"]
    # A distinct prefix prevents accidental prefix-cache hits between requests.
    head = f"[request {seed:06d}] "
    repeats = max(round(max(workload["prompt_tokens"] - 18, 0) / 10.0), 1)
    filler = "The quick brown fox jumps over the lazy dog. " * repeats
    return head + filler + "\n\nSummarize the text above in one paragraph."


def request_result(ok, request_id=0, input_tokens=None, token_count_source="chunks",
                   ttft_s=None, latency_s=None, output_tokens=0, itls_s=None, error=None, finish_reason=None,
                   raw_answer='', raw_reasoning=''):
    """Return a request record with consistent defaults, including on failure."""
    return dict(ok=ok, request_id=request_id, input_tokens=input_tokens,
                token_count_source=token_count_source, ttft_s=ttft_s,
                latency_s=latency_s, output_tokens=output_tokens,
                itls_s=[] if itls_s is None else itls_s, error=error,
                finish_reason=finish_reason, raw_answer=raw_answer, raw_reasoning=raw_reasoning)


async def _one_request(client, model: str, wl: dict, seed: int) -> dict:
    t0 = time.perf_counter()
    ttft: float | None = None
    last = t0
    itls: list[float] = []
    n = 0
    finish_reason = None
    content_parts, reasoning_parts = [], []
    try:
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": make_prompt(wl, seed)}],
            temperature=wl["temperature"],
            stream=True,
        )
        if wl.get("max_output_tokens") is not None:
            kwargs["max_tokens"] = wl["max_output_tokens"]
        try:
            # Server-reported token counts are authoritative; llama-server has
            # not always accepted this option, so degrade to counting chunks
            # rather than failing the request.
            stream = await client.chat.completions.create(
                **kwargs, stream_options={"include_usage": True}
            )
        except Exception as exc:
            if getattr(exc, "status_code", None) not in (400, 422) or "stream_options" not in str(exc):
                raise
            stream = await client.chat.completions.create(**kwargs)
        usage_tokens = None
        input_tokens = None
        async for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                usage_tokens = chunk.usage.completion_tokens
                input_tokens = getattr(chunk.usage, "prompt_tokens", None)
            if not chunk.choices:
                continue
            finish_reason = getattr(chunk.choices[0], "finish_reason", None) or finish_reason
            delta = chunk.choices[0].delta
            reasoning = (getattr(delta, 'reasoning', None) or
                         getattr(delta, 'reasoning_content', None)) if delta else None
            if isinstance(reasoning, str) and reasoning:
                reasoning_parts.append(reasoning)
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
            content_parts.append(text)
        total = time.perf_counter() - t0
        return request_result(
            ok=n > 0 and (usage_tokens is None or usage_tokens > 0),
            request_id=seed, input_tokens=input_tokens,
            token_count_source="server" if usage_tokens is not None else "chunks",
            ttft_s=ttft,
            latency_s=total,
            # Prefer server-reported usage; chunk count is a fallback and can
            # differ when a chunk carries multiple tokens.
            output_tokens=usage_tokens if usage_tokens is not None else n,
            itls_s=itls, finish_reason=finish_reason,
            raw_answer=''.join(content_parts), raw_reasoning=''.join(reasoning_parts),
        )
    except Exception as e:
        return request_result(ok=False, request_id=seed, error=f"{type(e).__name__}: {e}",
            latency_s=time.perf_counter() - t0, finish_reason=finish_reason,
            raw_answer=''.join(content_parts), raw_reasoning=''.join(reasoning_parts))


async def run_concurrency_level(
    base_url: str,
    model: str,
    concurrency: int,
    wl: dict,
    requests_per_agent: int = 3,
    api_key: str = "EMPTY",
    monitor_gpu: bool = True,
    warmup_requests: int = 0,
    idle_seconds: float = 0,
    telemetry_interval: float = .1,
    system_baseline: dict | None = None,
    reference_path: str | None = None,
    request_offset: int = 0,
) -> dict:
    """Run a finite closed-loop request pool, including startup and drain.

    `request_offset` shifts request ids so successive trials continue through a
    prompt/QA dataset instead of repeating its first `concurrency * requests_per_agent` items.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=600.0, max_retries=0)
    for seed in range(warmup_requests):
        warmup = await _one_request(client, model, wl, -seed - 1)
        if not warmup["ok"]:
            await client.close()
            raise RuntimeError(f"Warmup failed: {warmup['error']}")
    total_requests = concurrency * requests_per_agent
    counter = iter(range(request_offset, request_offset + total_requests))
    lock = asyncio.Lock()
    results: list[dict] = []

    async def agent() -> None:
        """One simulated agent: issues requests back-to-back, keeping exactly
        one in flight, so `concurrency` is the true in-flight request count."""
        while True:
            async with lock:
                seed = next(counter, None)
            if seed is None:
                return
            results.append(await _one_request(client, model, wl, seed))

    power = (PowerMonitor() if telemetry_interval == .1 else
             PowerMonitor(interval=telemetry_interval)) if monitor_gpu else None
    idle = None
    try:
        if idle_seconds:
            if not monitor_gpu:
                raise ValueError("Idle measurement requires power monitoring")
            idle = await measure_reference('server_idle', idle_seconds, telemetry_interval)
            if reference_path:
                with open(reference_path, 'a') as f:
                    f.write(json.dumps(dict(idle, model=model)) + '\n')
        with power if power else nullcontext():
            t0 = time.perf_counter()
            await asyncio.gather(*(agent() for _ in range(concurrency)))
            t1 = time.perf_counter()
            wall_s = t1 - t0
    finally:
        await client.close()

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    out_tokens = sum(r["output_tokens"] for r in ok)
    all_itls = [x for r in ok for x in r["itls_s"]]
    user_rates = [r["output_tokens"] / r["latency_s"] for r in ok
                  if r["token_count_source"] == "server" and r["latency_s"] > 0]
    for r in results:
        r["user_output_tok_s"] = (r["output_tokens"] / r["latency_s"]
            if r["ok"] and r["token_count_source"] == "server" and r["latency_s"] > 0 else None)

    record = {
        "input_tokens": sum(r["input_tokens"] for r in ok) if ok and all(r["input_tokens"] is not None for r in ok) else None,
        "latency_definition": "client request including queueing; ITL legacy fields measure content chunks",
        "schema_version": 2,
        "request_results": [dict(r, chunk_latencies_s=r["itls_s"]) for r in results],
        "token_counts_exact": all(r["token_count_source"] == "server" for r in ok) and bool(ok),
        "latency_s_p95": _pct([r["latency_s"] for r in ok if r["latency_s"] is not None], 95) if ok else None,
        "chunk_latency_s_mean": _mean(all_itls) if all_itls else None,
        "chunk_latency_s_p95": _pct(all_itls, 95) if all_itls else None,
        "concurrency": concurrency,
        "requests": len(results),
        "length_limited_requests": sum(r.get("finish_reason") == "length" for r in results),
        "failed": len(failed),
        "wall_s": round(wall_s, 3),
        "output_tokens": out_tokens,
        # System-level: rises with concurrency until the GPU saturates.
        "output_tok_s": round(out_tokens / wall_s, 2) if wall_s else 0.0,
        # Per-request delivered tokens/s includes prefill and queueing. Never
        # substitute stream-chunk counts or aggregate throughput / concurrency.
        "user_output_tok_s_mean": statistics.fmean(user_rates) if user_rates else None,
        "user_output_tok_s_p50": _pct(user_rates, 50) if user_rates else None,
        "user_output_tok_s_p05": _pct(user_rates, 5) if user_rates else None,
        # Per-agent experience: degrades with concurrency.
        "ttft_s_mean": _r(_mean([r["ttft_s"] for r in ok if r["ttft_s"]])),
        "ttft_s_p95": _r(_pct([r["ttft_s"] for r in ok if r["ttft_s"]], 95)),
        "itl_s_mean": _r(_mean(all_itls), 5),
        "itl_s_p95": _r(_pct(all_itls, 95), 5),
        "per_agent_tok_s": _r(1.0 / _mean(all_itls)) if all_itls else None,
        "latency_s_mean": _r(_mean([r["latency_s"] for r in ok if r["latency_s"]])),
        "workload": dict(wl),
    }
    if power:
        record.update(power_record(power, t0, t1))
        record['scenario'] = 'generation_load'
        record['gsmm_references'] = {
            name: {k: v for k, v in reference.items() if k != 'power_samples'} if reference else None
            for name, reference in [('system_baseline', system_baseline), ('server_idle', idle)]}
        record['gsmm_reference_artifact'] = reference_path
        # Without an artifact file, retain raw references in the returned record.
        if not reference_path:
            record['gsmm_references'] = dict(system_baseline=system_baseline, server_idle=idle)
        for name, reference in [('system_baseline', system_baseline), ('server_idle', idle)]:
            adjusted = adjusted_energy(record, reference, name)
            record[name + '_adjustment'] = adjusted
            record[name + '_adjusted_energy_j'] = adjusted['energy_j']
            record[name + '_adjusted_joules_per_output_token'] = adjusted['joules_per_output_token']
        energy = record.get("run_energy_j")
        record["joules_per_output_token"] = (energy / out_tokens if energy is not None
            and out_tokens and record["token_counts_exact"] and not failed else None)
    if power and power.scope == "nvml_device":
        record["gpu"] = power.summary()
        if power.errors:
            record["gpu_error"] = "; ".join(sorted(set(power.errors)))
    if failed:
        record["first_error"] = failed[0]["error"]
    if wl.get('evaluation_mode') == 'measured_qa':
        # Both the power monitor and request timer have stopped. Scoring does
        # not add model calls or enter the generation energy window.
        from .quality import score_measured_requests
        record['quality'] = score_measured_requests(record['request_results'], wl)
        record['qa_score'] = record['quality']['score']
        record['qa_exact_match'] = record['quality']['metrics']['exact_match']
        record['qa_bleu'] = record['quality']['metrics']['bleu']['score']
        record['qa_incomplete_reasoning'] = record['quality']['incomplete_reasoning']
    return record


def _mean(xs: list) -> float:
    return statistics.fmean(xs) if xs else 0.0


def _pct(xs: list, p: int) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


def _r(x: float | None, nd: int = 3) -> float | None:
    return None if x is None else round(x, nd)


def stable_id(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:20]


async def sweep(base_url, model, concurrency_levels, wl, label, out_path=None,
                requests_per_agent=3, metadata=None, repeats=1, warmup_requests=0,
                seed=0, resume=False, workloads=None, idle_seconds=0,
                telemetry_interval=.1, system_baseline=None, reference_path=None):
    metadata = metadata or {}
    completed = set()
    if resume and out_path and Path(out_path).exists():
        for row in read_records(out_path, repair=True):
            if (not row.get("failed") and not row.get("unexpected_thermal_throttling")
                and not row.get("throttled") and row.get("energy_valid")
                and row.get("token_counts_exact") and row.get("output_tok_s", 0) > 0):
                completed.add(row.get("trial_id"))
    jobs = [(w, c, trial) for w in (workloads or [wl])
            for c in concurrency_levels for trial in range(repeats)]
    random.Random(seed).shuffle(jobs)
    rows = []
    for w, c, trial in jobs:
        identity = {k: v for k, v in metadata.items()
                    if k not in ("hardware", "kv_ceiling_agents", "quality", "environment", "session_observations")}
        hardware = metadata.get("hardware", {})
        identity["device"] = {k: hardware.get(k) for k in ("name", "total_mib", "compute_cap")}
        identity.update(model=model, label=label, workload=dict(w), concurrency=c,
                        requests_per_agent=requests_per_agent, warmup_requests=warmup_requests, seed=seed)
        identity['measurement_protocol'] = dict(idle_seconds=idle_seconds,
            baseline_seconds=system_baseline.get('requested_duration_s', 0) if system_baseline else 0,
            telemetry_interval_s=telemetry_interval, energy_basis='gross',
            adjustment_method='reference_mean_power_times_duration_v1')
        config_id = stable_id(identity)
        trial_id = stable_id([config_id, trial])
        if trial_id in completed:
            continue
        row = await run_concurrency_level(base_url, model, c, w, requests_per_agent,
                                          warmup_requests=warmup_requests, idle_seconds=idle_seconds,
                                          telemetry_interval=telemetry_interval, system_baseline=system_baseline,
                                          reference_path=reference_path,
                                          request_offset=trial * c * requests_per_agent)
        row = dict(row)
        measured_quality = row.get('quality') if w.get('evaluation_mode') == 'measured_qa' else None
        row.update(metadata)
        if measured_quality is not None:
            row['quality'] = measured_quality
        row.update(label=label, model=model, config_id=config_id, trial_id=trial_id,
                   trial=trial, seed=seed, protocol=identity)
        rows.append(row)
        print(f"  {label} c={c} trial={trial}: {row['output_tok_s']} tok/s, "
              f"J/token={row.get('joules_per_output_token')}", flush=True)
        if out_path:
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
    return rows
