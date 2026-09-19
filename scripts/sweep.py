#!/usr/bin/env python3
"""Run the quantization x concurrency sweep for one model.

For each rung: start a server, walk the concurrency axis, tear it down. Rungs
that fail (OOM, unsupported kernel, missing plugin) are recorded and skipped
rather than aborting the sweep. Kernel support must be validated on GB10.

    python scripts/sweep.py --model qwen3-1.7b --rung bf16 --concurrency 1,2,4
    python scripts/sweep.py --rung all --concurrency all --requests-per-agent 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from edge_llm_pareto import serve  # noqa: E402
from edge_llm_pareto.bench import Workload, sweep  # noqa: E402
from edge_llm_pareto.hardware import (  # noqa: E402
    ModelShape, budget, check_benchmark_conditions, probe_gpu, probe_power_state,
    add_memory_arguments, memory_options,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/models.yaml"))
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--rung", default="bf16", help="precision variant; default bf16; all runs every variant")
    ap.add_argument("--concurrency", default="1", help="comma list; default 1; all uses config levels")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--prompt-tokens", type=int, default=2000)
    ap.add_argument("--output-tokens", type=int, default=256)
    ap.add_argument("--requests-per-agent", type=int, default=1)
    add_memory_arguments(ap)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--enforce-eager", action="store_true",
                    help="skip CUDA graphs: frees VRAM but slows decode; taints comparisons")
    ap.add_argument("--allow-throttled", action="store_true",
                    help="run even if the GPU is power-capped (results will not be comparable)")
    args = ap.parse_args()

    try:
        mem_options = memory_options(args)
        if min(args.context, args.prompt_tokens, args.output_tokens,
               args.requests_per_agent) <= 0:
            raise ValueError("context, token counts and requests must be positive")
        if args.prompt_tokens + args.output_tokens + 64 > args.context:
            raise ValueError("context must allow prompt + output + 64 template tokens")
    except ValueError as exc:
        ap.error(str(exc))
    gpu = probe_gpu()
    if not gpu.unified_memory:
        ap.error(f"This sweep targets GB10 unified memory; detected {gpu.name}")

    ps = probe_power_state()
    problems = check_benchmark_conditions(ps)
    if problems:
        print("REFUSING TO RUN -- the GPU cannot produce comparable timings:\n")
        for prob in problems:
            print(f"  * {prob}")
        print(f"\n  SM clock {ps.sm_clock_mhz}/{ps.sm_clock_max_mhz} MHz, "
              f"power {ps.power_w} W, on_ac={ps.on_ac}")
        if not args.allow_throttled:
            print("\nPlug in the machine and retry, or pass --allow-throttled to "
                  "record deliberately throttled numbers.")
            return 3
        print("\n--allow-throttled given; continuing. Results are NOT comparable.\n")

    cfg = yaml.safe_load(Path(args.config).read_text())
    model_cfg = cfg["models"][args.model]
    rungs = model_cfg["engine_rungs"]["vllm"]["rungs"]
    if args.rung != "all":
        rungs = [r for r in rungs if r["rung"] == args.rung]
        if not rungs:
            print(f"no such rung: {args.rung}", file=sys.stderr)
            return 2
    levels = (
        [int(x) for x in args.concurrency.split(",")]
        if args.concurrency != "all" else cfg["concurrency"]
    )

    if not levels or min(levels) <= 0:
        ap.error("concurrency levels must be positive")

    shape = ModelShape.from_hf(model_cfg["shape_from"])
    wl = Workload(prompt_tokens=args.prompt_tokens, max_output_tokens=args.output_tokens)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = results_dir / f"{args.model}-vllm-{stamp}.jsonl"
    meta_path = results_dir / f"{args.model}-vllm-{stamp}.meta.json"
    meta_path.write_text(json.dumps({
        "model": args.model,
        "engine": "vllm",
        "gpu": asdict(gpu),
        "memory_budget": mem_options,
        "params": shape.total_params,
        "kv_bytes_per_token": shape.kv_bytes_per_token(),
        "context": args.context,
        "concurrency_levels": levels,
        "workload": {"prompt_tokens": args.prompt_tokens, "output_tokens": args.output_tokens},
        "enforce_eager": args.enforce_eager,
        "vllm_version": _vllm_version(),
        "power_state": {
            "on_ac": ps.on_ac, "power_w": ps.power_w,
            "sm_clock_mhz": ps.sm_clock_mhz, "sm_clock_max_mhz": ps.sm_clock_max_mhz,
            "throttle_reasons": ps.throttle_reasons,
        },
        "throttled": bool(problems),
        # Deviations from vLLM's default execution path, so results stay
        # comparable against runs made on a box without these workarounds.
        "env_overrides": {
            k: v for k, v in serve.server_env().items()
            if k in ("VLLM_USE_FLASHINFER_SAMPLER", "CUDA_HOME")
        },
    }, indent=2))
    print(f"results -> {out_path}\n")

    successful_points = 0
    for r in rungs:
        gpu = probe_gpu()  # Refresh host availability after each server exits.
        b = budget(
            shape, gpu, r["rung"],
            linear_bits=float(r["linear_bits"]),
            embedding_bits=float(cfg["defaults"]["embedding_bits"]),
            kv_dtype_bytes=cfg["defaults"]["kv_dtype_bytes"],
            **mem_options,
        )
        ceiling = b.concurrency_at(args.context)
        print(f"== {r['rung']}  ({r['repo']})")
        print(f"   predicted: {b.weights_mib} MiB weights, {b.kv_mib} MiB KV, "
              f"~{ceiling} agents at {args.context} ctx")
        if not b.fits or ceiling < 1:
            print("   SKIP: predicted not to fit\n")
            _record_skip(out_path, r, "predicted_oom", b)
            continue

        # Above the KV ceiling vLLM queues rather than decoding in parallel, so
        # those points measure admission control, not parallel decode. Still
        # worth recording -- it is what an over-subscribed edge box actually does.
        # Pass the same effective process cap used by the predictor to vLLM.
        effective_util = min(args.gpu_mem_util,
                             max(0, gpu.free_mib - mem_options["host_reserve_mib"])
                             / gpu.total_mib)
        cmd = serve.vllm_command(
            r["repo"],
            quantization=r.get("quantization"),
            max_model_len=args.context,
            max_num_seqs=max(levels),
            gpu_memory_utilization=effective_util,
            enforce_eager=args.enforce_eager,
            port=args.port,
        )
        log_path = str(results_dir / f"serve-{args.model}-{r['rung']}-{stamp}.log")
        print(f"   $ {' '.join(cmd)}")
        try:
            h = serve.start(cmd, log_path, port=args.port)
        except FileNotFoundError as e:
            print(f"   SKIP: {e}\n")
            _record_skip(out_path, r, str(e), b)
            continue

        with h:
            t0 = time.perf_counter()
            if not serve.wait_ready(h):
                print(f"   FAILED to become ready. Log tail:\n{serve.tail_log(h, 15)}\n")
                _record_skip(out_path, r, "server_not_ready", b, log=log_path)
                continue
            print(f"   ready in {time.perf_counter() - t0:.0f}s")
            rows = asyncio.run(sweep(
                h.base_url, r["repo"], levels, wl,
                label=r["rung"], out_path=str(out_path),
                requests_per_agent=args.requests_per_agent,
                metadata={"kv_ceiling_agents": ceiling, "throttled": bool(problems),
                          "gpu_memory_utilization": effective_util,
                          "hardware": asdict(gpu), "memory_budget": mem_options},
            ))
            successful_points += sum(
                not row.get("failed") and row.get("output_tok_s", 0) > 0
                for row in rows
            )
        print()

    print(f"done -> {out_path}")
    if not successful_points:
        print("No successful benchmark points; check the server log and result errors.")
        return 1
    return 0


def _vllm_version() -> str:
    try:
        import vllm

        return vllm.__version__
    except Exception:
        return "unknown"


def _record_skip(out_path: Path, rung: dict, reason: str, b, log: str | None = None) -> None:
    with open(out_path, "a") as f:
        f.write(json.dumps({
            "label": rung["rung"], "model": rung["repo"], "skipped": True,
            "reason": reason, "predicted_weights_mib": b.weights_mib,
            "predicted_kv_mib": b.kv_mib, "serve_log": log,
        }) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
