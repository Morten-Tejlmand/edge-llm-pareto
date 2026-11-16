#!/usr/bin/env python3
"""Run the quantization x concurrency sweep for one model.

For each rung: start a server, walk the concurrency axis, tear it down. Rungs
that fail (OOM, unsupported kernel, missing GGUF) are recorded and skipped
rather than aborting the sweep. Kernel support must be validated on GB10.

    python scripts/sweep.py --model qwen3-1.7b --rung gguf-q4_k_m --concurrency 1,2,4
    python scripts/sweep.py --rung all --concurrency all --requests-per-agent 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import platform
from types import SimpleNamespace
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from edge_llm_pareto.telemetry import power_mode
from edge_llm_pareto.results import read_records
from edge_llm_pareto import serve  # noqa: E402
from edge_llm_pareto.bench import make_workload, sweep, stable_id
from edge_llm_pareto.gsmm import environment_metadata, measure_reference  # noqa: E402
from edge_llm_pareto.hardware import (  # noqa: E402
    ModelShape, budget, check_benchmark_conditions, probe_gpu, probe_power_state,
    add_memory_arguments, memory_options,
)


def matching_quality(quality, args, rung, revision):
    if not quality:
        return None
    expected = dict(base_model=args.model, engine=args.engine, label=rung['rung'],
                    model_repo=rung['repo'], model_revision=revision)
    if any(quality.get(k) != v for k, v in expected.items()):
        return None
    quality_limit = getattr(args, "quality_max_tokens", None)
    policy = "explicit_cap_or_eos_or_context" if quality_limit is not None else "eos_or_context"
    if quality.get("max_tokens") != quality_limit or quality.get("output_limit_policy") != policy:
        return None
    scorer = quality.get('scorer', '')
    if quality.get('failed') or not (scorer in ('normalized_exact_match_v1', 'normalized_exact_match_v2|leading_think_blocks_v1') or
                                    scorer.startswith('sacrebleu_corpus_v1|')):
        raise ValueError('Quality results contain failures or unsupported scoring')
    return quality


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/models.yaml"))
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--engine", choices=["llama_cpp", "vllm"], default="llama_cpp")
    ap.add_argument("--rung", default=None,
                    help="precision label or comma-separated labels; all runs every variant")
    ap.add_argument("--llama-server", default="llama-server", help="path to the CUDA llama-server binary")
    ap.add_argument("--vllm", default="vllm", help="path to vLLM executable (separate runtime environment)")
    ap.add_argument("--model-revision", default="main", help="Hugging Face revision for model downloads")
    ap.add_argument("--concurrency", default="1", help="comma list; default 1; all uses config levels")
    ap.add_argument("--server-parallel", type=int, help="fixed server slots; fail if estimated not to fit")
    ap.add_argument("--qa-dataset", help="QA JSONL used for concurrent measured requests; save and score those answers")
    ap.add_argument("--prompt-dataset", help="fixed JSONL prompts with unique id and prompt fields")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--prompt-tokens", type=int, default=2000)
    ap.add_argument("--output-tokens", type=int, default=256, help="performance output-token cap; default 256")
    ap.add_argument("--requests-per-agent", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--warmup-requests", type=int, default=0)
    ap.add_argument("--gsmm", action="store_true", help="collect 30 s system-baseline and per-trial server-idle windows unless overridden")
    ap.add_argument("--baseline-seconds", type=float, default=None, help="system baseline per variant before launch; 0 disables")
    ap.add_argument("--idle-seconds", type=float, default=None, help="server idle after warmup before each trial; 0 disables")
    ap.add_argument("--sample-interval", type=float, default=.1, help="requested power sampling interval in seconds")
    ap.add_argument("--ambient-temperature-c", type=float, help="operator-observed ambient temperature")
    ap.add_argument("--sensor-accuracy-note", help="sensor accuracy specification and source, if known")
    ap.add_argument("--sensor-calibration-note", help="calibration procedure/date and evidence, if available")
    ap.add_argument("--runtime-description", help="separate server stack versions or pinned container digest")
    ap.add_argument("--session-note", help="operating conditions and external activity observations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workloads", help="YAML list of prompt_tokens/max_output_tokens/temperature mappings; default output cap 256")
    ap.add_argument("--resume", help="append to this existing result JSONL, skipping successful trial IDs")
    ap.add_argument("--target", choices=["gb10", "jetson"], default="gb10")
    ap.add_argument("--power-setting", default="default", help="label for externally configured power/clock setting")
    ap.add_argument("--intentional-power-limit", action="store_true")
    ap.add_argument("--quality-dataset", help="evaluate this JSONL on each launched variant before benchmarking")
    ap.add_argument('--quality-max-tokens', type=int, default=None, help='independent QA output cap; default uncapped within context')
    ap.add_argument('--quality-scorer', choices=['exact-match', 'bleu'], default='exact-match')
    ap.add_argument("--quality-results", help="JSON produced by evaluate.py for these exact variants")
    add_memory_arguments(ap)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--allow-throttled", action="store_true",
                    help="run even if the GPU is power-capped (results will not be comparable)")
    args = ap.parse_args()
    args.baseline_seconds = args.baseline_seconds if args.baseline_seconds is not None else (30. if args.gsmm else 0.)
    args.idle_seconds = args.idle_seconds if args.idle_seconds is not None else (30. if args.gsmm else 0.)
    if any(not math.isfinite(v) or v < 0 for v in (args.baseline_seconds, args.idle_seconds)):
        ap.error("baseline and idle durations must be finite and nonnegative")
    if not math.isfinite(args.sample_interval) or args.sample_interval <= 0:
        ap.error("sample interval must be finite and positive")
    if args.ambient_temperature_c is not None and not math.isfinite(args.ambient_temperature_c):
        ap.error("ambient temperature must be finite")
    if args.qa_dataset and (args.prompt_dataset or args.quality_dataset or args.quality_results):
        ap.error("--qa-dataset replaces --prompt-dataset and separate --quality-dataset/--quality-results")
    if args.qa_dataset and args.quality_max_tokens is not None:
        ap.error("measured QA uses --output-tokens; --quality-max-tokens is for separate QA evaluation")
    if args.quality_dataset and args.quality_results:
        ap.error("choose --quality-dataset or --quality-results")
    if args.repeats < 1 or args.warmup_requests < 0:
        ap.error("repeats must be positive and warmup requests nonnegative")
    if args.target == "jetson" and not all(any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv) for flag in ("--host-reserve-gib", "--runtime-overhead-gib")):
        ap.error("Jetson requires explicit --host-reserve-gib and --runtime-overhead-gib calibrated for your device")
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.model not in cfg["models"]:
        ap.error(f"unknown model: {args.model}")
    model_cfg = cfg["models"][args.model]
    engine_cfg = model_cfg["engine_rungs"].get(args.engine)
    if not engine_cfg:
        ap.error(f"no {args.engine} variants for {args.model}")
    args.rung = args.rung or engine_cfg["default_rung"]
    rungs = [dict(r, repo=r.get("repo", engine_cfg.get("repo")), engine=args.engine)
             for r in engine_cfg["rungs"] if args.rung == "all" or r["rung"] in args.rung.split(",")]
    if args.rung != "all" and set(args.rung.split(",")) - {r["rung"] for r in rungs}:
        ap.error("unknown rung in selection")
    if args.server_parallel is not None and args.server_parallel < 1:
        ap.error("server-parallel must be positive")
    if not rungs:
        ap.error(f"no such {args.engine} rung: {args.rung}")

    try:
        mem_options = memory_options(args)
        levels = ([int(x) for x in args.concurrency.split(",")]
                  if args.concurrency != "all" else cfg["concurrency"])
        if not levels or min(levels) <= 0:
            raise ValueError("concurrency levels must be positive")
        if cfg["defaults"]["kv_dtype_bytes"] != 2:
            raise ValueError("Both backends use 16-bit KV; kv_dtype_bytes must be 2")
        if args.quality_max_tokens is not None and args.quality_max_tokens <= 0:
            raise ValueError("quality-max-tokens must be positive")
        if min(args.context, args.prompt_tokens, args.output_tokens,
               args.requests_per_agent) <= 0:
            raise ValueError("context, prompt tokens and requests must be positive")
        if not (args.prompt_dataset or args.qa_dataset) and args.prompt_tokens + args.output_tokens + 64 > args.context:
            raise ValueError("context must leave generation space after prompt and template tokens")
    except ValueError as exc:
        ap.error(str(exc))
    workloads = None
    if args.workloads:
        try:
            workloads = [make_workload(**w) for w in yaml.safe_load(Path(args.workloads).read_text())]
            if not workloads or any(w["prompt_tokens"] <= 0 or
                (not (args.prompt_dataset or args.qa_dataset) and w["prompt_tokens"] + (w["max_output_tokens"] or 1) + 64 > args.context) for w in workloads):
                raise ValueError("workloads must be positive and fit the context")
        except (TypeError, ValueError) as exc:
            ap.error(str(exc))
    random.Random(args.seed).shuffle(rungs)
    quality = json.loads(Path(args.quality_results).read_text()) if args.quality_results else None
    code_revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   capture_output=True, text=True).stdout.strip()
    import hashlib
    code_hash = hashlib.sha256()
    for source in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py")):
        code_hash.update(source.read_bytes())
    code_fingerprint = code_hash.hexdigest()
    try:
        executable = args.llama_server if args.engine == "llama_cpp" else args.vllm
        executable = serve.resolve_executable(executable)
        llama_version = serve.executable_version(executable)
        if args.engine == "llama_cpp":
            args.llama_server = executable
        else:
            args.vllm = executable
    except (OSError, subprocess.SubprocessError) as exc:
        ap.error(f"Cannot run {args.engine}: {exc}. See docs/engines.md")
    gpu = probe_gpu()
    if not gpu.unified_memory or (args.target == "gb10" and "GB10" not in gpu.name.upper()) or (args.target == "jetson" and "jetson" not in gpu.name.lower()):
        ap.error(f"This sweep targets {args.target} unified memory; detected {gpu.name}")

    ps = probe_power_state()
    problems = check_benchmark_conditions(ps, intentional_power_limit=args.intentional_power_limit)
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

    if model_cfg.get("budget_note"):
        print(f"Memory estimate: {model_cfg['budget_note']}")
    shape = ModelShape.from_hf(model_cfg["shape_from"])
    wl = make_workload(prompt_tokens=args.prompt_tokens, max_output_tokens=args.output_tokens)

    if args.prompt_dataset or args.qa_dataset:
        from edge_llm_pareto.workloads import attach_prompts, attach_qa
        try:
            def attach(workload):
                return (attach_qa(workload, args.qa_dataset, args.quality_scorer) if args.qa_dataset
                        else attach_prompts(workload, args.prompt_dataset))
            wl = attach(wl)
            if workloads:
                workloads = [attach(w) for w in workloads]
            if args.qa_dataset and args.requests_per_agent % len(wl['prompts']):
                print("QA coverage: requests-per-agent is not a multiple of the dataset size; "
                      "inspect unique_questions and dataset_questions in each trial.", flush=True)
        except (ValueError, OSError) as exc:
            ap.error(str(exc))

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = results_dir / f"{args.model}-{args.engine}-{stamp}.jsonl"
    if args.resume:
        out_path = Path(args.resume)
        if not out_path.is_file():
            ap.error("--resume must name an existing JSONL file")
    environment = environment_metadata()
    environment_id = stable_id(dict(software={k: v for k, v in environment.items() if k != 'captured_at_utc'},
        runtime_description=args.runtime_description))
    session_observations = dict(ambient_temperature_c=args.ambient_temperature_c,
        sensor_accuracy_note=args.sensor_accuracy_note,
        sensor_calibration_note=args.sensor_calibration_note,
        runtime_description=args.runtime_description, session_note=args.session_note,
        validation_status='operator declarations; sensor scope and stability require Spark validation')
    measurement_protocol = dict(baseline_seconds=args.baseline_seconds,
        idle_seconds=args.idle_seconds, telemetry_interval_s=args.sample_interval,
        baseline_frequency='before_each_variant_launch', idle_frequency='after_warmup_before_each_trial',
        energy_basis='gross', framework='GSMM-informed',
        reference_doi='10.1016/j.future.2024.01.033')
    reference_path = None
    if args.baseline_seconds or args.idle_seconds:
        reference_dir = results_dir / 'scenarios'
        reference_dir.mkdir(parents=True, exist_ok=True)
        reference_path = str(reference_dir / f"{args.model}-{args.engine}-{stamp}.jsonl")
    meta_path = results_dir / f"{args.model}-{args.engine}-{stamp}.meta.json"
    meta_path.write_text(json.dumps({
        "model": args.model,
        "environment": environment, "environment_id": environment_id,
        "session_observations": session_observations,
        "measurement_protocol": measurement_protocol,
        "reference_artifact": reference_path,
        "engine": args.engine,
        "gpu": asdict(gpu),
        "memory_budget": mem_options,
        "params": shape.total_params,
        "kv_bytes_per_token": shape.kv_bytes_per_token(),
        "context": args.context,
        "concurrency_levels": levels,
        "workload": {"prompt_tokens": args.prompt_tokens, "max_output_tokens": args.output_tokens,
                     "output_limit_policy": "explicit_cap_or_eos_or_context"},
        "quality_max_tokens": args.quality_max_tokens,
        "qa_dataset": args.qa_dataset,
        "qa_dataset_sha256": wl.get('qa_dataset_sha256'),
        "evaluation_mode": wl.get('evaluation_mode'),
        "backend_version": llama_version,
        "server_executable": executable,
        "power_state": {
            "on_ac": ps.on_ac, "power_w": ps.power_w,
            "sm_clock_mhz": ps.sm_clock_mhz, "sm_clock_max_mhz": ps.sm_clock_max_mhz,
            "throttle_reasons": ps.throttle_reasons,
        },
        "throttled": bool(problems),
    }, indent=2))
    print(f"results -> {out_path}\n")

    successful_points = 0
    if args.resume:
        successful_points = sum(not row.get('failed') and not row.get('skipped') and row.get('output_tok_s', 0) > 0
            for row in read_records(out_path, repair=True))
    for r in rungs:
        print(f"Resolving {r['repo']} {r.get('file', 'native weights')} (cached downloads are reused)", flush=True)
        try:
            gguf = (serve.download_gguf(r["repo"], r["file"], args.model_revision)
                    if r.get("file") else None)
            resolved_revision = (gguf["revision"] if gguf else
                                 serve.resolve_hf_revision(r["repo"], args.model_revision))
            tokenizer_repo = r.get("tokenizer", model_cfg["shape_from"])
            tokenizer_revision = (serve.resolve_hf_revision(tokenizer_repo)
                                  if args.engine == "vllm" and gguf else None)
        except Exception as exc:
            print(f"   SKIP: Model resolution failed: {exc}", flush=True)
            _record_skip(out_path, r, f"{'gguf_download_failed' if r.get('file') else 'model_resolution_failed'}: {exc}")
            continue
        variant_quality = matching_quality(quality, args, r, resolved_revision)
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

        # Fixed-size slots: allocate only as many as the estimate permits.
        # Larger client concurrency queues at the server. No hard RAM cap.
        parallel = args.server_parallel or min(max(levels), ceiling)
        if parallel > ceiling:
            raise ValueError(f"Requested {parallel} server slots exceed estimated capacity {ceiling}")
        served_model = f"{args.model}-{r['rung']}"
        if args.engine == "llama_cpp":
            cmd = serve.llama_server_command(
                gguf["path"], executable=args.llama_server, alias=served_model,
                ctx_size=args.context, parallel=parallel, port=args.port,
            )
        else:
            # Limit the KV reservation using current host availability, not only
            # dedicated-VRAM counters. Runtime overhead remains an estimate.
            cmd = serve.vllm_server_command(
                gguf["path"] if gguf else r["repo"], executable=args.vllm,
                revision=resolved_revision, alias=served_model,
                ctx_size=args.context, parallel=parallel, port=args.port,
                gpu_memory_utilization=args.gpu_mem_util,
                kv_cache_memory_bytes=b.kv_mib * 1024 * 1024,
                dtype=r.get("dtype", "bfloat16"),
                quantization="gguf" if gguf else r.get("quantization"),
                tokenizer=tokenizer_repo if gguf else None,
                tokenizer_revision=tokenizer_revision,
            )
        log_path = str(results_dir / f"serve-{args.model}-{args.engine}-{r['rung']}-{stamp}.log")
        print(f"   $ {' '.join(cmd)}")
        system_baseline = None
        if args.baseline_seconds:
            print(f"   measuring system baseline for {args.baseline_seconds:g}s (server stopped)", flush=True)
            system_baseline = asyncio.run(measure_reference('system_baseline',
                args.baseline_seconds, args.sample_interval))
            system_baseline.update(base_model=args.model, engine=args.engine, label=r['rung'],
                environment_id=environment_id, session_observations=session_observations)
            with open(reference_path, 'a') as f:
                f.write(json.dumps(system_baseline) + "\n")
        try:
            h = serve.start(cmd, log_path, port=args.port, env=dict(os.environ))
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
            if args.quality_dataset:
                from edge_llm_pareto.quality import evaluate
                quality_path = results_dir / f"{args.model}-{args.engine}-{r['rung']}-{stamp}.quality.json"
                quality_args = SimpleNamespace(dataset=args.quality_dataset, base_url=h.base_url,
                    model=served_model, model_repo=r['repo'], model_revision=resolved_revision,
                    base_model=args.model, engine=args.engine, rung=r['rung'], max_tokens=args.quality_max_tokens,
                    output=str(quality_path), scorer=args.quality_scorer)
                if asyncio.run(evaluate(quality_args)):
                    _record_skip(out_path, r, 'quality_evaluation_failed')
                    continue
                variant_quality = json.loads(quality_path.read_text())
            rows = asyncio.run(sweep(
                h.base_url, served_model, levels, wl,
                label=r["rung"], out_path=str(out_path),
                requests_per_agent=args.requests_per_agent,
                repeats=args.repeats, warmup_requests=args.warmup_requests,
                seed=args.seed, resume=bool(args.resume), workloads=workloads,
                idle_seconds=args.idle_seconds, telemetry_interval=args.sample_interval,
                system_baseline=system_baseline, reference_path=reference_path,

                metadata={"device_id": platform.node(), "code_revision": code_revision, "code_fingerprint": code_fingerprint,
                          "environment_id": environment_id, "environment": environment,
                          "session_observations": session_observations,
                          "measurement_protocol": measurement_protocol,
                          "model_revision": resolved_revision,
                          "backend_version": llama_version,
                          "power_setting": args.power_setting, "power_mode": power_mode(),
                          "power_limit_w": ps.power_limit_w,
                          "intentional_power_limit": args.intentional_power_limit,
                          "quality": variant_quality,
                          "kv_ceiling_agents": ceiling, "throttled": bool(problems),
                          "engine": args.engine, "base_model": args.model,
                          "context": args.context, "server_command": cmd,
                          "model_repo": r["repo"], "gguf": gguf,
                          "budget_note": model_cfg.get("budget_note"),
                          "llama_cpp_version": llama_version if args.engine == "llama_cpp" else None,
                          "quantization": r.get("quantization", "gguf" if gguf else None),
                          "dtype": r.get("dtype", "bfloat16") if args.engine == "vllm" else "float16",
                          "tokenizer_repo": tokenizer_repo if args.engine == "vllm" and gguf else None,
                          "tokenizer_revision": tokenizer_revision,
                          "server_parallel": parallel,
                          "gpu_memory_utilization": args.gpu_mem_util if args.engine == "vllm" else None,
                          "memory_policy": "explicit_kv_bytes" if args.engine == "vllm" else "estimated_slots",
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


def _record_skip(out_path: Path, rung: dict, reason: str, b=None, log: str | None = None) -> None:
    with open(out_path, "a") as f:
        f.write(json.dumps({
            "label": rung["rung"], "model": rung["repo"], "skipped": True,
            "engine": rung["engine"], "gguf_file": rung.get("file"),
            "reason": reason, "predicted_weights_mib": b.weights_mib if b else None,
            "predicted_kv_mib": b.kv_mib if b else None, "serve_log": log,
        }) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
