#!/usr/bin/env python3
"""Predict which quantization rungs fit on this GPU, and at what concurrency.

Run this before any sweep. Loading a model can take minutes and fail with an
opaque OOM; this answers the same question in seconds and tells you the
context length / parallel slot combination each rung can actually support.

    python scripts/budget.py
    python scripts/budget.py --context 8192 --model qwen3-4b-2507
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from edge_llm_pareto.hardware import (  # noqa: E402
    ModelShape, budget, probe_gpu, gb10_preview, add_memory_arguments, memory_options,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/models.yaml"))
    ap.add_argument("--model", default=None, help="model key; default = all")
    ap.add_argument("--context", type=int, default=4096,
                    help="per-agent context length used for the concurrency estimate")
    ap.add_argument("--engine", default="llama_cpp", choices=["llama_cpp", "vllm"])
    add_memory_arguments(ap)
    ap.add_argument("--target", choices=["local", "gb10"], default="local",
                    help="gb10 estimates nominal capacity without probing this machine")
    args = ap.parse_args()

    try:
        mem_options = memory_options(args)
        if args.context <= 0:
            raise ValueError("--context must be positive")
    except ValueError as exc:
        ap.error(str(exc))

    cfg = yaml.safe_load(Path(args.config).read_text())
    gpu = gb10_preview() if args.target == "gb10" else probe_gpu()
    print(f"GPU: {gpu.name}  |  {gpu.total_mib} MiB total, {gpu.used_mib} MiB in use, "
          f"sm_{gpu.compute_cap.replace('.', '')}")
    print(f"Memory source: {gpu.memory_source}; unified={gpu.unified_memory}")
    print("Estimates only; runtime overhead must be validated on GB10.")
    print(f"gpu_memory_utilization {args.gpu_mem_util}; "
          f"per-agent context {args.context} tokens\n")

    keys = [args.model] if args.model else list(cfg["models"])
    rc = 0
    for key in keys:
        m = cfg["models"][key]
        if m.get("budget_note"):
            print(f"Memory estimate: {m['budget_note']}")
        shape = ModelShape.from_hf(m["shape_from"])
        per_tok = shape.kv_bytes_per_token(cfg["defaults"]["kv_dtype_bytes"])
        print(f"== {key}  ({shape.total_params / 1e9:.2f}B params, "
              f"{shape.layers}L, {shape.kv_heads} KV heads, "
              f"KV {per_tok / 1024:.0f} KiB/token) ==")
        print(f"{'rung':<16}{'weights':>9}{'KV free':>10}{'KV tokens':>11}"
              f"{'agents':>8}   verdict")

        spec = m["engine_rungs"].get(args.engine)
        if not spec:
            print("  (no rungs for this engine)\n")
            continue
        for r in spec["rungs"]:
            b = budget(
                shape, gpu, r["rung"],
                linear_bits=float(r["linear_bits"]),
                embedding_bits=float(cfg["defaults"]["embedding_bits"]),
                kv_dtype_bytes=cfg["defaults"]["kv_dtype_bytes"],
                **mem_options,
            )
            agents = b.concurrency_at(args.context)
            if not b.fits:
                verdict = "WILL NOT FIT"
            elif agents == 0:
                verdict = f"fits, but < 1 agent at {args.context} ctx"
            elif agents < 4:
                verdict = "tight"
            else:
                verdict = "ok"
            expect = r.get("expect")
            if expect == "oom" and b.fits:
                verdict += "  (config says oom -- recheck)"
            print(f"{r['rung']:<16}{b.weights_mib:>7} Mi{b.kv_mib:>8} Mi"
                  f"{b.kv_tokens:>11}{agents:>8}   {verdict}")
        print()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
