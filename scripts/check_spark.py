#!/usr/bin/env python3
"""Run inside the Spark runtime before downloading benchmark weights."""
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_llm_pareto.hardware import probe_gpu


def main():
    gpu = probe_gpu()
    if not gpu.unified_memory:
        raise SystemExit(f"Expected GB10 unified memory, detected {gpu.name}")
    import torch
    import vllm

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this runtime")
    x = torch.ones((128, 128), device="cuda", dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    if not torch.all(y == 128).item():
        raise SystemExit("CUDA matrix check returned incorrect values")
    print(json.dumps({"hardware": asdict(gpu), "architecture": platform.machine(),
                      "torch": torch.__version__, "cuda": torch.version.cuda,
                      "vllm": vllm.__version__, "cuda_check": "passed"}, indent=2))


if __name__ == "__main__":
    main()
