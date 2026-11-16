#!/usr/bin/env python3
"""Check Spark hardware and the selected inference runtime before downloading weights."""
import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_llm_pareto.hardware import probe_gpu
from edge_llm_pareto import serve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['llama_cpp'], default='llama_cpp')
    parser.add_argument('--llama-server', default='llama-server')
    args = parser.parse_args()
    gpu = probe_gpu()
    if not gpu.unified_memory or 'GB10' not in gpu.name.upper():
        raise SystemExit(f"Expected GB10 unified memory, detected {gpu.name}")
    result = {'hardware': asdict(gpu), 'architecture': platform.machine(), 'engine': args.engine}
    try:
        executable = serve.resolve_executable(args.llama_server)
        version = serve.executable_version(executable)
        devices = subprocess.run([executable, '--list-devices'], capture_output=True,
                                 text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f'llama.cpp check failed: {exc}. See docs/engines.md') from exc
    listing = devices.stdout + devices.stderr
    if 'CUDA0' not in listing:
        raise SystemExit('llama-server does not list CUDA0; build with CUDA.\n' + listing)
    result.update(llama_cpp_version=version, devices=listing.strip(),
                  cuda_check='device enumerated; validate model offload with a smoke sweep')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
