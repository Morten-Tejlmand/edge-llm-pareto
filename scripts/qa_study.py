#!/usr/bin/env python3
"""Run the explicit QA study matrix; --dry-run prints commands without inference."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def build_commands(config, *, config_path, qa_dataset, repeats=None, requests=None,
                   llama_server='llama-server', vllm='vllm', results_dir='results'):
    """Return (commands, configurations, trials) for the GGUF matrix plus native vLLM rows.

    `model_rungs` lists the same GGUF artifacts for every engine. Optional
    `native_vllm_rungs` adds vLLM-only rows (publisher BF16, official quantized
    safetensors) so the optimized engine is measured, not only its GGUF plugin.
    """
    study = config['study']
    repetitions = study['repeats'] if repeats is None else repeats
    request_count = study['requests_per_agent'] if requests is None else requests
    if repetitions < 1 or request_count < 1:
        raise ValueError('Repetitions and requests must be positive')
    cells = [(model, engine, labels) for model, labels in study['model_rungs'].items()
             for engine in study['engines']]
    native = study.get('native_vllm_rungs') or {}
    if native and 'vllm' not in study['engines']:
        raise ValueError('native_vllm_rungs requires vllm in study engines')
    for model, labels in native.items():
        if model not in study['model_rungs']:
            raise ValueError(f'native_vllm_rungs model {model} is not in model_rungs')
        if set(labels) & set(study['model_rungs'][model]):
            raise ValueError(f'native_vllm_rungs for {model} repeat GGUF matrix rungs')
        cells.append((model, 'vllm', labels))
    commands = []
    configurations = 0
    for model, engine, labels in cells:
        rungs = config['models'][model]['engine_rungs'][engine]['rungs']
        if not labels or len(labels) != len(set(labels)) or set(labels) - {r['rung'] for r in rungs}:
            raise ValueError(f'Invalid study rungs for {model}/{engine}')
        configurations += len(labels) * len(config['concurrency'])
        commands.append([sys.executable, str(ROOT / 'scripts/sweep.py'),
            '--config', str(config_path), '--model', model, '--engine', engine,
            '--rung', ','.join(labels), '--qa-dataset', str(qa_dataset),
            '--quality-scorer', 'exact-match',
            '--concurrency', ','.join(map(str, config['concurrency'])),
            '--repeats', str(repetitions), '--requests-per-agent', str(request_count),
            '--warmup-requests', str(study['warmup_requests']),
            '--output-tokens', str(study['output_tokens']),
            '--context', str(study['context']), '--server-parallel', str(study['server_parallel']),
            '--seed', str(study['seed']), '--gsmm', '--results-dir', str(results_dir),
            '--llama-server', llama_server, '--vllm', vllm])
    return commands, configurations, configurations * repetitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/study.yaml'))
    parser.add_argument('--qa-dataset', help='override the configured QA dataset')
    parser.add_argument('--repeats', type=int, help='override repetitions for a pilot')
    parser.add_argument('--requests-per-agent', type=int, help='override request count for a pilot')
    parser.add_argument('--llama-server', default='llama-server')
    parser.add_argument('--vllm', default='vllm')
    parser.add_argument('--results-dir', default=str(ROOT / 'results'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    dataset = Path(args.qa_dataset or config['study']['qa_dataset'])
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    from edge_llm_pareto.bench import make_workload
    from edge_llm_pareto.workloads import attach_qa
    try:
        attach_qa(make_workload(), dataset)
        commands, configurations, trials = build_commands(config, config_path=config_path,
            qa_dataset=dataset, repeats=args.repeats, requests=args.requests_per_agent,
            llama_server=args.llama_server, vllm=args.vllm, results_dir=args.results_dir)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(f'Planned: {configurations} configurations, {trials} measured QA trials; '
          'unsupported cells may be skipped.', flush=True)
    failures = 0
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            try:
                failures += subprocess.run(command, check=False, cwd=ROOT).returncode != 0
            except OSError as exc:
                print(f'Failed to launch sweep: {exc}', file=sys.stderr)
                failures += 1
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
