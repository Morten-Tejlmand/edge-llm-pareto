#!/usr/bin/env python3
"""Run the explicit QA study matrix; --dry-run prints commands without inference."""
import argparse
import copy
import json
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

    `model_rungs` lists GGUF artifacts, run on every engine in `gguf_engines`
    (default: all `engines`). `native_vllm_rungs` adds vLLM-only rows (publisher
    BF16, official quantized safetensors), which use different weight files.
    """
    study = config['study']
    repetitions = study['repeats'] if repeats is None else repeats
    request_count = study['requests_per_agent'] if requests is None else requests
    if repetitions < 1 or request_count < 1:
        raise ValueError('Repetitions and requests must be positive')
    gguf_engines = study.get('gguf_engines', study['engines'])
    if set(gguf_engines) - set(study['engines']):
        raise ValueError('gguf_engines must be a subset of study engines')
    cells = [(model, engine, labels) for model, labels in study['model_rungs'].items()
             for engine in gguf_engines]
    native = study.get('native_vllm_rungs') or {}
    if native and 'vllm' not in study['engines']:
        raise ValueError('native_vllm_rungs requires vllm in study engines')
    for model, labels in native.items():
        if model not in study['model_rungs']:
            raise ValueError(f'native_vllm_rungs model {model} is not in model_rungs')
        if set(labels) & set(study['model_rungs'][model]):
            raise ValueError(f'native_vllm_rungs for {model} repeat GGUF matrix rungs')
        cells.append((model, 'vllm', labels))
    if not cells:
        raise ValueError('The study has no configurations')
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
            '--warmup-trials', str(study.get('warmup_trials', 0)),
            '--output-tokens', str(study['output_tokens']),
            '--context', str(study['context']), '--server-parallel', str(study['server_parallel']),
            '--seed', str(study['seed']), '--gsmm', '--results-dir', str(results_dir),
            '--llama-server', llama_server, '--vllm', vllm])
        if study.get('likelihood'):
            commands[-1] += ['--likelihood-dataset', str(qa_dataset),
                             '--likelihood-top-k', str(study.get('likelihood_top_k', 20))]
    return commands, configurations, configurations * repetitions


def subset(config, *, models=None, engines=None, rungs=None, concurrency=None):
    """Return a copy of the study restricted to the selected values (for pilots)."""
    config = copy.deepcopy(config)
    study = config['study']
    def pick(name, available, selected):
        if selected is None:
            return available
        unknown = set(selected) - set(available)
        if unknown:
            raise ValueError(f'Unknown {name}: {sorted(unknown)}; available: {sorted(available)}')
        return [value for value in available if value in selected]
    all_rungs = {r for blocks in ('model_rungs', 'native_vllm_rungs')
                 for labels in (study.get(blocks) or {}).values() for r in labels}
    pick('rungs', sorted(all_rungs), rungs)
    keep_models = pick('models', list(study['model_rungs']), models)
    study['engines'] = pick('engines', study['engines'], engines)
    if 'gguf_engines' in study:
        study['gguf_engines'] = [e for e in study['gguf_engines'] if e in study['engines']]
    if 'vllm' not in study['engines']:
        study.pop('native_vllm_rungs', None)
    config['concurrency'] = pick('concurrency', config['concurrency'], concurrency)
    for block in ('model_rungs', 'native_vllm_rungs'):
        if block in study:
            study[block] = {m: [r for r in labels if rungs is None or r in rungs]
                            for m, labels in (study[block] or {}).items() if m in keep_models}
            study[block] = {m: labels for m, labels in study[block].items() if labels}
    gguf_engines = study.get('gguf_engines', study['engines'])
    has_cells = (study['model_rungs'] and gguf_engines) or study.get('native_vllm_rungs')
    if not has_cells or not study['engines'] or not config['concurrency']:
        raise ValueError('The selected subset is empty')
    return config


def resume_file(results_dir, model, engine, rungs):
    """Earlier results file for this sweep (same model, engine and rungs), if any."""
    candidates = []
    for path in sorted(Path(results_dir).glob(f'{model}-{engine}-*.jsonl')):
        try:
            labels = {json.loads(line).get('label') for line in path.open() if line.strip()}
        except (OSError, ValueError):
            continue
        if labels & set(rungs):
            candidates.append(path)
    if len(candidates) > 1:
        raise ValueError(f'Several earlier results files match {model}/{engine}: '
                         f'{[str(c) for c in candidates]}; keep one or use a new --results-dir')
    return candidates[0] if candidates else None


def add_resume(commands, results_dir):
    """Append --resume to sweeps that already have results; return the resumed files."""
    resumed = []
    for command in commands:
        model = command[command.index('--model') + 1]
        engine = command[command.index('--engine') + 1]
        rungs = command[command.index('--rung') + 1].split(',')
        path = resume_file(results_dir, model, engine, rungs)
        if path:
            command += ['--resume', str(path)]
            resumed.append(path)
    return resumed


def comma_list(value, cast=str):
    return [cast(v) for v in value.split(',') if v]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/study.yaml'))
    parser.add_argument('--qa-dataset', help='override the configured QA dataset')
    parser.add_argument('--repeats', type=int, help='override repetitions for a pilot')
    parser.add_argument('--requests-per-agent', type=int, help='override request count for a pilot')
    parser.add_argument('--llama-server', default='llama-server')
    parser.add_argument('--vllm', default='vllm')
    parser.add_argument('--results-dir', default=str(ROOT / 'results'))
    parser.add_argument('--models', help='comma-separated subset of study models (pilot)')
    parser.add_argument('--engines', help='comma-separated subset: llama_cpp,vllm (pilot)')
    parser.add_argument('--rungs', help='comma-separated subset of quantization rungs (pilot)')
    parser.add_argument('--concurrency', help='comma-separated subset of concurrency levels (pilot)')
    parser.add_argument('--resume', action='store_true',
                        help='continue earlier results in --results-dir; completed trials are skipped')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    try:
        config = subset(config,
            models=comma_list(args.models) if args.models else None,
            engines=comma_list(args.engines) if args.engines else None,
            rungs=comma_list(args.rungs) if args.rungs else None,
            concurrency=comma_list(args.concurrency, int) if args.concurrency else None)
    except ValueError as exc:
        parser.error(str(exc))
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
    if args.resume:
        try:
            resumed = add_resume(commands, args.results_dir)
        except ValueError as exc:
            parser.error(str(exc))
        for path in resumed:
            print(f'Resuming {path}', flush=True)
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
