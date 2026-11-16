#!/usr/bin/env python3
"""Evaluate a model's configured GGUF variants on a fixed local corpus."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from edge_llm_pareto import serve
from edge_llm_pareto.perplexity import parse_perplexity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT / 'configs/models.yaml'))
    p.add_argument('--model', required=True)
    p.add_argument('--rung', default='all', help='all or comma-separated labels; reference always included')
    p.add_argument('--reference', required=True, help='configured F16/BF16 rung')
    p.add_argument('--corpus', required=True)
    p.add_argument('--context', type=int, default=2048)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--gpu-layers', type=int, default=999)
    p.add_argument('--model-revision', default='main')
    p.add_argument('--llama-perplexity', default='llama-perplexity')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if a.context < 2 or a.threads < 1 or a.gpu_layers < 0:
        p.error('invalid context, threads, or GPU layer count')
    cfg = yaml.safe_load(Path(a.config).read_text())['models'][a.model]['engine_rungs']['llama_cpp']
    labels = {r['rung'] for r in cfg['rungs']}
    selected = labels if a.rung == 'all' else set(a.rung.split(','))
    selected = selected | {a.reference}
    if selected - labels:
        p.error('unknown rung/reference')
    reference = next(r for r in cfg['rungs'] if r['rung'] == a.reference)
    if reference['linear_bits'] != 16:
        p.error('reference must be a configured 16-bit variant')
    executable = serve.resolve_executable(a.llama_perplexity)
    corpus = Path(a.corpus).resolve()
    raw = corpus.read_bytes()
    if not raw.strip():
        p.error('empty corpus')
    protocol = dict(corpus_sha256=hashlib.sha256(raw).hexdigest(), context=a.context,
                    threads=a.threads, gpu_layers=a.gpu_layers, seed=1, kv_type='f16',
                    scorer='llama_perplexity_default_v1', backend_version=serve.executable_version(executable))
    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    results = []
    # Resolve each repository only once so main cannot drift between variants.
    revisions = {}
    for rung in cfg['rungs']:
        if rung['rung'] not in selected:
            continue
        repo = rung.get('repo', cfg.get('repo'))
        gguf = serve.download_gguf(repo, rung['file'], revisions.get(repo, a.model_revision))
        revisions[repo] = gguf['revision']
        command = [executable, '-m', gguf['path'], '-f', str(corpus), '-c', str(a.context),
                   '-t', str(a.threads), '-ngl', str(a.gpu_layers), '-s', '1',
                   '--cache-type-k', 'f16', '--cache-type-v', 'f16']
        log = dest.with_name(dest.stem + '-' + rung['rung'] + '.log')
        print(f"Evaluating {rung['rung']}; log: {log}", flush=True)
        with log.open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        results.append(dict(base_model=a.model, label=rung['rung'], gguf=gguf,
                            perplexity=parse_perplexity(log.read_text()), command=command, log=str(log)))
    # A failed evaluation never publishes a partial bundle.
    payload = dict(schema_version=1, reference=a.reference, protocol=protocol, results=results)
    temporary = dest.with_suffix(dest.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(dest)


if __name__ == '__main__':
    main()
