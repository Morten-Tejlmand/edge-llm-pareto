#!/usr/bin/env python3
"""Summarize QA answer likelihood: perplexity, KL divergence and top-1 agreement.

Reads the *.likelihood.json files written by sweep.py (--likelihood-dataset) and,
per model, compares every artifact with two BF16 references:

  same-engine BF16   quantization loss within one engine (llama.cpp gguf-bf16,
                     vLLM native-bf16)
  llama.cpp BF16     one common reference for all rows of a model; for vLLM BF16
                     it is the cross-engine same-weights check (should be ~0)

    python scripts/likelihood_report.py 'results/full/*.likelihood.json' \\
        --output results/full/report/likelihood.json

Pass the output to report.py with --likelihood-results to add the columns to the
energy summary.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from edge_llm_pareto.likelihood import KL_METHOD, compare  # noqa: E402

COMMON_REFERENCE_ENGINE = 'llama_cpp'


def bf16_labels(config, model, engine):
    rungs = config['models'][model]['engine_rungs'].get(engine, {}).get('rungs', [])
    return {r['rung'] for r in rungs if r['linear_bits'] == 16}


def latest(records):
    """One record per (model, engine, label); refuse silently mixed revisions."""
    chosen = {}
    for path, rec in sorted(records, key=lambda x: x[0]):
        key = (rec['base_model'], rec['engine'], rec['label'])
        if key in chosen and chosen[key][1]['model_revision'] != rec['model_revision']:
            raise ValueError(f'{key} has records for different revisions: {chosen[key][0]} and {path}')
        chosen[key] = (path, rec)
    return {k: v[1] for k, v in chosen.items()}


def reference_for(records, config, model, engine):
    labels = bf16_labels(config, model, engine)
    found = [r for (m, e, l), r in records.items() if m == model and e == engine and l in labels]
    if len(found) > 1:
        raise ValueError(f'Several BF16 references for {model}/{engine}')
    return found[0] if found else None


def build(records, config):
    rows = []
    for (model, engine, label), rec in sorted(records.items()):
        row = dict(base_model=model, engine=engine, label=label, model_repo=rec['model_repo'],
                   model_revision=rec['model_revision'], top_k=rec['top_k'],
                   tokenizer_matches_frozen_ids=rec['tokenizer_matches_frozen_ids'],
                   **{k: rec['summary'][k] for k in ('items', 'answer_tokens', 'answer_nll', 'answer_ppl')})
        for prefix, ref in (('engine_bf16', reference_for(records, config, model, engine)),
                            ('llama_cpp_bf16', reference_for(records, config, model, COMMON_REFERENCE_ENGINE))):
            if ref is None:
                continue
            c = compare(ref, rec)
            row.update({f'kl_to_{prefix}': c['kl_mean'], f'kl_to_{prefix}_ci_low': c['kl_ci_low'],
                        f'kl_to_{prefix}_ci_high': c['kl_ci_high'],
                        f'top1_agreement_{prefix}': c['top1_agreement'],
                        f'answer_nll_delta_{prefix}': c['answer_nll_delta'],
                        f'reference_{prefix}': c['reference']})
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('results', nargs='+', help='likelihood JSON paths or globs')
    ap.add_argument('--config', default=str(ROOT / 'configs/study.yaml'))
    ap.add_argument('--output', required=True, help='JSON bundle; a CSV is written next to it')
    args = ap.parse_args()
    paths = sorted({p for pattern in args.results for p in glob.glob(pattern)})
    if not paths:
        ap.error('no likelihood files found')
    config = yaml.safe_load(Path(args.config).read_text())
    try:
        rows = build(latest([(p, json.loads(Path(p).read_text())) for p in paths]), config)
    except (ValueError, KeyError) as exc:
        ap.error(str(exc))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(schema_version=1, kl_method=KL_METHOD, rows=rows), indent=2))
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with out.with_suffix('.csv').open('w') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)

    def fmt(v, spec='.4f'):
        return format(v, spec) if isinstance(v, float) else '-'
    print(f"{'model':<28}{'engine':<11}{'label':<18}{'tokens':>7}{'ans PPL':>9}"
          f"{'KL eng':>9}{'top1 eng':>9}{'KL llama':>10}")
    for r in rows:
        print(f"{r['base_model']:<28}{r['engine']:<11}{r['label']:<18}{r['answer_tokens']:>7}"
              f"{fmt(r['answer_ppl'], '.3f'):>9}{fmt(r.get('kl_to_engine_bf16')):>9}"
              f"{fmt(r.get('top1_agreement_engine_bf16'), '.3f'):>9}{fmt(r.get('kl_to_llama_cpp_bf16')):>10}"
              + ('' if r['tokenizer_matches_frozen_ids'] else '  tokenizer differs from frozen IDs'))
    print(f'\nKL: {KL_METHOD} (approximation of full-vocabulary KL). Wrote {out} and {out.with_suffix(".csv")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
