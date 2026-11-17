#!/usr/bin/env python3
"""Summarise a sweep and pick out the pareto-optimal (rung, concurrency) points.

Two objectives are in tension and both matter to an agent workload:

  output_tok_s     total tokens/s the box delivers   (operator's metric)
  per_agent_tok_s  tokens/s one agent experiences    (user's metric)

Raising concurrency trades the second for the first. A configuration is on the
front if nothing else beats it on both at once.

The default report ranks speed only. Energy reports support exact-match or BLEU quality
constraints and attached perplexity evaluations; combined-model reporting is
explicit and preserves the other workload and measurement boundaries.

    python scripts/report.py results/qwen3-1.7b-llama_cpp-*.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def load(patterns: list[str]) -> list[dict]:
    rows = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            from edge_llm_pareto.results import read_records
            rows.extend(r for r in read_records(path)
                        if r.get('scenario') not in ('system_baseline', 'server_idle'))
    return rows


def pareto(rows: list[dict]) -> list[dict]:
    """Maximise output_tok_s and per_agent_tok_s simultaneously."""
    pts = [r for r in rows if r.get("output_tok_s") and r.get("per_agent_tok_s")
           and not r.get("throttled") and not r.get("failed")]
    front = []
    for r in pts:
        dominated = any(
            o is not r
            and o["output_tok_s"] >= r["output_tok_s"]
            and o["per_agent_tok_s"] >= r["per_agent_tok_s"]
            and (o["output_tok_s"] > r["output_tok_s"]
                 or o["per_agent_tok_s"] > r["per_agent_tok_s"])
            for o in pts
        )
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: -r["output_tok_s"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="jsonl paths or globs")
    ap.add_argument("--energy", action="store_true")
    ap.add_argument("--export-dir", default="results/report")
    ap.add_argument("--energy-scope", choices=["nvml", "system", "system_total", "soc", "gpu"],
                    default="nvml", help="nvml = NVML GPU power; others use the Spark spbm rails")
    ap.add_argument("--objective", choices=["output_tok_s", "latency_s_p95", "ttft_s_p95", "throughput_ttft"],
                    default="output_tok_s",
                    help="axis ranked against J/output token; throughput_ttft ranks J/token, tokens/s and p95 TTFT together")
    ap.add_argument("--baseline", help="configuration ID to compare within its matching group")
    ap.add_argument("--min-quality", type=float)
    ap.add_argument('--perplexity-results', nargs='+', help='JSON bundles from perplexity.py')
    ap.add_argument('--max-perplexity-increase-pct', type=float)
    ap.add_argument('--likelihood-results', help='JSON bundle from likelihood_report.py (QA answer PPL and KL columns)')
    ap.add_argument('--combine-models', action='store_true', help='compare models on matched workloads; relative PPL loss is not equal task quality')
    args = ap.parse_args()
    if args.energy_scope != 'nvml' and not args.energy:
        ap.error('--energy-scope requires --energy')
    if (args.perplexity_results or args.likelihood_results or args.max_perplexity_increase_pct is not None
            or args.combine_models) and not args.energy:
        ap.error('perplexity and combined-model options require --energy')
    if args.max_perplexity_increase_pct is not None and not args.perplexity_results:
        ap.error('perplexity filtering requires --perplexity-results')

    rows = load(args.results)
    if not rows:
        print("no rows found", file=sys.stderr)
        return 1

    if args.energy:
        from edge_llm_pareto.analysis import export, with_energy_scope
        rows = [with_energy_scope(r, args.energy_scope) for r in rows]
        try:
            if args.perplexity_results:
                from edge_llm_pareto.perplexity import attach_results
                from pathlib import Path
                rows = attach_results(rows, [json.loads(Path(p).read_text()) for p in args.perplexity_results])
            if args.likelihood_results:
                from edge_llm_pareto.analysis import attach_likelihood
                rows = attach_likelihood(rows, json.loads(Path(args.likelihood_results).read_text()))
            data = export(rows, args.export_dir, args.objective, args.baseline, args.min_quality,
                          args.max_perplexity_increase_pct, args.combine_models)
            if args.combine_models and args.min_quality is None:
                print('Combined frontier compares efficiency; relative perplexity loss does not establish equal task quality.')
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"Exported {len(data)} configurations to {args.export_dir}")
        return 0

    skipped = [r for r in rows if r.get("skipped")]
    done = [r for r in rows if not r.get("skipped")]

    print(f"{'engine':<11}{'rung':<20}{'conc':>5}{'tot tok/s':>11}{'per-agent':>11}"
          f"{'ttft s':>9}{'sm%':>6}{'mem%':>6}{'ceil':>6}  note")
    for r in sorted(done, key=lambda r: (r["label"], r["concurrency"])):
        g = r.get("gpu", {})
        note = []
        if r.get("failed"):
            note.append(f"{r['failed']} failed")
        # A throttled clock makes a row incomparable to anything else, so say so
        # on the row rather than leaving it to be read as a real measurement.
        if r.get("throttled"):
            note.append("THROTTLED: excluded from Pareto front")
        if r.get("gpu_error"):
            note.append("some GPU telemetry unavailable")
        ceil = r.get("kv_ceiling_agents")
        if ceil and r["concurrency"] > ceil:
            note.append("above KV ceiling: queueing, not parallel decode")
        print(f"{r.get('engine', 'unknown'):<11}{r['label']:<20}{r['concurrency']:>5}{r['output_tok_s']:>11.1f}"
              f"{r.get('per_agent_tok_s') or 0:>11.1f}{r.get('ttft_s_mean') or 0:>9.2f}"
              f"{g.get('sm_util_pct_mean', 0):>6.0f}{g.get('mem_util_pct_mean', 0):>6.0f}"
              f"{ceil if ceil else '-':>6}  {'; '.join(note)}")

    for r in skipped:
        print(f"{r.get('engine', 'unknown'):<11}{r['label']:<20}{'-':>5}{'skipped':>11}  {r.get('reason')}")

    front = pareto(done)
    if front:
        print("\npareto front (nothing beats these on both throughput and per-agent speed):")
        for r in front:
            print(f"  {r.get('engine', 'unknown')}/{r['label']} @ concurrency {r['concurrency']}: "
                  f"{r['output_tok_s']:.1f} tok/s total, "
                  f"{r['per_agent_tok_s']:.1f} tok/s per agent")

    print("\nGPU activity counters alone do not establish compute/bandwidth saturation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
