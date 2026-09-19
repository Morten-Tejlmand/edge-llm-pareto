#!/usr/bin/env python3
"""Summarise a sweep and pick out the pareto-optimal (rung, concurrency) points.

Two objectives are in tension and both matter to an agent workload:

  output_tok_s     total tokens/s the box delivers   (operator's metric)
  per_agent_tok_s  tokens/s one agent experiences    (user's metric)

Raising concurrency trades the second for the first. A configuration is on the
front if nothing else beats it on both at once.

Accuracy is deliberately absent -- see docs/open-questions.md #3. Until it is
measured, this front ranks speed only, and a low-bit rung can look optimal here
while being unusable in practice.

    python scripts/report.py results/qwen3-1.7b-vllm-*.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import sys


def load(patterns: list[str]) -> list[dict]:
    rows = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
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
    args = ap.parse_args()

    rows = load(args.results)
    if not rows:
        print("no rows found", file=sys.stderr)
        return 1

    skipped = [r for r in rows if r.get("skipped")]
    done = [r for r in rows if not r.get("skipped")]

    print(f"{'rung':<15}{'conc':>5}{'tot tok/s':>11}{'per-agent':>11}"
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
        print(f"{r['label']:<15}{r['concurrency']:>5}{r['output_tok_s']:>11.1f}"
              f"{r.get('per_agent_tok_s') or 0:>11.1f}{r.get('ttft_s_mean') or 0:>9.2f}"
              f"{g.get('sm_util_pct_mean', 0):>6.0f}{g.get('mem_util_pct_mean', 0):>6.0f}"
              f"{ceil if ceil else '-':>6}  {'; '.join(note)}")

    for r in skipped:
        print(f"{r['label']:<15}{'-':>5}{'skipped':>11}  {r.get('reason')}")

    front = pareto(done)
    if front:
        print("\npareto front (nothing beats these on both throughput and per-agent speed):")
        for r in front:
            print(f"  {r['label']} @ concurrency {r['concurrency']}: "
                  f"{r['output_tok_s']:.1f} tok/s total, "
                  f"{r['per_agent_tok_s']:.1f} tok/s per agent")

    print("\nGPU activity counters alone do not establish compute/bandwidth saturation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
