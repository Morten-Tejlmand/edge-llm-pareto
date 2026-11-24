#!/usr/bin/env python3
"""Rescore the measured QA answers stored in sweep result files (no model server).

Each generation trial keeps its raw answers, finish reasons and workload, so a
change in answer extraction (e.g. leading_think_blocks_v1 -> think_blocks_v2)
can be applied to collected data. Speed and energy fields are untouched; only
`quality` and the derived `qa_*` fields change. Inputs are never overwritten:
each file is written to --output-dir under the same name, with `rescored_from`.

    python scripts/rescore_trials.py 'results/pilot/*.jsonl' --output-dir results/pilot-rescored
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edge_llm_pareto.quality import score_measured_requests  # noqa: E402


def rescore_row(row: dict) -> tuple[dict, bool]:
    """Return (row, changed). Rows without measured QA pass through unchanged."""
    workload = row.get("workload") or {}
    if row.get("skipped") or workload.get("evaluation_mode") != "measured_qa" or not row.get("request_results"):
        return row, False
    old = (row.get("quality") or {}).get("metrics", {}).get("exact_match")
    row["quality"] = score_measured_requests(row["request_results"], workload)
    row["qa_score"] = row["quality"]["score"]
    row["qa_exact_match"] = row["quality"]["metrics"]["exact_match"]
    row["qa_bleu"] = row["quality"]["metrics"]["bleu"]["score"]
    row["qa_incomplete_reasoning"] = row["quality"]["incomplete_reasoning"]
    return row, old != row["qa_exact_match"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="result .jsonl paths or globs")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    paths = sorted({p for pat in args.results for p in glob.glob(pat)})
    if not paths:
        print("no result files found", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        src = Path(path)
        dst = out_dir / src.name
        if dst.resolve() == src.resolve():
            print(f"refusing to overwrite {src}; choose another --output-dir", file=sys.stderr)
            return 1
        provenance = dict(path=str(src), sha256=hashlib.sha256(src.read_bytes()).hexdigest())
        rows = changed = scored = 0
        with open(src) as fin, open(dst, "w") as fout:
            for line in fin:
                if not line.strip():
                    continue
                row, diff = rescore_row(json.loads(line))
                if "quality" in row and row.get("request_results"):
                    row["rescored_from"] = provenance
                    scored += 1
                rows += 1
                changed += diff
                fout.write(json.dumps(row) + "\n")
        print(f"{src.name}: {rows} rows, {scored} rescored, {changed} with a changed exact match -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
