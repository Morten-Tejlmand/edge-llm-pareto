#!/usr/bin/env python3
"""Validate a running vLLM server against a pinned publisher tokenizer."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from edge_llm_pareto.likelihood import load_tasks
from edge_llm_pareto.tokenizer_validation import check_running_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--model', required=True, help='served model alias')
    parser.add_argument('--tokenizer-repo', required=True)
    parser.add_argument('--tokenizer-revision', required=True)
    parser.add_argument('--dataset', default=str(ROOT / 'data/squad-dev-100.jsonl'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    tasks, _ = load_tasks(args.dataset)
    record = asyncio.run(check_running_server(base_url=args.base_url, model=args.model,
        tokenizer_repo=args.tokenizer_repo, tokenizer_revision=args.tokenizer_revision, tasks=tasks))
    Path(args.output).write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"Tokenizer {'PASS' if record['valid'] else 'FAIL'}: {record['input_cases']} input cases, "
          f"{len(record['generations'])} output generations; report: {args.output}")
    return 0 if record['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
