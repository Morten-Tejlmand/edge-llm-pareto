#!/usr/bin/env python3
"""Rescore saved QA answers without invoking a model server."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from edge_llm_pareto.quality import rescore

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    raise SystemExit(rescore(args.dataset, args.input, args.output))
