#!/usr/bin/env python3
"""Prepare deterministic word-length excerpts from a local WikiText/raw corpus."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from edge_llm_pareto.workloads import excerpts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--words', type=int, default=192)
    p.add_argument('--count', type=int, default=100)
    a = p.parse_args()
    raw = Path(a.corpus).read_bytes()
    tasks = excerpts(raw.decode(), a.words, a.count)
    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    dest.with_suffix(dest.suffix + '.meta.json').write_text(json.dumps(dict(
        corpus_sha256=hashlib.sha256(raw).hexdigest(), words=a.words, count=a.count,
        preparation='nonoverlapping_words_v1', prompt_sha256=hashlib.sha256(dest.read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
