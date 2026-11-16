#!/usr/bin/env python3
"""Convert a local SQuAD 1.1 development file into a reproducible QA subset."""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE_URL = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json'


def prepare(source, output, count=100, seed=42):
    raw = Path(source).read_bytes()
    dataset = json.loads(raw)
    if dataset.get('version') != '1.1':
        raise ValueError('Expected SQuAD 1.1 data')
    tasks = []
    for article in dataset['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                answers = list(dict.fromkeys(a['text'] for a in qa['answers']))
                if not answers or any(not a.strip() for a in answers):
                    raise ValueError('Expected nonempty answer spans')
                tasks.append(dict(id=qa['id'], title=article['title'],
                    prompt=('Answer the question using only the passage below. '
                            'Reply with only the shortest answer span, without explanation.\n\n'
                            f"Passage: {paragraph['context']}\n\nQuestion: {qa['question']}\nAnswer:"),
                    answers=answers))
    if len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('Duplicate question IDs')
    if not 1 <= count <= len(tasks):
        raise ValueError(f'count must be between 1 and {len(tasks)}')
    # Hash ranking makes selection stable across Python versions and source ordering.
    tasks.sort(key=lambda t: (hashlib.sha256(f"{seed}:{t['id']}".encode()).hexdigest(), t['id']))
    selected = tasks[:count]
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    encoded = ''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in selected).encode()
    dest.write_bytes(encoded)
    metadata = dict(dataset='SQuAD 1.1', split='dev', source_url=SOURCE_URL,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        dataset_sha256=hashlib.sha256(encoded).hexdigest(),
        license='CC BY-SA 4.0', license_url='https://creativecommons.org/licenses/by-sa/4.0/',
        attribution='Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang (2016)',
        preparation='squad_context_qa_sha256_rank_v1', seed=seed, count=count,
        source_count=len(tasks), selected_ids=[t['id'] for t in selected])
    dest.with_suffix(dest.suffix + '.meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='local dev-v1.1.json downloaded from SQuAD')
    parser.add_argument('--output', default='data/squad-dev-100.jsonl')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    metadata = prepare(args.source, args.output, args.count, args.seed)
    print(f"Prepared {metadata['count']} questions: {args.output}")


if __name__ == '__main__':
    main()
