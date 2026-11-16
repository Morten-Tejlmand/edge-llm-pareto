"""Native llama-perplexity evaluation and exact-artifact result matching."""
import math
import re


def parse_perplexity(text):
    matches = re.findall(r'Final estimate:\s*PPL\s*=\s*([0-9.eE+-]+)', text)
    if len(matches) != 1:
        raise ValueError('Expected one completed Final estimate: PPL result; inspect native log')
    value = float(matches[0])
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid perplexity')
    return value


def artifact_key(row):
    gguf = row.get('gguf') or {}
    return (row.get('base_model'), row.get('label'), gguf.get('repo'),
            gguf.get('revision'), tuple(gguf.get('files', [])))


def attach_results(rows, bundles):
    index = {}
    for bundle in bundles:
        protocol = bundle['protocol']
        results = bundle['results']
        references = [r for r in results if r['label'] == bundle['reference']]
        if len(references) != 1:
            raise ValueError('Perplexity bundle requires exactly one reference')
        reference = references[0]
        for r in results:
            value, base = r['perplexity'], reference['perplexity']
            if not all(isinstance(v, (float, int)) and math.isfinite(v) and v > 0 for v in (value, base)):
                raise ValueError('Invalid perplexity value')
            key = artifact_key(r)
            if not all(key) or r['base_model'] != reference['base_model']:
                raise ValueError('Incomplete or incompatible perplexity identity')
            record = dict(perplexity=value, reference_perplexity=base,
                          perplexity_increase_pct=100 * (value / base - 1),
                          reference=artifact_key(reference), protocol=protocol)
            if key in index and index[key] != record:
                raise ValueError('Conflicting perplexity evaluations for the same artifact')
            index[key] = record
    return [dict(r, perplexity_evaluation=index.get(artifact_key(r))) for r in rows]
