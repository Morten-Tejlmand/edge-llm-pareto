"""QA answer likelihood: parsing, KL estimator, cross-engine consistency, reporting."""
import asyncio
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from edge_llm_pareto import likelihood as lk  # noqa: E402
from edge_llm_pareto.analysis import attach_likelihood  # noqa: E402

spec = importlib.util.spec_from_file_location('likelihood_report', ROOT / 'scripts/likelihood_report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
CONFIG = yaml.safe_load((ROOT / 'configs/study.yaml').read_text())
VOCAB = 8


def tokenize_text(text):
    return [1] + [ord(c) % VOCAB for c in text]  # BOS + one token per character


def logprobs_after(prefix, sharpness=1.0):
    """Deterministic fake next-token distribution over VOCAB tokens."""
    logits = [sharpness * math.sin(len(prefix) * (t + 1) + sum(prefix) % 5) for t in range(VOCAB)]
    z = math.log(sum(math.exp(x) for x in logits))
    return [x - z for x in logits]


def fake_server(engine, sharpness=1.0):
    def handler(request):
        body = json.loads(request.content)
        path = request.url.path
        if path == '/tokenize':
            text = body.get('content', body.get('prompt'))
            ids = tokenize_text(text)
            return httpx.Response(200, json={'tokens': ids} if engine == 'vllm'
                                  else {'tokens': ids})
        if engine == 'llama_cpp' and path == '/completion':
            lps = logprobs_after(body['prompt'], sharpness)
            top = sorted(range(VOCAB), key=lambda t: -lps[t])[:body['n_probs']]
            return httpx.Response(200, json={'completion_probabilities': [{
                'top_logprobs': [{'id': t, 'logprob': lps[t]} for t in top]}]})
        if engine == 'vllm' and path == '/v1/completions':
            ids, k = body['prompt'], body['prompt_logprobs']
            out = [None]
            for i in range(1, len(ids)):
                lps = logprobs_after(ids[:i], sharpness)
                order = sorted(range(VOCAB), key=lambda t: -lps[t])
                entry = {str(t): {'logprob': lps[t], 'rank': order.index(t) + 1} for t in order[:k]}
                entry[str(ids[i])] = {'logprob': lps[ids[i]], 'rank': order.index(ids[i]) + 1}
                out.append(entry)
            return httpx.Response(200, json={'choices': [{'prompt_logprobs': out}]})
        return httpx.Response(404)
    return httpx.MockTransport(handler)


TASKS = [{'id': 'a', 'prompt': 'Passage: x. Question: y?', 'answers': ['abc']},
         {'id': 'b', 'prompt': 'Passage: z. Question: w?', 'answers': ['de', 'other']}]


class LikelihoodTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.dataset = self.dir / 'qa.jsonl'
        self.dataset.write_text(''.join(json.dumps(t) + '\n' for t in TASKS))

    def tearDown(self):
        self.tmp.cleanup()

    def run_engine(self, engine, label, sharpness=1.0, top_k=3):
        return asyncio.run(lk.evaluate(
            base_url='http://fake/v1', engine=engine, model='m', base_model='qwen3.8-27b',
            label=label, model_repo='repo', model_revision='rev', dataset=self.dataset,
            tokens_dir=self.dir / 'tokens', output=self.dir / f'{engine}-{label}.likelihood.json',
            top_k=top_k, transport=fake_server(engine, sharpness)))

    def test_split_answer_requires_prefix(self):
        self.assertEqual(lk.split_answer([1, 2], [1, 2, 3, 4]), [3, 4])
        self.assertIsNone(lk.split_answer([1, 2], [1, 5, 3]))
        self.assertIsNone(lk.split_answer([1, 2], [1, 2]))

    def test_engines_agree_on_identical_model(self):
        llama = self.run_engine('llama_cpp', 'gguf-bf16')
        vllm = self.run_engine('vllm', 'native-bf16')
        self.assertEqual(llama['sequences_sha256'], vllm['sequences_sha256'])
        self.assertEqual(llama['summary']['answer_tokens'], len(' abc') + len(' de'))
        self.assertAlmostEqual(llama['summary']['answer_ppl'], vllm['summary']['answer_ppl'])
        c = lk.compare(llama, vllm)
        self.assertAlmostEqual(c['kl_mean'], 0, places=9)
        self.assertEqual(c['top1_agreement'], 1)

    def test_kl_positive_and_matches_exact_kl_with_full_vocab(self):
        ref = self.run_engine('llama_cpp', 'gguf-bf16', top_k=VOCAB)
        cand = self.run_engine('llama_cpp', 'gguf-ud-q4_k_xl', sharpness=1.5, top_k=VOCAB)
        exact = []
        for item in lk.load_or_build_tokens(Path(ref['tokens_file']), None, base_model='qwen3.8-27b',
                                            dataset_sha256=ref['dataset_sha256'], engine='x')['items']:
            ids = item['context_ids'] + item['answer_ids']
            kls = []
            for i in range(len(item['context_ids']), len(ids)):
                p, q = logprobs_after(ids[:i]), logprobs_after(ids[:i], 1.5)
                kls.append(sum(math.exp(a) * (a - b) for a, b in zip(p, q)))
            exact.append(sum(kls) / len(kls))
        c = lk.compare(ref, cand)
        self.assertGreater(c['kl_mean'], 0)
        self.assertAlmostEqual(c['kl_mean'], sum(exact) / len(exact), places=9)

    def test_llama_falls_back_when_target_outside_top_k(self):
        rec = self.run_engine('llama_cpp', 'gguf-bf16', top_k=1)
        self.assertTrue(all(len(p['top']) == 1 for i in rec['items'] for p in i['positions']))
        vllm = self.run_engine('vllm', 'native-bf16', top_k=1)
        self.assertAlmostEqual(rec['summary']['answer_nll'], vllm['summary']['answer_nll'])

    def test_frozen_tokens_reject_other_dataset(self):
        self.run_engine('llama_cpp', 'gguf-bf16')
        cache = next((self.dir / 'tokens').glob('*.json'))
        with self.assertRaises(ValueError):
            lk.load_or_build_tokens(cache, None, base_model='qwen3.8-27b', dataset_sha256='x', engine='vllm')

    def test_compare_rejects_different_sequences(self):
        a = self.run_engine('llama_cpp', 'gguf-bf16')
        b = dict(copy.deepcopy(a), sequences_sha256='other')
        with self.assertRaises(ValueError):
            lk.compare(a, b)

    def test_report_uses_same_engine_and_llama_references(self):
        recs = {}
        for engine, label, sharp in (('llama_cpp', 'gguf-bf16', 1), ('llama_cpp', 'gguf-ud-q4_k_xl', 1.5),
                                     ('vllm', 'native-bf16', 1), ('vllm', 'nvidia-nvfp4', 1.3)):
            recs[('qwen3.8-27b', engine, label)] = self.run_engine(engine, label, sharp)
        rows = {(r['engine'], r['label']): r for r in report.build(recs, CONFIG)}
        self.assertAlmostEqual(rows[('vllm', 'native-bf16')]['kl_to_llama_cpp_bf16'], 0, places=9)
        self.assertEqual(rows[('vllm', 'nvidia-nvfp4')]['reference_engine_bf16'], 'vllm/native-bf16')
        self.assertGreater(rows[('vllm', 'nvidia-nvfp4')]['kl_to_engine_bf16'], 0)
        self.assertGreater(rows[('llama_cpp', 'gguf-ud-q4_k_xl')]['kl_to_engine_bf16'], 0)

        bundle = dict(rows=list(rows.values()))
        bench = [dict(base_model='qwen3.8-27b', engine='vllm', label='nvidia-nvfp4', model_revision='rev'),
                 dict(base_model='qwen3.8-27b', engine='vllm', label='qwen-fp8')]
        attached = attach_likelihood(bench, bundle)
        self.assertGreater(attached[0]['likelihood']['kl_to_engine_bf16'], 0)
        self.assertIsNone(attached[1]['likelihood'])
        with self.assertRaises(ValueError):
            attach_likelihood([dict(bench[0], model_revision='other')], bundle)


if __name__ == '__main__':
    unittest.main()
