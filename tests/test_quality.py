import asyncio
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from edge_llm_pareto.quality import corpus_bleu, evaluate, extract_answer, rescore

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BleuTests(unittest.TestCase):
    def test_reasoning_extraction(self):
        self.assertEqual(extract_answer('<think>\n</think>\n\nParis'), ('Paris', 'ok'))
        self.assertEqual(extract_answer('<think>London? No.</think>Paris'), ('Paris', 'ok'))
        self.assertEqual(extract_answer('<think>Paris'), ('', 'incomplete_reasoning'))
        self.assertEqual(extract_answer('literal <think> tag'), ('literal <think> tag', 'ok'))

    def test_reasoning_opened_by_the_prompt(self):
        # Qwen3.8 on vLLM: the chat template ends the prompt with <think>, so the
        # completion holds only the closing tag (scored 0.00 EM under v1).
        self.assertEqual(extract_answer('We need the span.\n</think>\n\nDiepoldsau'), ('Diepoldsau', 'ok'))
        self.assertEqual(extract_answer('reasoning</think>\n<think>more</think>\nParis'), ('Paris', 'ok'))
        self.assertEqual(extract_answer('Paris'), ('Paris', 'ok'))

    def test_output_limit_before_reasoning_ends_is_incomplete(self):
        self.assertEqual(extract_answer('still thinking about it', 'length'), ('', 'incomplete_reasoning'))
        self.assertEqual(extract_answer('<think>still', 'length'), ('', 'incomplete_reasoning'))
        # Reasoning finished, answer cut off: scored as written.
        self.assertEqual(extract_answer('done</think>\nPar', 'length'), ('Par', 'ok'))
        self.assertEqual(extract_answer('still thinking', 'stop'), ('still thinking', 'ok'))

    def test_offline_rescore_preserves_raw_and_checks_dataset(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            dataset, source, output = [Path(tmp) / name for name in ('qa.jsonl', 'old.json', 'new.json')]
            dataset.write_text(json.dumps(dict(id='1', prompt='Q', answers=['Paris'])) + '\n')
            raw_answer = '<think>\n</think>\n\nParis'
            original = dict(dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
                scorer='normalized_exact_match_v1', results=[dict(id='1', answer=raw_answer, correct=False)])
            source.write_text(json.dumps(original))
            self.assertEqual(rescore(dataset, source, output), 0)
            revised = json.loads(output.read_text())
            self.assertEqual(revised['score'], 1)
            self.assertEqual(revised['results'][0]['raw_answer'], raw_answer)
            self.assertEqual(revised['results'][0]['answer'], 'Paris')
            self.assertEqual(json.loads(source.read_text()), original)
            with self.assertRaises(ValueError):
                rescore(dataset, source, source)
            dataset.write_text(dataset.read_text() + '\n')
            with self.assertRaises(ValueError):
                rescore(dataset, source, output)

    def test_short_answers_and_variable_references(self):
        result = corpus_bleu(['Paris', '42'], [['London', 'Paris'], ['42']])
        self.assertAlmostEqual(result['score'], 100)
        self.assertIn('eff:yes', result['signature'])
        self.assertIn('tok:13a', result['signature'])
        self.assertEqual(corpus_bleu(['wrong'], [['Paris']])['score'], 0)
        self.assertEqual(corpus_bleu([''], [['Paris']])['score'], 0)

    def test_corpus_statistics_not_average_sentence_scores(self):
        # Two matching unigrams out of four, with no matching bigrams.
        result = corpus_bleu(['a b x', 'd'], [['a q c'], ['d']])
        self.assertEqual(result['system_length'], 4)
        self.assertEqual(result['precisions'][0], 50)
        with self.assertRaises(ValueError):
            corpus_bleu(['a'], [])

    def test_evaluation_primary_score_and_failure(self):
        client = Mock()
        client.chat.completions.create = AsyncMock(side_effect=[
            types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='Paris'))]),
            RuntimeError('server failure')])
        manager = Mock(__aenter__=AsyncMock(return_value=client), __aexit__=AsyncMock())
        fake = types.SimpleNamespace(AsyncOpenAI=Mock(return_value=manager))
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / 'qa.jsonl'
            dataset.write_text('\n'.join(json.dumps(dict(id=str(i), prompt='Q', answers=['Paris'])) for i in range(2)))
            args = types.SimpleNamespace(dataset=str(dataset), output=str(Path(tmp) / 'out.json'),
                base_url='', model='m', model_repo='repo', model_revision='sha', base_model='base',
                engine='llama_cpp', rung='q4', max_tokens=64, scorer='bleu')
            with patch.dict('sys.modules', {'openai': fake}):
                self.assertEqual(asyncio.run(evaluate(args)), 1)
            record = json.loads(Path(args.output).read_text())
            self.assertEqual(record['failed'], 1)
            self.assertEqual(client.chat.completions.create.call_args.kwargs['max_tokens'], 64)
            self.assertEqual(record['max_tokens'], 64)
            self.assertEqual(record['metrics']['exact_match'], .5)
            self.assertEqual(record['metrics']['bleu']['reference_length'], 2)
            self.assertEqual(record['score'], record['metrics']['bleu']['score'] / 100)
            self.assertTrue(record['scorer'].startswith('sacrebleu_corpus_v1|'))
            sweep = script('sweep')
            match_args = types.SimpleNamespace(model='base', engine='llama_cpp', quality_max_tokens=64)
            rung = dict(rung='q4', repo='repo')
            with self.assertRaises(ValueError):
                sweep.matching_quality(record, match_args, rung, 'sha')
            record['failed'] = 0
            self.assertIs(sweep.matching_quality(record, match_args, rung, 'sha'), record)
            match_args.quality_max_tokens = None
            self.assertIsNone(sweep.matching_quality(record, match_args, rung, 'sha'))


class QaPreparationTests(unittest.TestCase):
    def test_reproducible_selection_and_reference_preservation(self):
        prepare = script('prepare_qa').prepare
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp) / 'source.json', Path(tmp) / 'tasks.jsonl'
            questions = [dict(id=str(i), question='Where?', answers=[
                dict(text='France'), dict(text='France'), dict(text='in France')]) for i in range(5)]
            article = dict(title='Cities', paragraphs=[dict(context='Paris is in France.', qas=questions)])
            source.write_text(json.dumps(dict(version='1.1', data=[article])))
            first = prepare(source, dest, 3, 42)
            original = dest.read_bytes()
            self.assertEqual(first, prepare(source, dest, 3, 42))
            self.assertEqual(original, dest.read_bytes())
            rows = [json.loads(line) for line in dest.read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]['answers'], ['France', 'in France'])
            self.assertIn('Paris is in France.', rows[0]['prompt'])
            with self.assertRaises(ValueError):
                prepare(source, dest, 6, 42)


if __name__ == '__main__':
    unittest.main()
