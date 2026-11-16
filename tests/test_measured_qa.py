"""QA scoring uses the same responses as the concurrent energy/speed trials."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from edge_llm_pareto import analysis, bench, quality
from edge_llm_pareto.workloads import attach_qa
import test_llama_cpp


TASKS = [dict(id='paris', prompt='Capital of France?', answers=['Paris']),
         dict(id='rome', prompt='Capital of Italy?', answers=['Rome'])]


def workload(directory):
    path = Path(directory) / 'questions.jsonl'
    path.write_text('\n'.join(json.dumps(t) for t in TASKS))
    return path, attach_qa(bench.make_workload(), path)


class MeasuredQATests(unittest.TestCase):
    def test_dataset_validates_answers_and_preserves_identity(self):
        with tempfile.TemporaryDirectory() as d:
            path, wl = workload(d)
            self.assertEqual(wl['evaluation_mode'], 'measured_qa')
            self.assertEqual(wl['qa_dataset_sha256'], wl['prompt_dataset_sha256'])
            self.assertEqual(bench.make_prompt(wl, 3), TASKS[1]['prompt'])
            path.write_text(json.dumps(dict(TASKS[0], answers=[])))
            with self.assertRaisesRegex(ValueError, 'accepted answers'):
                attach_qa(bench.make_workload(), path)

    def test_stream_retains_full_content_separate_reasoning_and_termination(self):
        async def chunks():
            for text, reasoning in [('', 'considering'), ('Pa', ''), ('ris', '')]:
                yield SimpleNamespace(usage=None, choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=text, reasoning=reasoning), finish_reason=None)])
            yield SimpleNamespace(usage=SimpleNamespace(completion_tokens=8, prompt_tokens=4),
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason='length')])
        client = Mock()
        client.chat.completions.create = AsyncMock(return_value=chunks())
        result = asyncio.run(bench._one_request(client, 'model', bench.make_workload(), 0))
        self.assertEqual(result['raw_answer'], 'Paris')
        self.assertEqual(result['raw_reasoning'], 'considering')
        self.assertEqual(result['finish_reason'], 'length')
        self.assertEqual(result['output_tokens'], 8)

    def test_concurrent_repeats_failures_and_truncation_are_scored_after_monitor_stops(self):
        events = []
        class Power:
            samples = [dict(t=0, power_w=10), dict(t=4, power_w=10)]
            source, scope, errors, interval = 'fake', 'module', [], 1
            active = False
            def __enter__(self):
                self.active = True
                events.append('start')
                return self
            def __exit__(self, *args):
                self.active = False
                events.append('stop')
        power = Power()
        original = quality.score_measured_requests
        def score(*args):
            self.assertFalse(power.active)
            events.append('score')
            return original(*args)
        async def request(client, model, wl, seed):
            events.append('request')
            raw = {0:'Paris', 1:'Rome', 2:'partial', 3:'wrong', 4:'<think>unfinished', 5:'Rome'}.get(seed, 'warmup')
            return bench.request_result(seed != 2, request_id=seed, token_count_source='server',
                latency_s=1, ttft_s=.1, output_tokens=2 if seed != 2 else 0,
                raw_answer=raw, error='timeout' if seed == 2 else None,
                finish_reason='length' if seed == 4 else 'stop')
        client = Mock(close=AsyncMock())
        with tempfile.TemporaryDirectory() as d:
            _, wl = workload(d)
            with patch('openai.AsyncOpenAI', return_value=client), \
                    patch.object(bench, '_one_request', side_effect=request) as send, \
                    patch.object(bench, 'PowerMonitor', return_value=power), \
                    patch.object(bench.time, 'perf_counter', side_effect=[1, 3]), \
                    patch.object(quality, 'score_measured_requests', side_effect=score):
                row = asyncio.run(bench.run_concurrency_level('', 'model', 2, wl, 3, warmup_requests=2))
        self.assertEqual(send.call_count, 8)  # Two warmups; no separate QA calls.
        self.assertEqual(events[-2:], ['stop', 'score'])
        self.assertEqual(row['requests'], 6)
        self.assertEqual(row['quality']['requests'], 6)
        self.assertEqual(row['quality']['unique_questions'], 2)
        self.assertEqual(row['quality']['failed'], 1)
        self.assertEqual(row['qa_exact_match'], .5)
        self.assertEqual(row['qa_incomplete_reasoning'], 1)
        self.assertEqual(row['length_limited_requests'], 1)
        self.assertEqual(row['run_energy_j'], 20)
        self.assertIsNone(row['joules_per_output_token'])
        by_id = {r['request_id']: r for r in row['request_results']}
        self.assertEqual(by_id[2]['raw_answer'], 'partial')
        self.assertFalse(by_id[2]['correct'])
        self.assertEqual(by_id[5]['question_id'], 'rome')
        self.assertEqual(by_id[0]['reference_answers'], ['Paris'])

    def test_sweep_does_not_overwrite_trial_quality_with_preflight_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            _, wl = workload(d)
            summary = dict(evaluation_mode='measured_qa', score=.75)
            with patch.object(bench, 'run_concurrency_level', AsyncMock(return_value=dict(
                    output_tok_s=10, failed=0, quality=summary))):
                rows = asyncio.run(bench.sweep('', 'model', [1,2], wl, 'q4',
                    metadata={'quality': None}, repeats=2))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r['quality']['score'] == .75 for r in rows))

    def test_quality_aggregation_uses_all_trials_and_filters_mean_score(self):
        with tempfile.TemporaryDirectory() as d:
            _, wl = workload(d)
            rows = []
            for i, answer in enumerate(('Paris', 'wrong')):
                requests = [bench.request_result(True, raw_answer=answer)]
                q = quality.score_measured_requests(requests, wl)
                rows.append(dict(config_id='a', trial_id=str(i), output_tok_s=10,
                    latency_s_p95=1, ttft_s_p95=.1, joules_per_output_token=2,
                    run_energy_j=20, energy_valid=True, token_counts_exact=True,
                    quality=q, qa_score=q['score'], qa_exact_match=q['metrics']['exact_match'],
                    qa_bleu=q['metrics']['bleu']['score']))
            data = analysis.aggregate(rows)
            self.assertEqual(data[0]['quality']['score'], .5)
            self.assertEqual(data[0]['quality']['trials'], 2)
            self.assertEqual(data[0]['qa_exact_match'], .5)
            self.assertEqual(data[0]['quality']['metrics']['exact_match'], .5)
            self.assertEqual(data[0]['quality']['minimum_unique_questions'], 1)
            # A threshold above the two-trial mean must not keep the first trial's perfect score.
            with self.assertRaisesRegex(ValueError, 'No valid energy trials'):
                analysis.export(rows, d, min_quality=.75)
            conflicting = copy.deepcopy(rows)
            conflicting[1]['quality']['max_tokens'] = 1024
            with self.assertRaisesRegex(ValueError, 'one dataset and scoring protocol'):
                analysis.aggregate(conflicting)

    def test_cli_uses_qa_as_the_measured_workload(self):
        with tempfile.TemporaryDirectory() as d:
            path, _ = workload(d)
            inputs = Path(d) / 'inputs'
            inputs.mkdir()
            path.rename(inputs / path.name)
            rc, _, _, run, _, _ = test_llama_cpp.LlamaSweepTests().run_cli(d,
                extra=['--qa-dataset', str(inputs / path.name)])
        self.assertEqual(rc, 0)
        wl = run.call_args.args[3]
        self.assertEqual(wl['evaluation_mode'], 'measured_qa')
        self.assertEqual(wl['max_output_tokens'], 256)
        self.assertEqual(wl['prompts'][0]['answers'], ['Paris'])


class TrialCoverageTest(unittest.TestCase):
    def test_trials_continue_through_the_dataset(self):
        row = dict(output_tok_s=10, failed=0, energy_valid=True, token_counts_exact=True)
        with patch.object(bench, 'run_concurrency_level', AsyncMock(return_value=row)) as run:
            asyncio.run(bench.sweep('', 'm', [1, 4], bench.make_workload(), 'q4',
                                    requests_per_agent=3, repeats=3))
        offsets = {(call.args[2], call.kwargs['request_offset']) for call in run.await_args_list}
        self.assertEqual(offsets, {(1, 0), (1, 3), (1, 6), (4, 0), (4, 12), (4, 24)})
