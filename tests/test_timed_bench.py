"""Timed load must retain concurrency and match token and energy boundaries."""
import asyncio
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from edge_llm_pareto import analysis, bench


class ConstantPower:
    interval = .1
    scope = 'test'
    source = 'test'
    errors = []

    def __enter__(self):
        self.samples = [dict(t=time.perf_counter(), power_w=10)]
        return self

    def __exit__(self, *args):
        self.samples.append(dict(t=time.perf_counter(), power_w=10))


class TimedBenchTests(unittest.TestCase):
    def test_sixteen_workers_refill_and_drain_outside_energy_window(self):
        client = Mock(close=AsyncMock())
        active = peak = calls = 0

        async def request(client, model, wl, seed, *, timed):
            nonlocal active, peak, calls
            self.assertTrue(timed)
            active += 1
            peak = max(peak, active)
            calls += 1
            started = time.perf_counter()
            events = [(started, 2)]
            # One straggler spans warmup, measurement and drain; the other
            # fifteen workers must continue well beyond a 48-request pool.
            await asyncio.sleep(.20 if seed == 0 else .005)
            events.append((time.perf_counter(), 3))
            active -= 1
            result = bench.request_result(True, seed, input_tokens=1,
                token_count_source='server', output_tokens=5,
                latency_s=time.perf_counter() - started)
            return dict(result, token_events=events, window_token_counts_exact=True)

        with patch('openai.AsyncOpenAI', return_value=client), \
             patch.object(bench, '_one_request', side_effect=request), \
             patch.object(bench, 'PowerMonitor', ConstantPower):
            row = asyncio.run(bench.run_concurrency_level('', 'm', 16, bench.make_workload(),
                duration_seconds=.08, load_warmup_seconds=.03))
        self.assertEqual(peak, 16)
        self.assertEqual(active, 0)
        self.assertGreater(calls, 48)
        self.assertGreater(row['effective_concurrency'], 15)
        self.assertLessEqual(row['effective_concurrency'], 16)
        self.assertGreater(row['drain_s'], .05)
        self.assertLess(row['wall_s'], .15)
        self.assertTrue(row['token_counts_exact'])
        lo, hi = row['measurement_start'], row['measurement_end']
        expected = sum(n for r in row['request_results'] for t, n in r['token_events'] if lo <= t < hi)
        self.assertEqual(row['output_tokens'], expected)
        self.assertNotEqual(expected, sum(r['output_tokens'] for r in row['request_results']))
        self.assertAlmostEqual(row['run_energy_j'], 10 * (hi - lo))
        self.assertAlmostEqual(row['joules_per_output_token'], 10 * (hi - lo) / expected)
        client.close.assert_awaited_once()

    def test_missing_window_counts_cannot_be_used_as_throughput(self):
        async def request(*args, **kwargs):
            await asyncio.sleep(.005)
            return bench.request_result(True, output_tokens=10, token_count_source='server', latency_s=.005)

        with patch('openai.AsyncOpenAI', return_value=Mock(close=AsyncMock())), \
             patch.object(bench, '_one_request', side_effect=request):
            row = asyncio.run(bench.run_concurrency_level('', 'm', 1, bench.make_workload(),
                monitor_gpu=False, duration_seconds=.02, load_warmup_seconds=0))
        self.assertFalse(row['token_counts_exact'])
        self.assertEqual(row['output_tok_s'], 0)
        self.assertIn('token_accounting_error', row)

    def test_stream_counts_tokens_in_bundled_and_reasoning_chunks(self):
        async def stream():
            for text, reasoning, count in ((None, 'thinking', 2), ('answer', None, 3)):
                yield SimpleNamespace(usage=None, choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=text, reasoning=reasoning), finish_reason=None,
                    logprobs=SimpleNamespace(content=[object()] * count))])
            yield SimpleNamespace(choices=[], usage=SimpleNamespace(completion_tokens=5, prompt_tokens=1))

        client = Mock()
        client.chat.completions.create = AsyncMock(side_effect=lambda **kwargs: stream())
        result = asyncio.run(bench._one_request(client, 'm', bench.make_workload(), 0, timed=True))
        self.assertEqual([n for _, n in result['token_events']], [2, 3])
        self.assertTrue(result['window_token_counts_exact'])
        self.assertTrue(client.chat.completions.create.call_args.kwargs['logprobs'])

    def test_timed_protocol_changes_trial_and_comparison_ids(self):
        async def run(*args, **kwargs):
            return dict(output_tok_s=10, failed=0, energy_valid=True, token_counts_exact=True,
                        duration_seconds=kwargs.get('duration_seconds', 0))

        async def sweeps():
            common = ('', 'm', [16], bench.make_workload(), 'q4')
            with patch.object(bench, 'run_concurrency_level', side_effect=run):
                finite = (await bench.sweep(*common))[0]
                timed = (await bench.sweep(*common, duration_seconds=60))[0]
                longer = (await bench.sweep(*common, duration_seconds=120))[0]
            return finite, timed, longer

        rows = asyncio.run(sweeps())
        self.assertEqual(len({r['trial_id'] for r in rows}), 3)
        self.assertEqual(len({analysis.comparison_key(r) for r in rows}), 3)

    def test_invalid_duration_is_rejected_before_client_creation(self):
        for duration in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                asyncio.run(bench.run_concurrency_level('', 'm', 16, bench.make_workload(),
                    duration_seconds=duration, monitor_gpu=False))
