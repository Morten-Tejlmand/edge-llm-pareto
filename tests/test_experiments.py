import asyncio
import copy
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from edge_llm_pareto.energy import integrate
from edge_llm_pareto import bench, analysis, hardware, telemetry


class EnergyTests(unittest.TestCase):
    def test_clipping_and_linear_interpolation(self):
        samples = [dict(t=0, power_w=0), dict(t=2, power_w=20)]
        r = integrate(samples, .5, 1.5, 3)
        self.assertTrue(r['energy_valid'])
        self.assertAlmostEqual(r['run_energy_j'], 10)

    def test_gaps_and_missing_boundaries_are_not_extrapolated(self):
        for samples in ([], [dict(t=.2, power_w=10), dict(t=.8, power_w=10)],
                        [dict(t=0, power_w=10), dict(t=1, power_w=10)]):
            r = integrate(samples, 0, 1, .5)
            self.assertFalse(r['energy_valid'])
            self.assertIsNone(r['run_energy_j'])

    def test_invalid_power_does_not_become_zero_energy(self):
        r = integrate([dict(t=0, power_w=float('nan')), dict(t=1, power_w=-1)], 0, 1)
        self.assertFalse(r['energy_valid'])


class ProtocolTests(unittest.TestCase):
    def test_resume_repeats_and_identity(self):
        row = dict(output_tok_s=10, failed=0, energy_valid=True, token_counts_exact=True)
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / 'r.jsonl')
            with patch.object(bench, 'run_concurrency_level', AsyncMock(return_value=row)) as run:
                kwargs = dict(out_path=path, repeats=2, warmup_requests=3, seed=4)
                asyncio.run(bench.sweep('', 'm', [1, 2], bench.make_workload(), 'q4', **kwargs))
                self.assertEqual(run.await_count, 4)
                self.assertEqual(run.call_args.kwargs['warmup_requests'], 3)
                asyncio.run(bench.sweep('', 'm', [1, 2], bench.make_workload(), 'q4', resume=True, **kwargs))
                self.assertEqual(run.await_count, 4)
                asyncio.run(bench.sweep('', 'm', [1], bench.make_workload(prompt_tokens=64), 'q4', resume=True, **kwargs))
                self.assertEqual(run.await_count, 6)
            saved = [json.loads(line) for line in Path(path).read_text().splitlines()]
            self.assertEqual(len({r['trial_id'] for r in saved}), 6)

    def test_warmup_trial_runs_once_unrecorded_and_changes_identity(self):
        row = dict(output_tok_s=10, failed=0, energy_valid=True, token_counts_exact=True)
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / 'r.jsonl')
            with patch.object(bench, 'run_concurrency_level', AsyncMock(return_value=row)) as run:
                kwargs = dict(out_path=path, repeats=2, seed=4, requests_per_agent=3)
                asyncio.run(bench.sweep('', 'm', [1, 2], bench.make_workload(), 'q4', **kwargs))
                plain_ids = {json.loads(line)['trial_id'] for line in Path(path).read_text().splitlines()}
                run.reset_mock()
                asyncio.run(bench.sweep('', 'm', [1, 2], bench.make_workload(), 'q4', warmup_trials=1, **kwargs))
                # One discarded trial first, unmonitored, past the measured request range.
                self.assertEqual(run.await_count, 5)
                first = run.await_args_list[0]
                self.assertFalse(first.kwargs['monitor_gpu'])
                c = first.args[2]
                self.assertEqual(first.kwargs['request_offset'], 2 * c * 3)
                self.assertEqual(run.await_args_list[1].args[2], c)  # Same concurrency as first measured trial.
                saved = [json.loads(line) for line in Path(path).read_text().splitlines()]
                self.assertEqual(len(saved), 8)  # Discarded trial is never written.
                self.assertFalse({r['trial_id'] for r in saved[4:]} & plain_ids)
                # Resume with nothing pending skips the warmup trial as well.
                run.reset_mock()
                asyncio.run(bench.sweep('', 'm', [1, 2], bench.make_workload(), 'q4', warmup_trials=1,
                                        resume=True, **kwargs))
                self.assertEqual(run.await_count, 0)

    def test_warmup_excluded_and_chunk_counts_marked(self):
        client = Mock(close=AsyncMock())
        fake = types.SimpleNamespace(AsyncOpenAI=Mock(return_value=client))
        request = bench.request_result(True, output_tokens=2, latency_s=.1, ttft_s=.01, itls_s=[.01])
        with patch.dict('sys.modules', {'openai': fake}), patch.object(bench, '_one_request', AsyncMock(return_value=request)) as run:
            row = asyncio.run(bench.run_concurrency_level('', 'm', 2, bench.make_workload(), 2,
                                                         monitor_gpu=False, warmup_requests=3))
        self.assertEqual(run.await_count, 7)
        self.assertEqual(row['requests'], 4)
        self.assertFalse(row['token_counts_exact'])
        self.assertEqual(len(row['request_results']), 4)
        self.assertEqual(row['latency_s_p95'], .1)
        client.close.assert_awaited_once()

    def test_intentional_power_cap_is_not_invalid(self):
        state = hardware.PowerState(True, 30, 30, 200, 2000, ['SW Power Cap'])
        self.assertTrue(hardware.check_benchmark_conditions(state))
        self.assertFalse(hardware.check_benchmark_conditions(state, intentional_power_limit=True))
        state.throttle_reasons.append('HW Thermal Slowdown')
        self.assertTrue(hardware.check_benchmark_conditions(state, intentional_power_limit=True))

    def test_idle_gb10_power_cap_flag_is_not_throttling(self):
        # Observed on the Spark: SW Power Cap reported at 4.95 W, 208/3003 MHz.
        idle = hardware.PowerState(None, 4.95, None, 208, 3003, ['SW Power Cap'])
        self.assertEqual(hardware.check_benchmark_conditions(idle), [])
        idle.throttle_reasons.append('HW Thermal Slowdown')
        self.assertTrue(hardware.check_benchmark_conditions(idle))
        unknown_power = hardware.PowerState(None, None, None, 208, 3003, ['SW Power Cap'])
        self.assertTrue(hardware.check_benchmark_conditions(unknown_power))


class AnalysisTests(unittest.TestCase):
    def row(self, config='a', trial='1', energy=2, throughput=10):
        return dict(config_id=config, trial_id=trial, energy_valid=True, token_counts_exact=True,
                    output_tok_s=throughput, latency_s_p95=1, ttft_s_p95=.1,
                    joules_per_output_token=energy, run_energy_j=20, hardware={'name': 'test'},
                    base_model='m', workload={'prompt_tokens': 10}, energy_scope='module',
                    label='q4', concurrency=1)

    def test_trials_aggregate_before_frontier(self):
        rows = [self.row(trial='1', energy=1), self.row(trial='2', energy=5),
                self.row('b', '3', energy=2)]
        data = analysis.aggregate(rows + [rows[0]])
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['trials'], 2)
        self.assertEqual(data[0]['joules_per_output_token'], 3)
        self.assertEqual([r['config_id'] for r in analysis.frontier(data)], ['b'])

    def test_incomparable_workload_cannot_dominate(self):
        a, b = self.row(), self.row('b', '2', 1, 100)
        b['workload'] = {'prompt_tokens': 2000}
        self.assertEqual(len(analysis.frontier(analysis.aggregate([a, b]))), 2)

    def test_bad_telemetry_and_approximate_tokens_excluded(self):
        for key, value in [('energy_valid', False), ('token_counts_exact', False), ('failed', 1),
                           ('unexpected_thermal_throttling', True), ('run_energy_j', float('nan'))]:
            row = self.row()
            row[key] = value
            self.assertEqual(analysis.aggregate([row]), [])

    def test_latency_frontier_minimizes_both(self):
        a, b = self.row(), self.row('b', '2', 1)
        b['latency_s_p95'] = .5
        self.assertEqual([r['config_id'] for r in analysis.frontier(analysis.aggregate([a,b]), 'latency_s_p95')], ['b'])

    def test_ttft_frontier_minimizes_both(self):
        a, b = self.row(), self.row('b', '2', 1)
        b['ttft_s_p95'] = .05
        self.assertEqual([r['config_id'] for r in analysis.frontier(analysis.aggregate([a, b]), 'ttft_s_p95')], ['b'])

    def test_three_way_frontier_keeps_tradeoffs(self):
        # a: cheapest energy; b: more throughput at higher energy; c: fastest TTFT;
        # d: worse than b on every axis.
        a = self.row('a', '1', 1, 10)
        b = self.row('b', '2', 2, 20)
        c = self.row('c', '3', 2, 10)
        c['ttft_s_p95'] = .01
        d = self.row('d', '4', 3, 15)
        d['ttft_s_p95'] = .2
        data = analysis.aggregate([a, b, c, d])
        front = sorted(r['config_id'] for r in analysis.frontier(data, 'throughput_ttft'))
        self.assertEqual(front, ['a', 'b', 'c'])
        # Without TTFT, c is dominated by a (same throughput, more energy).
        self.assertEqual(sorted(r['config_id'] for r in analysis.frontier(data)), ['a', 'b'])

    def test_unknown_objective_rejected(self):
        with self.assertRaises(ValueError):
            analysis.frontier([], 'nope')

class AdapterTests(unittest.TestCase):
    def test_orin_and_nano_units(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            hw = root / 'class/hwmon/hwmon0'
            hw.mkdir(parents=True)
            (hw / 'in1_label').write_text('VDD_IN')
            (hw / 'in1_input').write_text('12000')
            (hw / 'curr1_input').write_text('2000')
            self.assertEqual(telemetry.rail_watts(telemetry.find_input_rail(root)), 24)
            (hw / 'in1_label').unlink()
            old = root / 'bus/i2c/drivers/ina3221x/6-0040/iio:device0'
            old.mkdir(parents=True)
            (old / 'rail_name_0').write_text('VDD_IN')
            (old / 'in_power0_input').write_text('4500')
            self.assertEqual(telemetry.rail_watts(telemetry.find_input_rail(root)), 4.5)

    def test_ambiguous_sensor_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for i in (0, 1):
                path = Path(d) / f'class/hwmon/hwmon{i}'
                path.mkdir(parents=True)
                (path / 'in1_label').write_text('VDD_IN')
            with self.assertRaises(RuntimeError):
                telemetry.find_input_rail(Path(d))


class QualityTests(unittest.TestCase):
    def test_exact_match_and_failure_accounting(self):
        from edge_llm_pareto.quality import evaluate
        client = Mock()
        client.chat.completions.create = AsyncMock(side_effect=[
            types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=' 42\n'))]),
            RuntimeError('server failure')])
        manager = Mock(__aenter__=AsyncMock(return_value=client), __aexit__=AsyncMock())
        fake = types.SimpleNamespace(AsyncOpenAI=Mock(return_value=manager))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'tasks.jsonl'
            path.write_text('\n'.join(json.dumps(dict(id=str(i), prompt='question', answers=['42'])) for i in range(2)))
            args = types.SimpleNamespace(dataset=str(path), output=str(Path(d) / 'quality.json'),
                base_url='', model='m', model_repo='repo', model_revision='sha', base_model='base',
                engine='llama_cpp', rung='q4', max_tokens=64)
            with patch.dict('sys.modules', {'openai': fake}):
                self.assertEqual(asyncio.run(evaluate(args)), 1)
            record = json.loads(Path(args.output).read_text())
            self.assertEqual(record['score'], .5)
            self.assertEqual(record['failed'], 1)


class RecoveryTests(unittest.TestCase):
    def test_only_partial_final_record_is_recovered(self):
        from edge_llm_pareto.results import read_records
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'result.jsonl'
            p.write_bytes(b'{"trial": 1}\n{"trial":')
            with self.assertWarns(UserWarning):
                self.assertEqual(read_records(p, repair=True), [{'trial': 1}])
            self.assertEqual(p.read_bytes(), b'{"trial": 1}\n')
            p.write_bytes(b'broken\n{"trial": 1}\n')
            with self.assertRaises(json.JSONDecodeError):
                read_records(p, repair=True)

    def test_complete_unterminated_record_gets_newline(self):
        from edge_llm_pareto.results import read_records
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'result.jsonl'
            p.write_text('{"trial": 1}')
            self.assertEqual(read_records(p, repair=True), [{'trial': 1}])
            self.assertTrue(p.read_bytes().endswith(b'\n'))


class EnergyBenchmarkIntegrationTests(unittest.TestCase):
    def test_measurement_window_excludes_warmup_and_normalizes_exact_tokens(self):
        class Power:
            samples = [dict(t=0, power_w=10), dict(t=4, power_w=10)]
            interval = 1
            source, scope, errors = 'fake', 'module', []
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                pass
        client = Mock(close=AsyncMock())
        fake = types.SimpleNamespace(AsyncOpenAI=Mock(return_value=client))
        request = bench.request_result(True, token_count_source='server', input_tokens=5,
                                      output_tokens=10, latency_s=1, ttft_s=.1)
        with patch.dict('sys.modules', {'openai': fake}), \
                patch.object(bench, 'PowerMonitor', Power), \
                patch.object(bench.time, 'perf_counter', side_effect=[1,3]), \
                patch.object(bench, '_one_request', AsyncMock(return_value=request)):
            row = asyncio.run(bench.run_concurrency_level('', 'm', 1, bench.make_workload(), 2, warmup_requests=3))
        self.assertEqual(row['run_energy_j'], 20)
        self.assertEqual(row['output_tokens'], 20)
        self.assertEqual(row['input_tokens'], 10)
        self.assertEqual(row['joules_per_output_token'], 1)
        self.assertEqual(row['requests'], 2)



class RequestDictionaryTests(unittest.TestCase):
    def test_stream_usage_and_prompt_are_preserved(self):
        async def chunks():
            yield types.SimpleNamespace(
                usage=None, choices=[types.SimpleNamespace(
                    delta=types.SimpleNamespace(content="hello"))])
            yield types.SimpleNamespace(
                usage=types.SimpleNamespace(completion_tokens=3, prompt_tokens=12),
                choices=[])
        client = Mock()
        client.chat.completions.create = AsyncMock(return_value=chunks())
        workload = bench.make_workload(prompt_tokens=64, max_output_tokens=8)
        result = asyncio.run(bench._one_request(client, "model", workload, 7))
        self.assertTrue(result["ok"])
        self.assertEqual(result["output_tokens"], 3)
        self.assertEqual(result["input_tokens"], 12)
        self.assertEqual(result["token_count_source"], "server")
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["max_tokens"], 8)
        self.assertEqual(kwargs["messages"][0]["content"], bench.make_prompt(workload, 7))

    def test_failure_keeps_complete_record_and_independent_lists(self):
        client = Mock()
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError("offline"))
        result = asyncio.run(bench._one_request(client, "model", bench.make_workload(), 2))
        self.assertFalse(result["ok"])
        self.assertEqual(result["request_id"], 2)
        self.assertIn("offline", result["error"])
        self.assertEqual(result["output_tokens"], 0)
        self.assertIsNone(result["input_tokens"])
        other = bench.request_result(False)
        result["itls_s"].append(1)
        self.assertEqual(other["itls_s"], [])


if __name__ == "__main__":
    unittest.main()
