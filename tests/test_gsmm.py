"""GSMM reference timing, attribution and invalid-data handling without hardware."""
import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from edge_llm_pareto import bench, gsmm
from edge_llm_pareto.analysis import energy_exclusions, aggregate
import test_llama_cpp


class Power:
    interval = 1
    source, scope, errors = 'fake', 'module', []
    samples = [dict(t=0, power_w=20), dict(t=6, power_w=20)]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def reference(scenario='system_baseline', watts=5):
    return dict(reference_id='ref-' + scenario, scenario=scenario,
        energy_valid=True, run_energy_j=watts, mean_power_w=watts,
        energy_source='fake', energy_scope='module', sensor_metadata=gsmm.sensor_details(Power()),
        measurement_start=0, measurement_end=1, requested_duration_s=1,
        power_samples=[dict(t=0, power_w=watts), dict(t=1, power_w=watts)])


def load():
    return dict(run_energy_j=40, energy_valid=True, output_tokens=10,
        token_counts_exact=True, failed=0, measurement_start=3, measurement_end=5,
        energy_source='fake', energy_scope='module', sensor_metadata=gsmm.sensor_details(Power()))


class AdjustmentTests(unittest.TestCase):
    def test_adjustment_preserves_gross_energy_and_normalizes_tokens(self):
        row = load()
        result = gsmm.adjusted_energy(row, reference(), 'system_baseline')
        self.assertTrue(result['valid'])
        self.assertEqual(result['energy_j'], 30)
        self.assertEqual(result['joules_per_output_token'], 3)
        self.assertEqual(row['run_energy_j'], 40)
        self.assertEqual(result['reference_age_s'], 2)

    def test_missing_invalid_or_mismatched_references_do_not_become_zero(self):
        for ref in (None, dict(reference(), energy_valid=False),
                    dict(reference(), energy_scope='wall'),
                    dict(reference(), mean_power_w=float('nan')),
                    dict(reference(), scenario='server_idle'),
                    dict(reference(), unexpected_thermal_throttling=True),
                    dict(reference(), measurement_end=10)):
            with self.subTest(ref=ref):
                result = gsmm.adjusted_energy(load(), ref, 'system_baseline')
                self.assertFalse(result['valid'])
                self.assertIsNone(result['energy_j'])
                self.assertTrue(result['reason'])

    def test_negative_estimates_are_retained_and_flagged(self):
        result = gsmm.adjusted_energy(load(), reference(watts=30), 'system_baseline')
        self.assertEqual(result['energy_j'], -20)
        self.assertTrue(result['negative_estimate'])

    def test_failed_or_approximate_requests_have_no_adjusted_j_per_token(self):
        for change in (dict(failed=1), dict(token_counts_exact=False)):
            result = gsmm.adjusted_energy(dict(load(), **change), reference(), 'system_baseline')
            self.assertIsNone(result['joules_per_output_token'])


class ReferenceTests(unittest.TestCase):
    def test_reference_integrates_actual_duration_and_retains_samples(self):
        power = Power()
        power.samples = [dict(t=0, power_w=10), dict(t=4, power_w=10)]
        with patch.object(gsmm, 'PowerMonitor', return_value=power), \
                patch.object(gsmm.time, 'perf_counter', side_effect=[1, 3]), \
                patch.object(gsmm.asyncio, 'sleep', AsyncMock()) as sleep:
            row = asyncio.run(gsmm.measure_reference('server_idle', 2, 1))
        sleep.assert_awaited_once_with(2)
        self.assertEqual(row['wall_s'], 2)
        self.assertEqual(row['run_energy_j'], 20)
        self.assertEqual(row['mean_power_w'], 10)
        self.assertEqual(row['power_samples'], power.samples)
        self.assertEqual(row['sensor_metadata']['calibration'], 'not validated')

    def test_idle_follows_warmup_and_precedes_measured_requests(self):
        events = []
        client = Mock(close=AsyncMock())
        async def request(*args):
            events.append('request')
            return bench.request_result(True, token_count_source='server',
                output_tokens=10, latency_s=1, ttft_s=.1)
        async def idle(*args):
            events.append('idle')
            return reference('server_idle')
        power = Power()
        power.samples = [dict(t=2, power_w=20), dict(t=6, power_w=20)]
        with patch('openai.AsyncOpenAI', return_value=client), \
                patch.object(bench, '_one_request', side_effect=request), \
                patch.object(bench, 'measure_reference', side_effect=idle), \
                patch.object(bench, 'PowerMonitor', return_value=power), \
                patch.object(bench.time, 'perf_counter', side_effect=[3, 5]):
            row = asyncio.run(bench.run_concurrency_level('', 'model', 1,
                bench.make_workload(), 1, warmup_requests=2, idle_seconds=1,
                system_baseline=reference(watts=2)))
        self.assertEqual(events, ['request', 'request', 'idle', 'request'])
        self.assertEqual(row['requests'], 1)
        self.assertEqual(row['run_energy_j'], 40)
        self.assertEqual(row['joules_per_output_token'], 4)
        self.assertEqual(row['system_baseline_adjusted_energy_j'], 36)
        self.assertEqual(row['server_idle_adjusted_energy_j'], 30)
        self.assertIn('power_samples', row['gsmm_references']['server_idle'])
        client.close.assert_awaited_once()

    def test_cli_enables_reference_artifact_and_passes_idle_protocol(self):
        with tempfile.TemporaryDirectory() as d, patch(
                'edge_llm_pareto.gsmm.environment_metadata', return_value=dict(captured_at_utc='now')), \
                patch('edge_llm_pareto.gsmm.measure_reference', AsyncMock(return_value=reference())) as measure:
            rc, saved, start, run, _, _ = test_llama_cpp.LlamaSweepTests().run_cli(d,
                extra=['--gsmm', '--baseline-seconds', '1', '--idle-seconds', '2'])
            artifacts = list((Path(d) / 'scenarios').glob('*.jsonl'))
            self.assertEqual(len(artifacts), 1)
            scenario = json.loads(artifacts[0].read_text())
            self.assertEqual(scenario['scenario'], 'system_baseline')
        self.assertEqual(rc, 0)
        measure.assert_awaited_once_with('system_baseline', 1, .1)
        self.assertEqual(run.call_args.kwargs['idle_seconds'], 2)
        self.assertEqual(run.call_args.kwargs['system_baseline']['reference_id'], 'ref-system_baseline')
        self.assertEqual(saved['measurement_protocol']['energy_basis'], 'gross')
        start.assert_called_once()

    def test_cli_rejects_nonfinite_durations_before_hardware_probing(self):
        spec = importlib.util.spec_from_file_location('gsmm_cli', test_llama_cpp.ROOT / 'scripts/sweep.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        for args in (['--baseline-seconds', 'nan'], ['--idle-seconds', '-1'],
                     ['--sample-interval', 'inf']):
            with patch('sys.argv', ['sweep.py'] + args), patch.object(cli, 'probe_gpu') as probe:
                with self.assertRaises(SystemExit):
                    cli.main()
                probe.assert_not_called()

    def test_missing_environment_tools_are_recorded_without_failing(self):
        with patch.object(gsmm.subprocess, 'run', side_effect=FileNotFoundError('absent')):
            result = gsmm.environment_metadata()
        self.assertIsNone(result['tools']['cuda_toolkit']['value'])
        self.assertEqual(result['tools']['nvidia_driver']['error'], 'absent')
        self.assertIn('python', result)

    def test_energy_audit_identifies_missing_telemetry(self):
        reasons = energy_exclusions(dict(config_id='one', token_counts_exact=True))
        self.assertIn('missing_or_invalid_energy_valid', reasons)
        self.assertIn('missing_or_nonfinite_run_energy_j', reasons)
        self.assertEqual(aggregate([dict(config_id='one', token_counts_exact=True)]), [])

    def test_resume_keeps_reference_observations_out_of_configuration_identity(self):
        row = dict(output_tok_s=10, failed=0, energy_valid=True, token_counts_exact=True)
        with tempfile.TemporaryDirectory() as d, patch.object(
                bench, 'run_concurrency_level', AsyncMock(return_value=row)) as run:
            path = str(Path(d) / 'load.jsonl')
            kwargs = dict(out_path=path, system_baseline=reference(), idle_seconds=2,
                          metadata=dict(environment_id='stable', environment={'captured_at_utc': 'first'}))
            asyncio.run(bench.sweep('', 'm', [1], bench.make_workload(), 'q4', **kwargs))
            kwargs['system_baseline'] = dict(reference(), reference_id='fresh-reference')
            kwargs['metadata']['environment'] = {'captured_at_utc': 'later'}
            asyncio.run(bench.sweep('', 'm', [1], bench.make_workload(), 'q4', resume=True, **kwargs))
            self.assertEqual(run.await_count, 1)
            kwargs['idle_seconds'] = 3
            asyncio.run(bench.sweep('', 'm', [1], bench.make_workload(), 'q4', resume=True, **kwargs))
            self.assertEqual(run.await_count, 2)
