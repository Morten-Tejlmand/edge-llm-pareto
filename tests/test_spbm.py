import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_llm_pareto import analysis, gsmm, spbm, telemetry  # noqa: E402

# Layout copied from a DGX Spark `sensors` / sysfs listing.
POWER = {1: 'sys_total', 2: 'soc_pkg', 3: 'cpu_gpu', 4: 'cpu_p', 5: 'cpu_e', 6: 'vcore',
         7: 'dc_input', 8: 'gpu', 9: 'prereg', 10: 'dla',
         11: 'pl1', 12: 'pl2', 13: 'syspl1', 14: 'syspl2'}
ENERGY = {1: 'pkg', 2: 'cpu_e', 3: 'cpu_p', 4: 'gpu'}
TEMP = {1: 'tj_max', 2: 'cpu_e_clu0', 7: 'gpu', 8: 'soc'}


def write(path, value):
    path.write_text(f'{value}\n')


def fake_sysfs(root, watts=None):
    watts = watts or {}
    other = root / 'class/hwmon/hwmon1'
    other.mkdir(parents=True)
    write(other / 'name', 'nvme')
    d = root / 'class/hwmon/hwmon6'
    d.mkdir(parents=True)
    write(d / 'name', 'spbm')
    for i, label in POWER.items():
        write(d / f'power{i}_label', label)
        write(d / f'power{i}_input', int(watts.get(label, 10) * 1e6))
        if i >= 11:
            for field, value in (('cap', 140e6), ('max', 250e6), ('min', 1e5)):
                write(d / f'power{i}_{field}', int(value))
    for i, label in ENERGY.items():
        write(d / f'energy{i}_label', label)
        write(d / f'energy{i}_input', 1_000_000_000)
    for i, label in TEMP.items():
        write(d / f'temp{i}_label', label)
        write(d / f'temp{i}_input', 31_500)
    return d


class DiscoveryTests(unittest.TestCase):
    def test_finds_device_by_name_and_separates_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = fake_sysfs(Path(tmp), dict(dc_input=30.3, gpu=5.27))
            self.assertEqual(spbm.find_device(Path(tmp)), d)
            dev = spbm.Spbm(d)
            power = dev.read('power')
            self.assertAlmostEqual(power['dc_input'], 30.3)
            self.assertAlmostEqual(power['gpu'], 5.27)
            self.assertNotIn('pl1', power)
            self.assertEqual(set(dev.limits), {'pl1', 'pl2', 'syspl1', 'syspl2'})
            self.assertEqual(dev.limit_metadata()['pl1']['cap_w'], 140)
            self.assertEqual(dev.read('energy')['pkg'], 1000)
            self.assertEqual(dev.read('temp')['gpu'], 31.5)

    def test_absent_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(spbm.find_device(Path(tmp)))


class CounterTests(unittest.TestCase):
    def samples(self, values):
        return [dict(t=t, spbm_energy_j=dict(gpu=e)) for t, e in values]

    def test_interpolates_window_edges(self):
        s = self.samples([(0, 100), (1, 110), (2, 130)])
        self.assertAlmostEqual(spbm.counter_delta(s, .5, 1.5, 'gpu'), 15)

    def test_wrap_or_reset_is_invalid(self):
        s = self.samples([(0, 100), (1, 5), (2, 20)])
        self.assertIsNone(spbm.counter_delta(s, 0, 2, 'gpu'))

    def test_unbracketed_window_is_invalid(self):
        s = self.samples([(1, 100), (2, 110)])
        self.assertIsNone(spbm.counter_delta(s, 0, 2, 'gpu'))

    def test_update_interval_reports_slow_counter(self):
        s = self.samples([(t / 10, 100 + t // 5) for t in range(21)])
        self.assertAlmostEqual(spbm.counter_update_interval(s, 'gpu'), .5)


class Monitor:
    interval = 1
    source, scope, errors = 'NVML:device0:power', 'nvml_device', []


def spbm_samples(device):
    samples = []
    for t in range(0, 11):
        samples.append(dict(
            t=t, power_w=5.0, throttle_reasons=0,
            spbm_power_w=dict(dc_input=30.0, sys_total=31.0, soc_pkg=18.0, gpu=5.0),
            spbm_energy_j=dict(gpu=1000 + 5.0 * t, pkg=2000 + 18.0 * t),
            spbm_temp_c=dict(gpu=40 + t), spbm_limit_w=dict(pl1=17.0), spbm_read_s=.002))
    return samples


class RecordTests(unittest.TestCase):
    def make(self, tmp):
        dev = spbm.Spbm(fake_sysfs(Path(tmp)))
        mon = Monitor()
        mon.spbm, mon.samples = dev, spbm_samples(dev)
        return mon

    def test_scopes_counters_and_crosschecks(self):
        with tempfile.TemporaryDirectory() as tmp:
            mon = self.make(tmp)
            r = gsmm.power_record(mon, 2, 6)
        self.assertEqual(r['run_energy_j'], 20)  # NVML unchanged
        self.assertEqual(r['system_energy_j'], 120)
        self.assertEqual(r['system_total_energy_j'], 124)
        self.assertEqual(r['soc_energy_j'], 72)
        self.assertEqual(r['gpu_energy_j'], 20)
        x = r['spbm']['crosscheck']
        self.assertAlmostEqual(x['spbm_gpu_over_nvml'], 1)
        self.assertAlmostEqual(x['gpu_counter_over_gpu_integrated'], 1)
        self.assertAlmostEqual(x['pkg_counter_over_soc_pkg_integrated'], 1)
        self.assertEqual(r['spbm']['temperatures_c']['gpu']['max'], 46)
        self.assertEqual(r['spbm']['power_limits']['pl1']['mean_w'], 17)
        self.assertEqual(r['spbm']['sensor']['limit_values']['pl1']['cap_w'], 140)
        # Rails missing in samples are reported invalid, not zero.
        self.assertFalse(r['spbm']['rails']['cpu_p']['valid'])
        self.assertIsNone(r['spbm']['rails']['cpu_p']['energy_j'])

    def test_no_device_leaves_schema_unchanged(self):
        mon = Monitor()
        mon.samples = [dict(t=0, power_w=5.0), dict(t=1, power_w=5.0)]
        r = gsmm.power_record(mon, 0, 1)
        self.assertNotIn('spbm', r)
        self.assertNotIn('system_energy_j', r)

    def test_scope_j_per_token_and_idle_adjustment(self):
        with tempfile.TemporaryDirectory() as tmp:
            mon = self.make(tmp)
            load = gsmm.power_record(mon, 6, 10)
            idle = gsmm.power_record(mon, 0, 2)
        idle.update(spbm_mean_power_w={k: v['mean_w'] for k, v in idle['spbm']['rails'].items()})
        load['output_tokens'] = 40
        out = gsmm.scope_energy_fields(load, dict(server_idle=idle, system_baseline=None), True)
        self.assertEqual(out['joules_per_output_token_system'], 3)
        self.assertEqual(out['server_idle_adjusted_system_energy_j'], 0)
        self.assertIsNone(out['system_baseline_adjusted_system_energy_j'])
        self.assertIsNone(gsmm.scope_energy_fields(load, {}, False)['joules_per_output_token_system'])

    def test_report_scope_repoints_energy_and_never_falls_back(self):
        row = dict(run_energy_j=20, joules_per_output_token=.5, energy_valid=True,
                   system_energy_j=120, joules_per_output_token_system=3)
        scoped = analysis.with_energy_scope(row, 'system')
        self.assertEqual(scoped['joules_per_output_token'], 3)
        self.assertEqual(scoped['energy_scope'], 'spbm:dc_input')
        missing = analysis.with_energy_scope(dict(run_energy_j=20, energy_valid=True), 'soc')
        self.assertFalse(missing['energy_valid'])
        self.assertIn('missing_or_invalid_energy_valid', analysis.energy_exclusions(missing))
        self.assertIs(analysis.with_energy_scope(row, 'nvml'), row)


class MonitorTests(unittest.TestCase):
    def test_monitor_samples_spbm_next_to_nvml(self):
        fake = types.SimpleNamespace(
            NVML_CLOCK_SM=0, nvmlInit=Mock(), nvmlShutdown=Mock(),
            nvmlDeviceGetHandleByIndex=Mock(return_value=1),
            nvmlDeviceGetName=Mock(return_value='RTX'),
            nvmlDeviceGetPowerUsage=Mock(return_value=5000),
            nvmlDeviceGetClockInfo=Mock(return_value=2000),
            nvmlDeviceGetTemperature=Mock(return_value=45),
            nvmlDeviceGetCurrentClocksThrottleReasons=Mock(return_value=0),
            nvmlDeviceGetUtilizationRates=Mock(return_value=types.SimpleNamespace(gpu=1, memory=1)),
            nvmlDeviceGetMemoryInfo=Mock(return_value=types.SimpleNamespace(used=0)))
        with tempfile.TemporaryDirectory() as tmp:
            d = fake_sysfs(Path(tmp), dict(dc_input=30))
            with patch.dict(sys.modules, {'pynvml': fake}), \
                    patch.object(telemetry, 'is_jetson', return_value=False), \
                    patch.object(spbm, 'find_device', return_value=d):
                with telemetry.PowerMonitor(interval=60) as mon:
                    pass
        self.assertEqual(mon.samples[0]['power_w'], 5)
        self.assertEqual(mon.samples[0]['spbm_power_w']['dc_input'], 30)
        self.assertIn('pl1', mon.samples[0]['spbm_limit_w'])
        self.assertEqual(mon.errors, [])

    def test_spbm_failure_does_not_stop_nvml(self):
        with patch.object(telemetry, 'is_jetson', return_value=False), \
                patch.object(spbm, 'find_device', side_effect=PermissionError('denied')), \
                patch.dict(sys.modules, {'pynvml': None}):
            with telemetry.PowerMonitor(interval=60) as mon:
                pass
        self.assertTrue(any('spbm' in e for e in mon.errors))


if __name__ == '__main__':
    unittest.main()
