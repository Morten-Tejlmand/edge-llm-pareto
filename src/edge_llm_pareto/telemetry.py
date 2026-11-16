"""One sampling thread for power, GPU activity, clocks, and memory.

Sensors are read only; power modes are configured externally.
"""
import subprocess
import threading
import time
from pathlib import Path


def is_jetson():
    p = Path('/proc/device-tree/model')
    return p.exists() and 'jetson' in p.read_text().lower()


def power_mode():
    if not is_jetson():
        return None
    try:
        return subprocess.run(['nvpmodel', '-q'], capture_output=True, text=True,
                              timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def find_input_rail(root=Path('/sys')):
    labels = list(root.glob('class/hwmon/hwmon*/in*_label'))
    labels += list(root.glob('bus/i2c/drivers/ina3221x/*/iio*/rail_name_*'))
    rails = [p for p in labels if p.read_text().strip() in ('VDD_IN', 'POM_5V_IN')]
    if len(rails) != 1:
        raise RuntimeError('Expected one module input rail; unsupported or ambiguous sensor layout')
    return rails[0]


def rail_watts(rail):
    if rail.name.startswith('rail_name_'):
        channel = rail.name.rsplit('_', 1)[1]
        return float((rail.parent / f'in_power{channel}_input').read_text()) / 1000
    channel = rail.name.split('_')[0][2:]
    return (float((rail.parent / f'in{channel}_input').read_text()) *
            float((rail.parent / f'curr{channel}_input').read_text()) / 1e6)


class PowerMonitor:
    def __init__(self, interval=0.1, index=0):
        self.index = index
        self.unified = False
        self.interval = interval
        self.samples = []
        self.errors = []
        self.source = None
        self.scope = None
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.nvml = None

    def __enter__(self):
        try:
            if is_jetson():
                self.rail = find_input_rail()
                self.source, self.scope = str(self.rail), 'module'
            else:
                import pynvml
                pynvml.nvmlInit()
                self.nvml = pynvml
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
                self.source, self.scope = f'NVML:device{self.index}:power', 'nvml_device'
                name = pynvml.nvmlDeviceGetName(self.handle)
                name = name.decode() if isinstance(name, bytes) else name
                self.unified = 'GB10' in name.upper()
            self.sample()
        except Exception as exc:
            self.errors.append(str(exc))
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        with self.lock:
            row = {'t': time.perf_counter(), 'power_w': None}
            try:
                if self.scope == 'module':
                    row['power_w'] = rail_watts(self.rail)
                    from .hardware import system_memory_mib
                    total, available = system_memory_mib()
                    row['system_used_mib'] = total - available
                    row['temperatures_c'] = {str(p): float(p.read_text()) / 1000
                        for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')}
                    row['clocks_hz'] = {str(p): int(p.read_text()) for p in
                                       Path('/sys/class/devfreq').glob('*/cur_freq')}
                    row['cpu_clocks_khz'] = {str(p): int(p.read_text()) for p in
                        Path('/sys/devices/system/cpu').glob('cpu*/cpufreq/scaling_cur_freq')}
                    row['throttle_status'] = 'unknown'
                elif self.nvml:
                    n, h = self.nvml, self.handle
                    # Each counter is independent: unsupported power or memory
                    # must not discard utilization, clocks, or temperature.
                    getters = (
                        ('power_w', lambda: n.nvmlDeviceGetPowerUsage(h) / 1000),
                        ('temperature_c', lambda: n.nvmlDeviceGetTemperature(h, 0)),
                        ('clock_mhz', lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM)),
                        ('throttle_reasons', lambda: n.nvmlDeviceGetCurrentClocksThrottleReasons(h)),
                    )
                    for key, getter in getters:
                        try:
                            row[key] = getter()
                        except Exception as exc:
                            row[key] = None
                            self.errors.append(f"{key}: {exc}")
                    try:
                        activity = n.nvmlDeviceGetUtilizationRates(h)
                        row['sm_util_pct'] = activity.gpu
                        row['mem_util_pct'] = activity.memory
                    except Exception as exc:
                        self.errors.append(f"utilization: {exc}")
                    try:
                        if self.unified:
                            from .hardware import system_memory_mib
                            total, available = system_memory_mib()
                            row['system_used_mib'] = total - available
                        else:
                            row['mem_used_mib'] = n.nvmlDeviceGetMemoryInfo(h).used // 1024**2
                    except Exception as exc:
                        self.errors.append(f"memory: {exc}")
                row['t'] = time.perf_counter()
            except Exception as exc:
                self.errors.append(str(exc))
            self.samples.append(row)

    def _run(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.sample()
        if self.nvml:
            self.nvml.nvmlShutdown()


    def summary(self):
        """Keep the existing GPU summary fields for reports and saved results."""
        out = {"samples": sum(s.get("sm_util_pct") is not None for s in self.samples)}
        for source, name in (
            ("sm_util_pct", "sm_util_pct"), ("mem_util_pct", "mem_util_pct"),
            ("mem_used_mib", "mem_used_mib"), ("system_used_mib", "system_used_mib"),
            ("power_w", "power_w"), ("clock_mhz", "sm_clock_mhz"),
        ):
            values = [s[source] for s in self.samples if s.get(source) is not None]
            if values:
                out[name + "_mean"] = round(sum(values) / len(values), 1)
                out[name + "_max"] = round(max(values), 1)
        return out
