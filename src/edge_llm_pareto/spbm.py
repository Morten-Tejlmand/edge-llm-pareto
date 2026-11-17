"""Read the DGX Spark `spbm` hwmon device: per-rail power, energy counters, temperatures.

The device exposes more scopes than NVML (whole system, SoC package, GPU, CPU
clusters). It is an undocumented driver, so every reading is recorded under its
own label and cross-checked against NVML rather than trusted blindly.

Units follow the hwmon ABI: power in microwatts, energy in microjoules,
temperature in millidegrees Celsius.
"""
import math
from pathlib import Path

NAME = 'spbm'
# Labels whose integrated power gets a top-level energy field. Rail meanings
# are inferred from labels and still need confirmation from the driver author.
SCOPES = {'system': 'dc_input', 'system_total': 'sys_total', 'soc': 'soc_pkg', 'gpu': 'gpu'}
# Counter labels that should agree with integrated power of a rail.
COUNTER_CHECKS = {'gpu': 'gpu', 'pkg': 'soc_pkg'}


def find_device(root=Path('/sys')):
    for d in sorted(root.glob('class/hwmon/hwmon*')):
        try:
            if (d / 'name').read_text().strip() == NAME:
                return d
        except OSError:
            continue
    return None


class Spbm:
    """Discovered once; each read returns {label: value} per sensor kind."""

    KINDS = {'power': 1e-6, 'energy': 1e-6, 'temp': 1e-3}

    def __init__(self, device):
        self.device = Path(device)
        self.channels = {kind: {} for kind in self.KINDS}
        self.limits = {}
        for label_file in sorted(self.device.glob('*_label')):
            prefix = label_file.name[:-len('_label')]
            kind = prefix.rstrip('0123456789')
            if kind not in self.KINDS or not (self.device / f'{prefix}_input').exists():
                continue
            label = label_file.read_text().strip()
            if (self.device / f'{prefix}_cap').exists():
                # Power-limit channels (pl1, syspl1, ...): their inputs are kept
                # separate from rails so nobody integrates a limit as energy.
                self.limits[label] = prefix
                continue
            self.channels[kind][label] = prefix

    def read(self, kind):
        scale, out = self.KINDS[kind], {}
        for label, prefix in self.channels[kind].items():
            out[label] = int((self.device / f'{prefix}_input').read_text()) * scale
        return out

    def read_limit_inputs(self):
        return {label: int((self.device / f'{p}_input').read_text()) * 1e-6
                for label, p in self.limits.items()}

    def limit_metadata(self):
        out = {}
        for label, prefix in self.limits.items():
            row = {}
            for field in ('cap', 'max', 'min'):
                path = self.device / f'{prefix}_{field}'
                try:
                    row[field + '_w'] = int(path.read_text()) * 1e-6
                except (OSError, ValueError):
                    row[field + '_w'] = None
            out[label] = row
        return out

    def describe(self):
        return dict(device=str(self.device), name=NAME,
                    channels={k: dict(v) for k, v in self.channels.items()},
                    limits=dict(self.limits), limit_values=self.limit_metadata())


def counter_delta(samples, start, end, label):
    """Energy from a cumulative counter, linearly interpolated at the window edges.

    Returns None if the window is not bracketed by readings, the counter went
    backwards (wrap or reset), or any reading inside the window is missing.
    """
    points = [(s['t'], s['spbm_energy_j'][label]) for s in samples
              if label in (s.get('spbm_energy_j') or {})]
    points.sort()

    def at(t):
        for (a, ea), (b, eb) in zip(points, points[1:]):
            if a <= t <= b:
                return ea if b == a else ea + (eb - ea) * (t - a) / (b - a)
        return None

    inside = [e for t, e in points if start <= t <= end]
    lo, hi = at(start), at(end)
    if lo is None or hi is None:
        return None
    seq = [lo] + inside + [hi]
    if any(b < a for a, b in zip(seq, seq[1:])):
        return None
    return hi - lo


def counter_update_interval(samples, label):
    """Median time between changes of a counter, to show its real resolution."""
    changes, last = [], None
    for s in samples:
        value = (s.get('spbm_energy_j') or {}).get(label)
        if value is None:
            continue
        if last is not None and value != last[1]:
            changes.append(s['t'] - last[0])
        if last is None or value != last[1]:
            last = (s['t'], value)
    if not changes:
        return None
    changes.sort()
    return changes[len(changes) // 2]


def ratio(a, b):
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b) or b == 0:
        return None
    return a / b
