#!/usr/bin/env python3
"""Sample every power sensor the benchmark uses and print what each one measured.

Run once at idle and once while a model is generating, then compare:

    .venv/bin/python scripts/check_power.py --seconds 10
    .venv/bin/python scripts/check_power.py --seconds 10 --json > idle.json

Uses the same PowerMonitor and integration as the benchmark, so what this
prints is what a trial would record.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_llm_pareto.gsmm import power_record  # noqa: E402
from edge_llm_pareto.telemetry import PowerMonitor  # noqa: E402


def fmt(x, unit='', nd=2):
    return '-' if x is None else f'{x:.{nd}f}{unit}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seconds', type=float, default=10)
    ap.add_argument('--interval', type=float, default=.1)
    ap.add_argument('--json', action='store_true', help='print the full record without raw samples')
    args = ap.parse_args()

    with PowerMonitor(interval=args.interval) as mon:
        t0 = time.perf_counter()
        time.sleep(args.seconds)
        t1 = time.perf_counter()
    r = power_record(mon, t0, t1)
    if args.json:
        print(json.dumps({k: v for k, v in r.items() if k != 'power_samples'}, indent=2))
        return 0

    obs = r['sampling_observations']
    print(f"window {t1 - t0:.2f} s, {obs['sample_count']} samples, "
          f"mean interval {fmt(obs['mean_interval_s'], ' s', 3)}, max {fmt(obs['max_interval_s'], ' s', 3)}")
    print(f"NVML ({r['energy_source']}): {fmt(r['run_energy_j'], ' J')}, "
          f"mean {fmt(r['run_energy_j'] / (t1 - t0) if r['energy_valid'] else None, ' W')}, "
          f"valid={r['energy_valid']}")
    if r['telemetry_errors']:
        print('errors:', '; '.join(r['telemetry_errors']))
    s = r.get('spbm')
    if not s:
        print('\nno spbm hwmon device found; only NVML is recorded')
        return 0
    print(f"\nspbm at {s['sensor']['device']}, read time mean {fmt(s['read_s_mean'] and s['read_s_mean'] * 1e3, ' ms')} "
          f"max {fmt(s['read_s_max'] and s['read_s_max'] * 1e3, ' ms')}")
    print(f"\n{'rail':<12}{'file':<10}{'energy J':>12}{'mean W':>10}{'max W':>10}")
    for label, rail in s['rails'].items():
        print(f"{label:<12}{s['sensor']['channels']['power'][label]:<10}{fmt(rail['energy_j']):>12}"
              f"{fmt(rail['mean_w']):>10}{fmt(rail['max_w']):>10}")
    print(f"\n{'counter':<12}{'file':<10}{'energy J':>12}{'updates every':>16}")
    for label, c in s['counters'].items():
        print(f"{label:<12}{s['sensor']['channels']['energy'][label]:<10}{fmt(c['energy_j']):>12}"
              f"{fmt(c['update_interval_s'], ' s', 3):>16}")
    print('\npower limits (live / cap):')
    for label, lim in s['power_limits'].items():
        cap = s['sensor']['limit_values'].get(label, {}).get('cap_w')
        print(f"  {label:<8}{fmt(lim['mean_w'], ' W'):>10} / {fmt(cap, ' W')}")
    print('\ntemperatures (mean / max):')
    for label, t in s['temperatures_c'].items():
        print(f"  {label:<12}{fmt(t['mean'], ' C', 1):>8} / {fmt(t['max'], ' C', 1)}")
    print('\ncross-checks (1.00 = agree):')
    for name, value in s['crosscheck'].items():
        print(f"  {name:<40}{fmt(value, '', 3)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
