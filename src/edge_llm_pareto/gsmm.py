"""GSMM-informed reference scenarios, provenance and explicit energy estimates."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import importlib.metadata
import math
import platform
import statistics
import subprocess
import sys
import time
import uuid

from .energy import integrate
from . import spbm as spbm_sensor
from .telemetry import PowerMonitor


def environment_metadata():
    """Read software versions without importing an inference stack or env secrets."""
    packages = {}
    for name in ('edge-llm-pareto', 'openai', 'huggingface-hub', 'nvidia-ml-py',
                 'torch', 'vllm', 'vllm-gguf-plugin'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    tools = {}
    for name, command in (
        ('cuda_toolkit', ['nvcc', '--version']),
        ('nvidia_driver', ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']),
    ):
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=3, check=True)
            tools[name] = dict(value=result.stdout.strip(), error=None)
        except (OSError, subprocess.SubprocessError) as exc:
            tools[name] = dict(value=None, error=str(exc))
    return dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),
                host=platform.node(), os=platform.platform(),
                kernel=platform.release(), architecture=platform.machine(),
                python=sys.version, python_executable=sys.executable,
                client_packages=packages, tools=tools,
                runtime_stack_note='Client environment; separate server/container versions require session notes.')


def sensor_details(monitor):
    return dict(source=monitor.source, scope=monitor.scope,
                requested_interval_s=monitor.interval,
                accuracy='unknown; requires device documentation or calibration',
                calibration='not validated',
                scope_validation='not validated on this hardware',
                timestamp_clock='time.perf_counter; host monotonic seconds')


def power_record(monitor, start, end):
    """Use the same integration and scope metadata for every scenario."""
    result = integrate(monitor.samples, start, end, monitor.interval * 5)
    gaps = [b['t'] - a['t'] for a, b in zip(monitor.samples, monitor.samples[1:])
            if b['t'] > a['t']]
    masks = [s.get('throttle_reasons') for s in monitor.samples]
    result.update(power_samples=monitor.samples, energy_source=monitor.source,
                  energy_scope=monitor.scope, telemetry_errors=sorted(set(monitor.errors)),
                  sensor_metadata=sensor_details(monitor),
                  measurement_start=start, measurement_end=end,
                  sampling_observations=dict(sample_count=len(monitor.samples),
                      mean_interval_s=statistics.fmean(gaps) if gaps else None,
                      max_interval_s=max(gaps) if gaps else None),
                  unexpected_thermal_throttling=(any(m & (32 | 64) for m in masks if m is not None)
                    if any(m is not None for m in masks) else None))
    result.update(spbm_record(monitor, start, end, result))
    return result


def spbm_record(monitor, start, end, nvml):
    """Per-rail energy from the Spark spbm device, plus cross-checks.

    Adds nothing when the device is absent, so other platforms keep their schema.
    Top-level fields name the scope; the full per-rail detail sits under `spbm`.
    """
    device = getattr(monitor, 'spbm', None)
    if device is None:
        return {}
    samples, gap = monitor.samples, monitor.interval * 5
    rails = {}
    for label in device.channels['power']:
        def value(s, label=label):
            return (s.get('spbm_power_w') or {}).get(label)
        r = integrate(samples, start, end, gap, value)
        watts = [value(s) for s in samples if start <= s['t'] <= end and value(s) is not None]
        rails[label] = dict(energy_j=r['run_energy_j'], valid=r['energy_valid'],
                            coverage=r['energy_coverage'],
                            mean_w=r['run_energy_j'] / (end - start) if r['energy_valid'] else None,
                            max_w=max(watts) if watts else None)
    counters = {}
    for label in device.channels['energy']:
        counters[label] = dict(energy_j=spbm_sensor.counter_delta(samples, start, end, label),
                               update_interval_s=spbm_sensor.counter_update_interval(samples, label))
    temps = {}
    for label in device.channels['temp']:
        values = [s['spbm_temp_c'][label] for s in samples
                  if start <= s['t'] <= end and label in (s.get('spbm_temp_c') or {})]
        if values:
            temps[label] = dict(mean=statistics.fmean(values), max=max(values))
    limits = {}
    for label in device.limits:
        values = [s['spbm_limit_w'][label] for s in samples
                  if start <= s['t'] <= end and label in (s.get('spbm_limit_w') or {})]
        if values:
            limits[label] = dict(mean_w=statistics.fmean(values), max_w=max(values))
    reads = [s['spbm_read_s'] for s in samples if s.get('spbm_read_s') is not None]
    nvml_j = nvml.get('run_energy_j') if nvml.get('energy_scope') == 'nvml_device' else None

    def rail_j(label):
        return (rails.get(label) or {}).get('energy_j')

    checks = dict(
        spbm_gpu_over_nvml=spbm_sensor.ratio(rail_j('gpu'), nvml_j),
        **{f'{c}_counter_over_{r}_integrated': spbm_sensor.ratio(
            (counters.get(c) or {}).get('energy_j'), rail_j(r))
           for c, r in spbm_sensor.COUNTER_CHECKS.items()})
    out = {f'{scope}_energy_j': rail_j(label) for scope, label in spbm_sensor.SCOPES.items()}
    out['spbm'] = dict(
        sensor=device.describe(), rails=rails, counters=counters,
        temperatures_c=temps, power_limits=limits, crosscheck=checks,
        read_s_mean=statistics.fmean(reads) if reads else None,
        read_s_max=max(reads) if reads else None,
        rail_meaning='inferred from driver labels; not confirmed by driver documentation')
    return out


async def measure_reference(scenario, seconds, interval=.1):
    if scenario not in ('system_baseline', 'server_idle'):
        raise ValueError('Unknown reference scenario')
    if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(interval) or interval <= 0:
        raise ValueError('Reference duration and sampling interval must be finite and positive')
    recorded_at = datetime.now(timezone.utc).isoformat()
    with PowerMonitor(interval=interval) as monitor:
        start = time.perf_counter()
        await asyncio.sleep(seconds)
        end = time.perf_counter()
    record = power_record(monitor, start, end)
    duration = end - start
    values = [s['power_w'] for s in monitor.samples if start <= s['t'] <= end
              and s.get('power_w') is not None and math.isfinite(s['power_w']) and s['power_w'] >= 0]
    record.update(scenario=scenario, reference_id=uuid.uuid4().hex,
                  recorded_at_utc=recorded_at, requested_duration_s=seconds,
                  wall_s=duration,
                  mean_power_w=record['run_energy_j'] / duration if record['energy_valid'] else None,
                  power_sample_std_w=statistics.stdev(values) if len(values) > 1 else None,
                  spbm_mean_power_w={label: rail['mean_w'] for label, rail in
                                     record.get('spbm', {}).get('rails', {}).items()} or None,
                  stationarity_validation='requires pilot trace review',
                  external_activity_validation='not verified')
    return record


def adjusted_energy(load, reference, scenario):
    """Subtract reference mean watts × load duration; never alter gross fields.

    Negative estimates are retained: they expose drift or an unsuitable reference.
    No attribution or uncertainty guarantee follows from this arithmetic.
    """
    result = dict(reference_id=reference.get('reference_id') if reference else None,
                  method='gross_energy_minus_reference_mean_power_times_duration',
                  valid=False, reason=None, energy_j=None,
                  joules_per_output_token=None, negative_estimate=False)
    if not reference:
        result['reason'] = 'reference_not_collected'
        return result
    result['reference_scenario'] = reference.get('scenario')
    if reference.get('scenario') != scenario:
        result['reason'] = 'reference_scenario_mismatch'
        return result
    for key in ('energy_source', 'energy_scope', 'sensor_metadata'):
        if load.get(key) is None or load.get(key) != reference.get(key):
            result['reason'] = 'sensor_mismatch'
            return result
    energy, power = load.get('run_energy_j'), reference.get('mean_power_w')
    if (not load.get('energy_valid') or not reference.get('energy_valid') or
        any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in (energy, power)) or
        reference.get('unexpected_thermal_throttling') or load.get('unexpected_thermal_throttling')):
        result['reason'] = 'invalid_energy_or_throttling'
        return result
    duration = load['measurement_end'] - load['measurement_start']
    if not math.isfinite(duration) or duration <= 0:
        result['reason'] = 'invalid_duration'
        return result
    if reference['measurement_end'] > load['measurement_start']:
        result['reason'] = 'reference_after_load_start'
        return result
    estimate = energy - power * duration
    result.update(valid=True, energy_j=estimate, negative_estimate=estimate < 0,
                  reference_mean_power_w=power, load_duration_s=duration,
                  reference_age_s=load['measurement_start'] - reference['measurement_end'])
    tokens = load.get('output_tokens', 0)
    if load.get('token_counts_exact') and not load.get('failed') and tokens > 0:
        result['joules_per_output_token'] = estimate / tokens
    return result


def scope_energy_fields(load, references, tokens_ok):
    """J/token per spbm scope, and gross minus the reference's mean power at that scope.

    `tokens_ok` carries the same rule as the NVML J/token: exact counts, no failures.
    """
    out = {}
    tokens = load.get('output_tokens') or 0
    duration = load['measurement_end'] - load['measurement_start']
    for scope, label in spbm_sensor.SCOPES.items():
        energy = load.get(f'{scope}_energy_j')
        out[f'joules_per_output_token_{scope}'] = (
            energy / tokens if energy is not None and tokens_ok and tokens > 0 else None)
        for name, reference in references.items():
            watts = ((reference or {}).get('spbm_mean_power_w') or {}).get(label)
            usable = (energy is not None and watts is not None and duration > 0
                      and reference['measurement_end'] <= load['measurement_start'])
            adjusted = energy - watts * duration if usable else None
            out[f'{name}_adjusted_{scope}_energy_j'] = adjusted
            out[f'{name}_adjusted_joules_per_output_token_{scope}'] = (
                adjusted / tokens if adjusted is not None and tokens_ok and tokens > 0 else None)
    return out
