"""Comparable trial aggregation, bootstrap intervals, and energy Pareto selection."""
import math
import random
import statistics
from collections import defaultdict
from .bench import stable_id
from .spbm import SCOPES as SPBM_SCOPES

METRICS = ('output_tok_s', 'latency_s_p95', 'ttft_s_p95', 'joules_per_output_token', 'run_energy_j')


def comparison_key(row, combine_models=False):
    hw = row.get('hardware', {})
    protocol = row.get('protocol', {})
    return stable_id(dict(device={k: hw.get(k) for k in ('name', 'total_mib', 'compute_cap')},
        device_id=row.get("device_id"), base_model=None if combine_models else row.get('base_model'), workload=row.get('workload'), context=row.get('context'),
        energy_scope=row.get('energy_scope'), energy_source=row.get('energy_source'),
        engine=row.get('engine'), backend=row.get('backend_version'),
        environment=row.get('environment_id'), server_parallel=row.get('server_parallel'),
        measurement_protocol=row.get('measurement_protocol'),
        code=row.get('code_fingerprint'), memory_budget=row.get('memory_budget'),
        requests_per_agent=protocol.get('requests_per_agent'), warmup=protocol.get('warmup_requests')))


def with_energy_scope(row, scope):
    """Re-point the energy fields at one spbm scope so the frontier ranks by it.

    Rows without that scope lose their energy and drop out as ineligible, never
    silently falling back to NVML.
    """
    if scope == 'nvml':
        return row
    label = SPBM_SCOPES[scope]
    energy = row.get(f'{scope}_energy_j')
    out = dict(row, run_energy_j=energy, energy_valid=energy is not None,
               joules_per_output_token=row.get(f'joules_per_output_token_{scope}'),
               energy_scope=f'spbm:{label}', energy_source='spbm')
    for name in ('system_baseline', 'server_idle'):
        out[f'{name}_adjusted_energy_j'] = row.get(f'{name}_adjusted_{scope}_energy_j')
        out[f'{name}_adjusted_joules_per_output_token'] = row.get(
            f'{name}_adjusted_joules_per_output_token_{scope}')
    return out


def interval(values):
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None, None
    rng = random.Random(0)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(2000))
    return mean, means[49], means[1949]


def energy_exclusions(row):
    """Eligibility for gross-energy analysis, before optional quality filters."""
    reasons = []
    for key in ('skipped', 'failed', 'throttled', 'unexpected_thermal_throttling'):
        if row.get(key):
            reasons.append(key)
    for key in ('energy_valid', 'token_counts_exact', 'config_id'):
        if not row.get(key):
            reasons.append('missing_or_invalid_' + key)
    for key in METRICS:
        if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]):
            reasons.append('missing_or_nonfinite_' + key)
    return reasons


def measured_quality_summary(trials):
    """Aggregate QA from every measured window, never reuse the first score."""
    qualities = [t.get('quality') for t in trials]
    if not all(q and q.get('evaluation_mode') == 'measured_qa' for q in qualities):
        raise ValueError('Measured QA requires quality results for every included trial')
    keys = ('scorer', 'answer_extraction', 'evaluation_mode', 'scoring_protocol',
            'dataset_sha256', 'max_tokens', 'output_limit_policy', 'temperature', 'dataset_questions')
    identities = {stable_id({k: q.get(k) for k in keys}) for q in qualities}
    if len(identities) != 1:
        raise ValueError('Measured QA trials must share one dataset and scoring protocol')
    summary = {k: qualities[0].get(k) for k in keys}
    summary.update(trials=len(qualities), aggregation='mean_of_trial_scores',
        requests=sum(q['requests'] for q in qualities),
        failed=sum(q['failed'] for q in qualities),
        incomplete_reasoning=sum(q['incomplete_reasoning'] for q in qualities),
        length_limited_requests=sum(q['length_limited_requests'] for q in qualities),
        minimum_unique_questions=min(q['unique_questions'] for q in qualities))
    summary['score'], summary['score_ci_low'], summary['score_ci_high'] = interval([q['score'] for q in qualities])
    summary['metrics'] = dict(exact_match=statistics.fmean(q['metrics']['exact_match'] for q in qualities),
        bleu=dict(score=statistics.fmean(q['metrics']['bleu']['score'] for q in qualities),
                  aggregation='mean_of_trial_corpus_bleu'))
    return summary


def aggregate(rows, combine_models=False):
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        if energy_exclusions(row):
            continue
        trial_id = row.get('trial_id')
        if trial_id in seen:
            continue
        seen.add(trial_id)
        groups[(comparison_key(row, combine_models), row['config_id'])].append(row)
    output = []
    for (group, config), trials in groups.items():
        first = trials[0]
        r = {k: first.get(k) for k in ('label', 'concurrency', 'engine', 'base_model', 'power_setting', 'quality', 'perplexity_evaluation', 'likelihood', 'server_parallel')}
        r.update(comparison_group=group, config_id=config, trials=len(trials))
        if any((t.get('quality') or {}).get('evaluation_mode') == 'measured_qa' for t in trials):
            r['quality'] = measured_quality_summary(trials)
        for metric in METRICS:
            values = [t[metric] for t in trials]
            r[metric], r[metric + '_ci_low'], r[metric + '_ci_high'] = interval(values)
            sd = statistics.stdev(values) if len(values) > 1 else None
            r[metric + '_std'] = sd
            r[metric + '_cv_pct'] = 100 * sd / abs(r[metric]) if sd is not None and r[metric] else None
        for metric in ('user_output_tok_s_mean', 'user_output_tok_s_p50', 'user_output_tok_s_p05',
                       'system_baseline_adjusted_energy_j', 'system_baseline_adjusted_joules_per_output_token',
                       'server_idle_adjusted_energy_j', 'server_idle_adjusted_joules_per_output_token',
                       'qa_score', 'qa_exact_match', 'qa_bleu', 'qa_incomplete_reasoning',
                       *(f'{p}joules_per_output_token_{scope}' for scope in SPBM_SCOPES
                         for p in ('', 'system_baseline_adjusted_', 'server_idle_adjusted_')),
                       *(f'{scope}_energy_j' for scope in SPBM_SCOPES)):
            values = [t[metric] for t in trials if isinstance(t.get(metric), (int, float)) and math.isfinite(t[metric])]
            r[metric + '_samples'] = len(values)
            if values:
                r[metric], r[metric + '_ci_low'], r[metric + '_ci_high'] = interval(values)
                sd = statistics.stdev(values) if len(values) > 1 else None
                r[metric + '_std'] = sd
                r[metric + '_cv_pct'] = 100 * sd / abs(r[metric]) if sd is not None and r[metric] else None
        output.append(r)
    return output


LIKELIHOOD_FIELDS = ('answer_ppl', 'answer_nll', 'answer_tokens',
                     'kl_to_engine_bf16', 'kl_to_engine_bf16_ci_low', 'kl_to_engine_bf16_ci_high',
                     'top1_agreement_engine_bf16', 'kl_to_llama_cpp_bf16',
                     'top1_agreement_llama_cpp_bf16')


def attach_likelihood(rows, bundle):
    """Add QA answer perplexity/KL (likelihood_report.py) to rows of the same artifact."""
    index = {}
    for rec in bundle['rows']:
        key = (rec['base_model'], rec['engine'], rec['label'])
        if key in index:
            raise ValueError(f'Duplicate likelihood record for {key}')
        index[key] = rec
    out = []
    for row in rows:
        rec = index.get((row.get('base_model'), row.get('engine'), row.get('label')))
        if rec and row.get('model_revision') not in (None, rec['model_revision']):
            raise ValueError(f"Likelihood revision differs from benchmark revision for {row.get('label')}")
        out.append(dict(row, likelihood={k: rec.get(k) for k in LIKELIHOOD_FIELDS} if rec else None))
    return out


# Energy per output token is always minimized; these name the other axes.
OBJECTIVES = {
    'output_tok_s': ('output_tok_s',),
    'latency_s_p95': ('latency_s_p95',),
    'ttft_s_p95': ('ttft_s_p95',),
    # Three-way frontier: J/token, throughput and TTFT together.
    'throughput_ttft': ('output_tok_s', 'ttft_s_p95'),
}
MAXIMIZE = {'output_tok_s'}
AXIS_LABELS = {'output_tok_s': 'Output tokens/s', 'latency_s_p95': 'p95 request latency (s)',
               'ttft_s_p95': 'p95 time to first token (s)'}


def frontier(rows, objective='output_tok_s'):
    """Configurations no other row in their comparison group beats on every axis."""
    if objective not in OBJECTIVES:
        raise ValueError(f'unknown objective {objective!r}; choose from {sorted(OBJECTIVES)}')
    axes = [('joules_per_output_token', 1)] + [(m, -1 if m in MAXIMIZE else 1) for m in OBJECTIVES[objective]]

    def dominates(o, r):
        pairs = [(sign * o[m], sign * r[m]) for m, sign in axes]
        return all(a <= b for a, b in pairs) and any(a < b for a, b in pairs)
    return [r for r in rows if not any(
        o['comparison_group'] == r['comparison_group'] and dominates(o, r) for o in rows)]


def export(rows, directory, objective='output_tok_s', baseline=None, min_quality=None,
           max_perplexity_increase_pct=None, combine_models=False):
    import csv
    import json
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)
    audit = []
    seen = set()
    for row in rows:
        reasons = energy_exclusions(row)
        if row.get('trial_id') in seen and not reasons:
            reasons.append('duplicate_trial')
        if not reasons:
            seen.add(row.get('trial_id'))
        audit.append(dict(trial_id=row.get('trial_id'), config_id=row.get('config_id'),
            engine=row.get('engine'), label=row.get('label'),
            eligible=not reasons, reasons=reasons))
    (dest / 'energy-eligibility.json').write_text(json.dumps(dict(
        scope='gross-energy eligibility before optional quality/perplexity filters',
        input_rows=len(rows), eligible_rows=sum(r['eligible'] for r in audit),
        rows=audit), indent=2))
    if max_perplexity_increase_pct is not None:
        if not math.isfinite(max_perplexity_increase_pct) or max_perplexity_increase_pct < 0:
            raise ValueError('maximum perplexity increase must be finite and nonnegative')
        rows = [r for r in rows if r.get('perplexity_evaluation') and
                r['perplexity_evaluation']['perplexity_increase_pct'] <= max_perplexity_increase_pct]
        protocols = {stable_id(r['perplexity_evaluation']['protocol']) for r in rows}
        if len(protocols) > 1:
            raise ValueError('Perplexity filtering requires one common evaluation protocol')
    data = aggregate(rows, combine_models)
    for r in data:
        r['comparison_mode'] = 'combined_models' if combine_models else 'within_model'
        r['quality_basis'] = ('shared_task_score' if min_quality is not None else
                             'relative_compression_loss_only' if max_perplexity_increase_pct is not None
                             else 'efficiency_only')
        if r.get('perplexity_evaluation'):
            r.update({k: r['perplexity_evaluation'][k] for k in
                      ('perplexity', 'reference_perplexity', 'perplexity_increase_pct')})
        for k, v in (r.pop('likelihood', None) or {}).items():
            r['qa_' + k] = v
    if min_quality is not None:
        if not 0 <= min_quality <= 1:
            raise ValueError('min-quality must be between 0 and 1')
        data = [r for r in data if r.get('quality') and r['quality'].get('score', -1) >= min_quality]
        datasets = {(r['quality'].get('dataset_sha256'), r['quality'].get('scorer'),
                     r['quality'].get('max_tokens'), r['quality'].get('temperature'),
                     r['quality'].get('evaluation_mode')) for r in data}
        if len(datasets) > 1:
            raise ValueError('Quality filtering requires one common evaluation dataset')
    front = frontier(data, objective)
    for r in data:
        r['pareto'] = r in front
        if baseline:
            candidates = [b for b in data if b['comparison_group'] == r['comparison_group']
                          and b['config_id'] == baseline]
            if candidates and candidates[0]['joules_per_output_token'] > 0:
                r['energy_saving_pct'] = 100 * (1 - r['joules_per_output_token'] / candidates[0]['joules_per_output_token'])
    (dest / 'summary.json').write_text(json.dumps(data, indent=2))
    if not data:
        raise ValueError('No valid energy trials; inspect coverage, exact token counts, and failures')
    fields = sorted(set().union(*(r.keys() for r in data)))
    with (dest / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(data)
    scopes = {comparison_key(r, combine_models): r.get('energy_scope') for r in rows}
    for group in sorted({r['comparison_group'] for r in data}):
        points = [r for r in data if r['comparison_group'] == group]
        fig, ax = plt.subplots()
        x_metric = OBJECTIVES[objective][0]
        color_metric = OBJECTIVES[objective][1] if len(OBJECTIVES[objective]) > 1 else None
        if color_metric:
            values = [r[color_metric] for r in points]
            norm = matplotlib.colors.Normalize(min(values), max(values))
            cmap = plt.get_cmap('viridis')
        for r in points:
            x, y = r[x_metric], r['joules_per_output_token']
            low, high = r[x_metric + '_ci_low'], r[x_metric + '_ci_high']
            yl, yh = r['joules_per_output_token_ci_low'], r['joules_per_output_token_ci_high']
            ax.errorbar(x, y, xerr=None if low is None else [[max(0, x-low)], [max(0, high-x)]],
                        yerr=None if yl is None else [[max(0, y-yl)], [max(0, yh-y)]],
                        fmt='*' if r['pareto'] else 'o',
                        color=cmap(norm(r[color_metric])) if color_metric else None)
            ax.annotate(f"{r['base_model']} {r['label']} c={r['concurrency']} {r['power_setting']}", (x, y), fontsize=7)
        if color_metric:
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                         label=AXIS_LABELS[color_metric])
        ax.set_xlabel(AXIS_LABELS[x_metric])
        ax.set_ylabel(f"J/output token, {scopes.get(group, 'unknown scope')} (including prompt processing)")
        fig.tight_layout()
        fig.savefig(dest / f'{group}-{objective}.png', dpi=180)
        fig.savefig(dest / f'{group}-{objective}.pdf')
        plt.close(fig)
    return data
