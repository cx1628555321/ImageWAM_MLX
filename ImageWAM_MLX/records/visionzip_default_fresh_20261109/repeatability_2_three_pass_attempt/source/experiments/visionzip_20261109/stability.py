"""Offline stability analysis of audited native-w4 output metrics (no inference).

Seeds are compared within the same observation cohort. Bootstrap units are
episodes, keeping their frames and seeds together. Intervals describe this
sample; they do not establish held-out or closed-loop control accuracy.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def rmse(rows, field):
    return float(np.sqrt(sum(r[field]['sum_sq'] for r in rows) /
                         sum(r[field]['count'] for r in rows)))


def analyze(records):
    path = records / 'validation/results.json'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    audit = json.loads((path.parent / 'independent_audit.json').read_text())
    assert audit['passed'] and audit['results_sha256'] == digest
    source = json.loads(path.read_text())
    assert source['status'] == 'complete'
    observations = {c['label']: c for c in source['observations']}
    base = {r['case']: r for r in source['baseline']['rows']}
    assert len(base) == 360
    def observation(row):
        return observations[row['case'].rsplit('_s', 1)[0]]
    def seed(row):
        return int(row['case'].rsplit('_s', 1)[1])
    report = {'status': 'complete_offline_analysis', 'validation_sha256': digest,
              'new_inference_calls': 0, 'bootstrap_repetitions': 4000,
              'bootstrap_unit': 'episode, retaining all associated frames and seeds',
              'variants': {}, 'limitations': [
                  'Existing train-labelled observations; no verified training holdout.',
                  'Same-seed repeated inference determinism is not measured by different-seed comparisons.',
                  'Threshold exceedance counts are descriptive, not task acceptance criteria.',
                  'Demonstration RMSE mixes action dimensions; use per-dimension results too.',
                  'Post-hoc analysis, not a new independent validation set.',
                  'No closed-loop task success rate.']}
    seed_table, dimension_table, case_table = [], [], []
    for name, variant in source['variants'].items():
        rows = variant['rows']
        assert len(rows) == len(base) and {r['case'] for r in rows} == set(base)
        assert np.isclose(rmse(rows, 'normalized'), variant['summary']['all']['normalized']['rmse'], rtol=1e-12)
        result = {'cohorts': {}}
        for cohort in ('all', 'replay', 'new_observations'):
            selected = [r for r in rows if cohort == 'all' or r['cohort'] == cohort]
            episodes = sorted({observation(r)['episode'] for r in selected})
            grouped = [[r for r in selected if observation(r)['episode'] == ep] for ep in episodes]
            # Columns: compression squared error, candidate demo squared error,
            # baseline demo squared error, element count.
            totals = np.array([[sum(r['normalized']['sum_sq'] for r in group),
                                sum(r['vs_demo']['sum_sq'] for r in group),
                                sum(base[r['case']]['vs_demo']['sum_sq'] for r in group),
                                sum(r['normalized']['count'] for r in group)] for group in grouped])
            rng = np.random.default_rng(20261109)
            draws = rng.integers(0, len(episodes), size=(4000, len(episodes)))
            sampled = totals[draws].sum(axis=1)
            norm_ci = np.percentile(np.sqrt(sampled[:, 0]/sampled[:, 3]), [2.5, 97.5])
            demo_delta_ci = np.percentile(np.sqrt(sampled[:, 1]/sampled[:, 3]) -
                                          np.sqrt(sampled[:, 2]/sampled[:, 3]), [2.5, 97.5])
            maxima = np.array([r['normalized']['max_abs'] for r in selected])
            norms = np.array([r['normalized']['rmse'] for r in selected])
            deltas = np.array([r['vs_demo']['rmse']-base[r['case']]['vs_demo']['rmse'] for r in selected])
            groups_by_observation = {}
            for row in selected:
                groups_by_observation.setdefault(observation(row)['label'], []).append(row)
            seed_spreads = [max(r['normalized']['rmse'] for r in g)-min(r['normalized']['rmse'] for r in g)
                            for g in groups_by_observation.values()]
            item = {'calls': len(selected), 'episodes': len(episodes), 'observations': len(groups_by_observation),
                    'normalized_rmse': rmse(selected, 'normalized'),
                    'normalized_rmse_episode_bootstrap_ci95': norm_ci.tolist(),
                    'case_rmse_p50_p95_p99_max': np.percentile(norms, [50,95,99,100]).tolist(),
                    'case_max_abs_p50_p95_p99_max': np.percentile(maxima, [50,95,99,100]).tolist(),
                    'max_abs_exceeds': {str(t): int(np.count_nonzero(maxima > t)) for t in (.015625,.03125,.0625,.125)},
                    'within_observation_seed_rmse_range_p50_p95_max': np.percentile(seed_spreads,[50,95,100]).tolist(),
                    'demo_rmse_baseline': rmse([base[r['case']] for r in selected], 'vs_demo'),
                    'demo_rmse_candidate': rmse(selected, 'vs_demo'),
                    'demo_rmse_delta_episode_bootstrap_ci95': demo_delta_ci.tolist(),
                    'demo_case_rmse_worse_equal_better': [int((deltas>0).sum()),int((deltas==0).sum()),int((deltas<0).sum())],
                    'worst_demo_rmse_increase': float(deltas.max()),
                    'worst_demo_rmse_increase_case': selected[int(deltas.argmax())]['case'],
                    'by_seed': {}}
            for s in sorted({seed(r) for r in selected}):
                subset = [r for r in selected if seed(r) == s]
                stat = {'count': len(subset), 'normalized_rmse': rmse(subset, 'normalized'),
                        'normalized_max_abs': max(r['normalized']['max_abs'] for r in subset),
                        'demo_rmse_baseline': rmse([base[r['case']] for r in subset], 'vs_demo'),
                        'demo_rmse_candidate': rmse(subset, 'vs_demo')}
                item['by_seed'][str(s)] = stat
                seed_table.append(dict(preset=name, cohort=cohort, seed=s, **stat))
            result['cohorts'][cohort] = item
        for index in range(14):
            dimension_table.append(dict(preset=name, dimension_index=index,
                normalized_rmse=float(np.sqrt(sum(r['normalized']['per_dimension_sumsq'][index] for r in rows)/(len(rows)*16))),
                raw_rmse=float(np.sqrt(sum(r['action']['per_dimension_sumsq'][index] for r in rows)/(len(rows)*16))),
                raw_max_abs=max(r['action']['per_dimension_max_abs'][index] for r in rows),
                baseline_demo_rmse=float(np.sqrt(sum(base[r['case']]['vs_demo']['per_dimension_sumsq'][index] for r in rows)/(len(rows)*16))),
                candidate_demo_rmse=float(np.sqrt(sum(r['vs_demo']['per_dimension_sumsq'][index] for r in rows)/(len(rows)*16)))))
        for row in rows:
            case_table.append(dict(preset=name, case=row['case'], episode=observation(row)['episode'],
                cohort=row['cohort'], seed=seed(row), normalized_rmse=row['normalized']['rmse'],
                normalized_max_abs=row['normalized']['max_abs'],
                demo_rmse_increase=row['vs_demo']['rmse']-base[row['case']]['vs_demo']['rmse']))
        result['action_chunk_positions'] = {
            f'{lo+1}-{hi}': float(np.sqrt(sum(sum(r['normalized']['per_step_sumsq'][lo:hi]) for r in rows)/(len(rows)*(hi-lo)*14)))
            for lo,hi in ((0,4),(4,8),(8,12),(12,16))}
        report['variants'][name] = result
    out = records / 'stability'
    out.mkdir(exist_ok=True)
    for name, rows in [('seeds',seed_table), ('dimensions',dimension_table), ('cases',case_table)]:
        with (out/f'{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({name:r['cohorts']['all'] for name,r in report['variants'].items()},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('records',type=Path)
    analyze(p.parse_args().records)
