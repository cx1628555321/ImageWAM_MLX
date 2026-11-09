"""Independent saved-output and raw-timing audit; never loads a model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def check_pairs(rows,summary):
    av=np.array([r['A'] for r in rows]);bv=np.array([r['B'] for r in rows])
    assert len(rows)==summary['pairs'] and np.isfinite(av).all() and np.isfinite(bv).all()
    assert min(av.min(),bv.min())>=0
    for key,val in [('baseline_mean_ms',av.mean()),('candidate_mean_ms',bv.mean()),
                    ('baseline_p95_ms',np.percentile(av,95)),('candidate_p95_ms',np.percentile(bv,95)),
                    ('saving_pct',100*(1-bv.mean()/av.mean()))]:assert abs(summary[key]-val)<1e-9
    block=(av-bv).reshape(-1,5).mean(axis=1)
    boot=np.random.default_rng(20261109).choice(block,(20000,len(block)),replace=True).mean(axis=1)
    assert np.allclose(np.percentile(boot,[2.5,97.5]),summary['block_bootstrap95_saving_ms'],rtol=0,atol=1e-9)

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path)
    modes=p.add_mutually_exclusive_group()
    for mode in ('benchmark','direct','zip','repeatability'):modes.add_argument('--'+mode,action='store_true')
    p.add_argument('--validation',type=Path)
    a=p.parse_args();r=json.loads((a.run/'results.json').read_text());assert r['status']=='complete'
    for name,sha in r['sources'].items():assert digest(a.run/'source'/name)==sha
    result={'passed':True,'results_sha256':digest(a.run/'results.json'),'variants':{}}
    if a.repeatability:
        assert a.validation and digest(a.validation/'results.json')==r['validation_sha256']
        assert r['all_outputs_byte_equal'] and not r['failures']
        expected={(repeat,f'{label}_s{seed}',name) for repeat in range(r['passes'])
                  for label,seed in r['selected'] for name in ['baseline',*r['enabled_presets']]}
        assert len(r['rows'])==r['calls']==len(expected)
        assert {(x['pass'],x['case'],x['preset']) for x in r['rows']}==expected
        for row in r['rows']:
            with np.load(a.validation/row['preset']/(row['case']+'.npz')) as ref:
                for field,check in row['fields'].items():
                    assert check['finite'] and check['byte_equal'] and check['max_abs_difference']==0
                    assert hashlib.sha256(ref[field].tobytes()).hexdigest()==check['sha256']
        result.update(calls=r['calls'],all_hashes_match_frozen_outputs=True)
    elif a.zip:
        assert set(r['results'])==set(r['enabled_presets'])
        for name,records in r['results'].items():
            assert set(records)==set(r['selected'])
            for label,rec in records.items():
                assert rec['denoise_byte_equal_validation'] and rec['boundary_output_verified']
                for cost in rec['costs'].values():
                    values=np.array(cost['samples_ms'])
                    assert len(values)==r['pairs_per_shape'] and np.isfinite(values).all() and values.min()>=0
                    for key,value in [('mean_ms',values.mean()),('p50_ms',np.median(values)),('p95_ms',np.percentile(values,95))]:
                        assert abs(cost[key]-value)<1e-9
                rows=rec['paired_rows'];assert len(rows)==r['pairs_per_shape']
                assert sum(x['order']=='AB' for x in rows)==len(rows)//2
                delta=np.array([x['B']-x['A'] for x in rows])
                assert np.isfinite(delta).all() and abs(delta.mean()-rec['full_boundary_minus_identity_mean_ms'])<1e-9
                blocks=delta.reshape(-1,5).mean(axis=1)
                boot=np.random.default_rng(20261109).choice(blocks,(20000,len(blocks)),replace=True).mean(axis=1)
                assert np.allclose(np.percentile(boot,[2.5,97.5]),rec['full_boundary_minus_identity_ci95_ms'],rtol=0,atol=1e-9)
                assert rec['skip_zip']==(rec['visual_keep']==288)
            result['variants'][name]={'shapes':len(records),'pairs_per_shape':r['pairs_per_shape']}
    elif a.direct:
        controls=r['vae_identical_control'];allrows=[]
        for case in r['selected']:
            rec=controls[case['label']];check_pairs(rec['rows'],rec['summary']);allrows+=rec['rows']
        check_pairs(allrows,controls['summary']);result['vae_identical_control']=controls['summary']
        records=r['merge_vs_prune_prefix'];allrows=[]
        for case in r['selected']:
            rec=records[case['label']];check_pairs(rec['rows'],rec['summary']);allrows+=rec['rows']
        check_pairs(allrows,records['summary']);result['merge_vs_prune_prefix']=records['summary']
        for field in ('prefix','action'):
            result[field]={}
            for name,records in r[field].items():
                allrows=[]
                for case in r['selected']:
                    rec=records[case['label']];check_pairs(rec['rows'],rec['summary']);allrows+=rec['rows']
                check_pairs(allrows,records['summary']);result[field][name]=records['summary']
        assert r['all_action_outputs_byte_equal'] and r['identical_vae_outputs']
    elif a.benchmark:
        for name,records in r['e2e'].items():
            rows=[]
            for c in r['selected']:
                rec=records[c['label']];check_pairs(rec['rows'],rec['summary']);rows+=rec['rows']
                assert sum(x['order']=='AB' for x in rec['rows'])==len(rec['rows'])//2
            check_pairs(rows,records['summary'])
            for repeat in range(2):check_pairs([x for x in rows if x['round']==repeat],records['round_summary'][str(repeat)])
            result['variants'][name]=records['summary']
        for category in ('modules','cold_text'):
            for name,records in r[category].items():
                allrows=[]
                for label,rec in records.items():
                    if label=='summary':continue
                    allrows+=rec['rows']
                    if category=='cold_text':check_pairs(rec['rows'],rec['summary'])
                    else:
                        for stage,summary in rec['summary'].items():
                            check_pairs([{'A':x['A_stages'][stage],'B':x['B_stages'][stage]} for x in rec['rows']],summary)
                if category=='cold_text':check_pairs(allrows,records['summary'])
                else:
                    for stage,summary in records['summary'].items():
                        check_pairs([{'A':x['A_stages'][stage],'B':x['B_stages'][stage]} for x in allrows],summary)
        for name,records in r['micro'].items():
            for rec in records.values():
                for cost in rec['costs'].values():
                    assert len(cost['samples_ms'])==100 and abs(np.mean(cost['samples_ms'])-cost['mean_ms'])<1e-9
        assert r['all_timed_outputs_byte_equal']
        for name,records in r.get('worst_case_traces',{}).items():
            for key,rec in records.items():
                with np.load(a.run/'traces'/(name+'_'+key+'.npz')) as arrays:
                    for i in range(10):
                        d=arrays[f'candidate_{i}'].astype(np.float64)-arrays[f'baseline_{i}'].astype(np.float64)
                        assert abs(np.sqrt(np.mean(d*d))-rec['denoise_steps'][str(i)]['rmse'])<1e-12
                assert all(rec['unchanged_features_byte_equal'].values()) and all(rec['early_prefix_keys_byte_equal'].values())
    else:
        noises=set();cohort_episodes={}
        for obs in r['observations']:
            base=Path(obs['base']);assert digest(base/'inputs'/(obs['label']+'.npz'))==obs['sha256']
            cohort_episodes.setdefault(obs['cohort'],set()).add(obs['episode'])
            for seed in obs['seeds']:
                if (str(base),seed) in noises:continue
                actual=np.load(base/'inputs'/f'noise_{seed}.npy')
                expected=np.random.default_rng(seed).standard_normal((1,16,14)).astype(np.float32)
                assert actual.tobytes()==expected.tobytes();noises.add((str(base),seed))
        assert not cohort_episodes.get('replay',set()) & cohort_episodes.get('new_observations',set())
        for name,v in r['variants'].items():
            all_diffs={'action':[],'normalized':[]};cohorts={}
            for row in v['rows']:
                with np.load(a.run/name/(row['case']+'.npz')) as actual,np.load(a.run/'baseline'/(row['case']+'.npz')) as base:
                    for key in all_diffs:
                        x=actual[key];y=base[key];assert x.shape==y.shape==(16,14) and x.dtype==np.float32 and np.isfinite(x).all()
                        assert hashlib.sha256(x.tobytes()).hexdigest()==row[key+'_sha256']
                        diff=x.astype(np.float64)-y.astype(np.float64)
                        assert abs(np.sqrt(np.mean(diff*diff))-row[key]['rmse'])<1e-12
                        all_diffs[key].append(diff)
                        cohorts.setdefault(row['cohort'],{}).setdefault(key,[]).append(diff)
            for cohort,diffs in {'all':all_diffs,**cohorts}.items():
                for key,values in diffs.items():
                    arr=np.stack(values);summary=v['summary'][cohort][key]
                    assert abs(np.sqrt(np.mean(arr*arr))-summary['rmse'])<1e-12
                    assert abs(np.max(np.abs(arr))-summary['worst_max_abs'])<1e-12
                    assert np.allclose(np.sqrt(np.mean(arr*arr,axis=(0,1))),summary['per_dimension_rmse'],rtol=0,atol=1e-12)
                    assert np.allclose(np.sqrt(np.mean(arr*arr,axis=(0,2))),summary['per_step_rmse'],rtol=0,atol=1e-12)
            result['variants'][name]={'count':len(v['rows']),'cohorts':{k:x['normalized']['rmse'] for k,x in v['summary'].items()},
                                      'prior_project_byte_equal_count':v['prior_project_byte_equal_count']}
        assert r['baseline']['replay_byte_equal'] and r['off_restores_baseline']
        assert r['identity_count']==sum(x['cohort']=='replay' for x in r['baseline']['rows'])
    (a.run/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
