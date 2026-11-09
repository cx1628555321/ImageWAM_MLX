"""Original-tree parity plus expanded numerical checks, separated by cohort."""
import argparse
import hashlib
import json
from pathlib import Path
from common import *

def summaries(rows):
    result={}
    for cohort in ['all','replay','new_observations']:
        group=[r for r in rows if cohort=='all' or r['cohort']==cohort]
        if not group:continue
        result[cohort]={field:aggregate([r[field] for r in group if field in r])
            for field in ('action','normalized','vs_demo','vs_accuracy') if any(field in r for r in group)}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--presets',default='merge252,late_align32,prune252,prune216')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    presets=a.presets.split(',');assert set(presets)<=set(PRESETS)
    cs=cases(expanded=not a.smoke)
    if a.smoke:cs=cs[:1];cs[0]['seeds']=[42]
    r={'status':'running','sources':snapshot(a.output),'cohorts':{'replay':'original 36 observations, 2 seeds',
       'new_observations':'24 additional episodes, 72 observations, 4 seeds; train metadata, not held out'},
       'variants':{},'observations':[{k:v for k,v in c.items() if k not in ('videos','parquet')} for c in cs]}
    path=a.output/'results.json';save(path,r)
    m=load_model();r['model']=m.study_info;original=m.prefill
    (a.output/'baseline').mkdir();baseline_rows=[]
    for c in cs:
        for seed in c['seeds']:
            kw,target=sample(c,seed);key=c['label']+f'_s{seed}'
            actual,normalized=m.predict(**kw)
            assert actual.shape==normalized.shape==(16,14) and np.isfinite(actual).all() and np.isfinite(normalized).all()
            if c['cohort']=='replay':
                with np.load(OLD/'native-w4_attempt1'/(key+'.npz')) as ref:
                    assert actual.tobytes()==ref['action'].tobytes(),('original baseline action mismatch',key)
                    assert normalized.tobytes()==ref['normalized'].tobytes(),('original baseline normalized mismatch',key)
            np.savez_compressed(a.output/'baseline'/(key+'.npz'),action=actual,normalized=normalized)
            baseline_rows.append({'case':key,'cohort':c['cohort'],'tokens':int(m.tokenize(c['prompt']).shape[1]),'vs_demo':metrics(actual,target)})
        print('BASELINE',c['label'],len(baseline_rows),flush=True)
    r['baseline']={'count':len(baseline_rows),'replay_byte_equal':True,'rows':baseline_rows,'summary':summaries(baseline_rows)};save(path,r)
    # Full rewrite at K=288 must be byte-identical on every original observation.
    m.prefill=mx.compile(build_prefill(m,Config('identity',keep=288)))
    identity_count=0
    for c in [c for c in cs if c['cohort']=='replay']:
        for seed in c['seeds']:
            kw,_=sample(c,seed);actual,n=m.predict(**kw)
            with np.load(a.output/'baseline'/(c['label']+f'_s{seed}.npz')) as ref:
                assert actual.tobytes()==ref['action'].tobytes() and n.tobytes()==ref['normalized'].tobytes()
            identity_count+=1
    m.prefill=original;r['identity_count']=identity_count;save(path,r)
    prior=Path('/Users/qklj/projects/embody_edge/experiments/imagewam_visionzip_20261109/runs/validate4')
    prior_names={'merge252':'d5_merge252','late_align32':'d15_align32','prune216':'d5_k216'}
    for name in presets:
        info=install(m,name);dest=a.output/name;dest.mkdir();rows=[];diagnostics={};port_count=0
        for c in cs:
            for seed in c['seeds']:
                kw,target=sample(c,seed);key=c['label']+f'_s{seed}'
                actual,normalized=m.predict(**kw)
                assert actual.shape==normalized.shape==(16,14) and np.isfinite(actual).all() and np.isfinite(normalized).all()
                with np.load(a.output/'baseline'/(key+'.npz')) as ref:
                    row={'case':key,'cohort':c['cohort'],'action':metrics(actual,ref['action']),
                         'normalized':metrics(normalized,ref['normalized']),'vs_demo':metrics(actual,target)}
                if c['cohort']=='replay':
                    with np.load(OLD/'accuracy_attempt1'/(key+'.npz')) as ref:row['vs_accuracy']=metrics(actual,ref['action'])
                    if name in prior_names and (prior/prior_names[name]/(key+'.npz')).exists():
                        with np.load(prior/prior_names[name]/(key+'.npz')) as ref:
                            assert actual.tobytes()==ref['action'].tobytes() and normalized.tobytes()==ref['normalized'].tobytes(),('port mismatch',name,key)
                        port_count+=1
                row['action_sha256']=hashlib.sha256(actual.tobytes()).hexdigest()
                row['normalized_sha256']=hashlib.sha256(normalized.tobytes()).hexdigest()
                np.savez_compressed(dest/(key+'.npz'),action=actual,normalized=normalized);rows.append(row)
            tokens=int(m.tokenize(c['prompt']).shape[1])
            if str(tokens) not in diagnostics:
                image,state=m.preprocess(kw['images'],kw['state']);hidden=m._encode_text(m.tokenize(kw['prompt']),kw['prompt'])
                cache,pos,score=build_prefill(m,PRESETS[name],diagnostic=True)(hidden,m.vae(image),state);mx.eval(cache,pos,score)
                cfg=PRESETS[name].resolved(tokens+1);positions=np.array(pos)
                lengths=[k.shape[2] for k,v in cache]
                expected=[tokens+1+288 if i<cfg.cut_after else tokens+1+cfg.keep for i in range(25)]
                assert lengths==expected and len(set(positions.tolist()))==cfg.keep
                assert np.isfinite(np.array(score)).all()
                diagnostics[str(tokens)]={'visual_kept':cfg.keep,'camera_quotas':cfg.quotas(),'prefix_kv_lengths':lengths,
                    'original_positions':positions.tolist(),'prefix_kv_bytes':sum(k.nbytes+v.nbytes for k,v in cache)}
            r['variants'][name]={'config':info,'count':len(rows),'rows':rows,'summary':summaries(rows),
                'diagnostics_by_text_length':diagnostics,'prior_project_byte_equal_count':port_count}
            save(path,r);print('CASE',name,c['label'],len(rows),flush=True)
        print('DONE',name,r['variants'][name]['summary']['all']['normalized']['rmse'],flush=True)
    install(m,'off');assert m.prefill is original
    # Runtime switching back to off must restore exact baseline output.
    kw,_=sample(cs[0],cs[0]['seeds'][0]);actual,normalized=m.predict(**kw)
    with np.load(a.output/'baseline'/(cs[0]['label']+f"_s{cs[0]['seeds'][0]}.npz")) as ref:
        assert actual.tobytes()==ref['action'].tobytes() and normalized.tobytes()==ref['normalized'].tobytes()
    r.update(status='complete',off_restores_baseline=True);save(path,r);print('COMPLETE',flush=True)

if __name__=='__main__':main()
