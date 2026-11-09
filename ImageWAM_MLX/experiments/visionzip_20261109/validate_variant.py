"""Fresh-process numerical validation; run structural diagnostics afterward."""
import argparse
import json
import shutil
from pathlib import Path
from common import *
from validate import summaries


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline-validation',type=Path,required=True)
    p.add_argument('--historical-validation',type=Path,required=True)
    p.add_argument('--preset',choices=tuple(PRESETS),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    baseline=json.loads((a.baseline_validation/'results.json').read_text())
    historical=json.loads((a.historical_validation/'results.json').read_text())
    assert baseline['baseline']['count']==360 and baseline['identity_count']==72
    assert baseline['model']['allocator_setter_guard'] and baseline['model']['cache_limit_bytes'] is None
    assert historical['status']=='complete'
    shutil.copytree(a.baseline_validation/'baseline',a.output/'baseline')
    r={'status':'running','sources':snapshot(a.output),'variants':{},
       'baseline':baseline['baseline'],'identity_count':baseline['identity_count'],
       'observations':baseline['observations'],'cohorts':baseline['cohorts'],
       'baseline_and_identity_reused_from':str(a.baseline_validation),
       'baseline_run_sha256':sha(a.baseline_validation/'results.json'),
       'historical_validation_sha256':sha(a.historical_validation/'results.json'),
       'protocol':'Fresh process for one preset. Structural diagnostics run after all numerical outputs; no allocator limit override.'}
    path=a.output/'results.json';save(path,r)
    m=load_model();r['model']=m.study_info;original=m.prefill
    name=a.preset;info=install(m,name);dest=a.output/name;dest.mkdir()
    rows=[];cs=cases();diagnostic_cases={}
    for c in cs:
        for seed in c['seeds']:
            kw,target=sample(c,seed);key=c['label']+f'_s{seed}'
            actual,normalized=m.predict(**kw)
            assert actual.shape==normalized.shape==(16,14) and np.isfinite(actual).all() and np.isfinite(normalized).all()
            with np.load(a.output/'baseline'/(key+'.npz')) as ref:
                row={'case':key,'cohort':c['cohort'],'action':metrics(actual,ref['action']),
                     'normalized':metrics(normalized,ref['normalized']),'vs_demo':metrics(actual,target)}
            with np.load(a.historical_validation/name/(key+'.npz')) as ref:
                assert actual.tobytes()==ref['action'].tobytes() and normalized.tobytes()==ref['normalized'].tobytes(),('allocator parity',name,key)
            if c['cohort']=='replay':
                with np.load(OLD/'accuracy_attempt1'/(key+'.npz')) as ref:row['vs_accuracy']=metrics(actual,ref['action'])
            row['action_sha256']=hashlib.sha256(actual.tobytes()).hexdigest()
            row['normalized_sha256']=hashlib.sha256(normalized.tobytes()).hexdigest()
            np.savez_compressed(dest/(key+'.npz'),action=actual,normalized=normalized);rows.append(row)
        tokens=int(m.tokenize(c['prompt']).shape[1]);diagnostic_cases.setdefault(tokens,c)
        r['variants'][name]={'config':info,'count':len(rows),'rows':rows,'summary':summaries(rows),
            'diagnostics_by_text_length':{},'prior_project_byte_equal_count':0,
            'restricted_allocator_reference_byte_equal_count':len(rows)}
        save(path,r);print('CASE',name,c['label'],len(rows),flush=True)
    # Separating diagnostics avoids retaining their temporary allocation sizes
    # before the request sequence. This is not a performance measurement.
    for tokens,c in diagnostic_cases.items():
        kw,_=sample(c,c['seeds'][0]);image,state=m.preprocess(kw['images'],kw['state'])
        hidden=m._encode_text(m.tokenize(c['prompt']),c['prompt'])
        cache,pos,score=build_prefill(m,PRESETS[name],diagnostic=True)(hidden,m.vae(image),state)
        mx.eval(cache,pos,score);cfg=PRESETS[name].resolved(tokens+1);positions=np.array(pos)
        lengths=[k.shape[2] for k,v in cache]
        expected=[tokens+1+288 if i<cfg.cut_after else tokens+1+cfg.keep for i in range(25)]
        assert lengths==expected and len(set(positions.tolist()))==cfg.keep
        assert np.isfinite(np.array(score)).all()
        r['variants'][name]['diagnostics_by_text_length'][str(tokens)]={
            'visual_kept':cfg.keep,'camera_quotas':cfg.quotas(),'prefix_kv_lengths':lengths,
            'original_positions':positions.tolist(),'prefix_kv_bytes':sum(k.nbytes+v.nbytes for k,v in cache)}
        del cache,pos,score
        save(path,r);print('DIAGNOSTIC',name,tokens,flush=True)
    install(m,'off');assert m.prefill is original
    kw,_=sample(cs[0],cs[0]['seeds'][0]);actual,normalized=m.predict(**kw)
    with np.load(a.output/'baseline'/(cs[0]['label']+f"_s{cs[0]['seeds'][0]}.npz")) as ref:
        assert actual.tobytes()==ref['action'].tobytes() and normalized.tobytes()==ref['normalized'].tobytes()
    r.update(status='complete',off_restores_baseline=True);save(path,r);print('COMPLETE',flush=True)


if __name__=='__main__':main()
