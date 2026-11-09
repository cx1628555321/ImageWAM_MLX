"""Repeated explicit native-w4/VisionZip modes against frozen output bytes.

Run in two fresh processes for process-level reproducibility. This checks
determinism, not task accuracy or the acceptability of compression errors.
"""
import argparse
import json
from pathlib import Path
from common import *


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--passes',type=int,default=3)
    p.add_argument('--presets',nargs='+',choices=tuple(PRESETS),default=list(PRESETS))
    a=p.parse_args();assert a.passes>=2 and len(set(a.presets))==len(a.presets)
    valid=json.loads((a.validation/'results.json').read_text())
    bench=json.loads((a.benchmark/'results.json').read_text())
    assert valid['status']==bench['status']=='complete'
    a.output.mkdir(parents=True,exist_ok=False)
    bylabel={c['label']:c for c in cases()};selected=[]
    def add(label,seed):
        if (label,seed) not in selected:selected.append((label,seed))
    for c in bench['selected']:
        for seed in bylabel[c['label']]['seeds']:add(c['label'],seed)
    for name in a.presets:
        for metric in ('rmse','max_abs'):
            row=max(valid['variants'][name]['rows'],key=lambda r:r['normalized'][metric])
            label,seed=row['case'].rsplit('_s',1);add(label,int(seed))
    r={'status':'running','sources':snapshot(a.output),'selected':selected,'passes':a.passes,
       'validation_sha256':sha(a.validation/'results.json'),'enabled_presets':a.presets,
       'rows':[],'failures':[],
       'call_order':'Shuffle observation groups each pass; shuffle seeds and presets within each group. This preserves realistic runs of the same instruction while changing call order.',
       'note':'Deterministic repeatability under changed call order; not a claim of unchanged task accuracy.'}
    path=a.output/'results.json';save(path,r)
    m=load_model();r['model']=m.study_info;functions={'baseline':m.prefill}
    for name in a.presets:
        info=install(m,name);assert info['enabled'] and info['preset']==name
        functions[name]=m.prefill
    jobs=[(label,seed,name) for label,seed in selected for name in functions]
    for repeat in range(a.passes):
        rng=np.random.default_rng(109+repeat)
        labels=list(dict.fromkeys(label for label,seed in selected));rng.shuffle(labels)
        order=[]
        for label in labels:
            group=[index for index,job in enumerate(jobs) if job[0]==label]
            rng.shuffle(group);order.extend(group)
        for index in order:
            label,seed,name=jobs[int(index)];key=f'{label}_s{seed}';kw,_=sample(bylabel[label],seed)
            m.prefill=functions[name];actual,normalized=m.predict(**kw)
            row={'pass':repeat,'case':key,'preset':name,'fields':{}}
            with np.load(a.validation/name/(key+'.npz')) as ref:
                for field,array in [('action',actual),('normalized',normalized)]:
                    expected=ref[field];finite=bool(np.isfinite(array).all());same_shape=array.shape==expected.shape
                    equal=finite and same_shape and array.dtype==expected.dtype and array.tobytes()==expected.tobytes()
                    row['fields'][field]={'byte_equal':equal,'finite':finite,
                        'sha256':hashlib.sha256(array.tobytes()).hexdigest(),
                        'max_abs_difference':float(np.max(np.abs(array.astype(np.float64)-expected))) if finite and same_shape else None}
                    if not equal:r['failures'].append({'pass':repeat,'case':key,'preset':name,'field':field})
            r['rows'].append(row);save(path,r)
        print('PASS_FINISHED',repeat,len(r['rows']),len(r['failures']),flush=True)
    r.update(status='complete' if not r['failures'] else 'failed',calls=len(r['rows']),all_outputs_byte_equal=not r['failures'])
    save(path,r)
    if r['failures']:raise SystemExit(1)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
