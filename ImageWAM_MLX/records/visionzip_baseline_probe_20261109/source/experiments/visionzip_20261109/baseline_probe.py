"""Same native-w4, same input: isolate allocator cache and timing distributions."""
import argparse
import json
import subprocess
from pathlib import Path
from common import *


def stats(values):
    a=np.array(values)
    return {'count':len(a),'mean_ms':float(a.mean()),'p50_ms':float(np.median(a)),
            'p95_ms':float(np.percentile(a,95)),'min_ms':float(a.min()),'max_ms':float(a.max()),
            'samples_ms':values}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    r={'status':'running','sources':snapshot(a.output),'windows':[],
       'note':'VisionZip disabled throughout. Same process/model/input; cache-limit ABBA then second ABBA. Other services are not stopped.'}
    path=a.output/'results.json'
    def load():return subprocess.check_output(['ps','-axo','pid,pcpu,pmem,comm'],text=True)
    r['load_start']=load()
    old=mx.set_cache_limit(64*1024**2);r['mlx_default_cache_limit']=old;mx.set_cache_limit(old)
    old=mx.set_memory_limit(13*1024**3);r['mlx_default_memory_limit']=old;mx.set_memory_limit(old)
    example=ROOT/'examples/robotwin_ep0_frame0'
    with np.load(example/'inputs.npz') as f:
        kw={'images':{k:f[k].copy() for k in CAMS},'state':f['state'].copy(),
            'prompt':(example/'prompt.txt').read_text(),'noise':np.load(example/'initial_noise_fp32.npy'),'steps':10}
    c=cases()[0];study,_=sample(c,42)
    r['standard_vs_study']={'prompt_equal':kw['prompt']==study['prompt'],'prompt_repr':repr(kw['prompt']),
        'study_prompt_repr':repr(study['prompt']),
        'arrays':{key:bool(np.array_equal(kw['images'][key],study['images'][key])) for key in CAMS},
        'state_equal':bool(np.array_equal(kw['state'],study['state'])),
        'noise_equal':bool(np.array_equal(kw['noise'],study['noise']))}
    historical=ROOT/'records/background_20261106_xcode'
    r['historical']={}
    for name in ('route4796_native_w4_attempt1','confirm_route4796_native_w4_1_attempt1','confirm_route4796_native_w4_2_attempt1'):
        source=historical/name/'results.json';d=json.loads(source.read_text())
        r['historical'][name]={'sha256':sha(source),'steady':d['steady'],'modules':d['modules'],
            'tokens':d['text_prepare']['tokens'],'peak_bytes':d.get('peak_bytes')}
    save(path,r)
    m=load_model();r['model']=m.study_info
    r['actual_configuration']={key:getattr(m,key,None) for key in ['inference_profile','prefix_precision','kv_workspace',
        'optimization','compact','fused_vae','overlap','vae_asymmetric','device_visual','visual_group_rows',
        'visual_shape_dispatch','tail_visual','staged_visual','action_epilogue','text_cache_size']}
    r['tokens']=int(m.tokenize(kw['prompt']).shape[1]);r['action_kind']=m.action.kind
    m.prepare_text(kw['prompt'])
    with np.load(historical/'confirm_route4796_native_w4_1_attempt1/standard.npz') as ref:
        historical_output={k:ref[k].copy() for k in ('action','normalized')}
    expected=None
    def check(out):
        nonlocal expected
        if expected is None:expected=tuple(x.copy() for x in out)
        assert all(x.tobytes()==y.tobytes() for x,y in zip(out,expected)), 'Numerical output changed'
    def timed(profile=False):
        mx.synchronize();start=time.perf_counter_ns();out=m.predict(**kw,profile=profile);mx.synchronize()
        elapsed=(time.perf_counter_ns()-start)/1e6;check(out);return elapsed,dict(m.profile)
    # 256 MiB is the prior VisionZip study setting, 1024 MiB remains bounded.
    for index,cache_mib in enumerate([256,1024,1024,256,1024,256,256,1024]):
        mx.synchronize();mx.clear_cache();mx.set_cache_limit(cache_mib*1024**2)
        for _ in range(10):timed()
        values=[timed()[0] for _ in range(50)]
        profiles=[timed(True)[1] for _ in range(25)]
        rec={'index':index,'cache_limit_mib':cache_mib,'e2e':stats(values),
             'modules':{k:stats([v[k] for v in profiles]) for k in profiles[0]},
             'memory':{'active':mx.get_active_memory(),'cached':mx.get_cache_memory(),'peak':mx.get_peak_memory()},'load':load()}
        r['windows'].append(rec);save(path,r)
        print('WINDOW',index,cache_mib,rec['e2e']['mean_ms'],rec['modules']['image_encoder']['mean_ms'],flush=True)
    r['historical_output_comparison']={key:{'byte_equal':actual.tobytes()==historical_output[key].tobytes(),
        'max_abs':float(np.max(np.abs(actual.astype(np.float64)-historical_output[key])))}
        for key,actual in zip(('action','normalized'),expected)}
    r['summary']={}
    for cache in (256,1024):
        windows=[w for w in r['windows'] if w['cache_limit_mib']==cache]
        r['summary'][str(cache)]={'e2e':stats([x for w in windows for x in w['e2e']['samples_ms']]),
            'modules':{k:stats([x for w in windows for x in w['modules'][k]['samples_ms']]) for k in windows[0]['modules']}}
    r.update(status='complete',all_current_outputs_byte_equal=True,load_end=load())
    save(path,r);print('COMPLETE',flush=True)


if __name__=='__main__':main()
