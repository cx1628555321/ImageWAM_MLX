"""Fixed-input module pairs, including identical VAE calls as a negative control.

These are isolated module measurements: prepared text/latents/KV are reused
only to isolate the corresponding module, never reported as full inference.
"""
import argparse
import json
from pathlib import Path
from common import *
from benchmark import pair_stats

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    b=json.loads((RECORDS/'benchmark/results.json').read_text());assert b['status']=='complete'
    valid=RECORDS/'validation';bylabel={c['label']:c for c in cases()}
    selected=b['selected'][:4]
    r={'status':'running','sources':snapshot(a.output),'selected':selected,'pairs_per_input':20,
       'vae_identical_control':{},'prefix':{},'action':{},'merge_vs_prune_prefix':{},
       'note':'Fixed intermediate inputs. No preprocessing/text/image cost is included in prefix or action measurements. This is not end-to-end inference.'}
    path=a.output/'results.json';save(path,r)
    m=load_model();r['model']=m.study_info;original=m.prefill
    def timed(fn,args):
        mx.synchronize();start=time.perf_counter_ns();out=fn(*args);mx.eval(out);mx.synchronize()
        return (time.perf_counter_ns()-start)/1e6,out
    def pairs(fa,aa,fb,ab,check=None):
        for fn,args in [(fa,aa),(fb,ab)]:
            for _ in range(5):mx.eval(fn(*args))
        order=np.tile([0,1],10);np.random.default_rng(109).shuffle(order);rows=[]
        for choice in order:
            row={'order':'AB' if choice==0 else 'BA'};outputs={}
            calls=[('A',fa,aa),('B',fb,ab)]
            for key,fn,args in (calls if choice==0 else calls[::-1]):row[key],outputs[key]=timed(fn,args)
            if check:check(outputs)
            rows.append(row)
        return {'rows':rows,'summary':pair_stats(rows)}
    for c in selected:
        label=c['label'];kw,_=sample(bylabel[label],42)
        image,state=m.preprocess(kw['images'],kw['state']);hidden=m._encode_text(m.tokenize(kw['prompt']),kw['prompt']);mx.eval(image,state,hidden)
        def control_check(outputs):
            assert np.array(outputs['A'].astype(mx.float32)).tobytes()==np.array(outputs['B'].astype(mx.float32)).tobytes()
        r['vae_identical_control'][label]=pairs(m.vae,(image,),m.vae,(image,),control_check)
        latent=m.vae(image);mx.eval(latent)
        base_cache=original(hidden,latent,state);mx.eval(base_cache)
        with np.load(valid/'baseline'/(label+'_s42.npz')) as f:base_expected=f['normalized'].copy()
        for name in ('merge252','late_align32','prune252'):
            install(m,name);candidate=m.prefill
            r['prefix'].setdefault(name,{})[label]=pairs(original,(hidden,latent,state),candidate,(hidden,latent,state))
            candidate_cache=candidate(hidden,latent,state);mx.eval(candidate_cache)
            with np.load(valid/name/(label+'_s42.npz')) as f:candidate_expected=f['normalized'].copy()
            def action_check(outputs):
                for key,expected in [('A',base_expected),('B',candidate_expected)]:
                    assert np.array(outputs[key][0].astype(mx.float32)).tobytes()==expected.tobytes()
            r['action'].setdefault(name,{})[label]=pairs(m.denoise,(base_cache,kw['noise'],10),m.denoise,(candidate_cache,kw['noise'],10),action_check)
            print('DIRECT',label,name,r['prefix'][name][label]['summary']['saving_pct'],flush=True)
            del candidate_cache
            save(path,r)
        merge_fn=mx.compile(build_prefill(m,PRESETS['merge252']))
        prune_fn=mx.compile(build_prefill(m,PRESETS['prune252']))
        r['merge_vs_prune_prefix'][label]=pairs(merge_fn,(hidden,latent,state),prune_fn,(hidden,latent,state))
        save(path,r)
        del base_cache
    r['vae_identical_control']['summary']=pair_stats([row for c in selected for row in r['vae_identical_control'][c['label']]['rows']])
    for field in ('prefix','action'):
        for name,records in r[field].items():records['summary']=pair_stats([row for c in selected for row in records[c['label']]['rows']])
    r['merge_vs_prune_prefix']['summary']=pair_stats([row for c in selected for row in r['merge_vs_prune_prefix'][c['label']]['rows']])
    install(m,'off');r.update(status='complete',all_action_outputs_byte_equal=True,identical_vae_outputs=True)
    save(path,r);print('COMPLETE',flush=True)

if __name__=='__main__':main()
