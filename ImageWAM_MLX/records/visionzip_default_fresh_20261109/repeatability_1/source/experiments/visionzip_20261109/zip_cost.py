"""Native-w4 compression-boundary microbenchmark with complete layout handling.

Uses actual cut-boundary tensors. Costs are synchronized compiled subgraphs,
not an additive decomposition of the fully compiled end-to-end request.
"""
import argparse
import json
from pathlib import Path
from common import *
from benchmark import pair_stats, process_load


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pairs',type=int,default=100)
    p.add_argument('--presets',nargs='+',choices=tuple(PRESETS),default=list(PRESETS))
    a=p.parse_args()
    assert a.pairs>=20 and a.pairs%10==0 and len(set(a.presets))==len(a.presets)
    valid=json.loads((a.validation/'results.json').read_text())
    bench=json.loads((a.benchmark/'results.json').read_text())
    assert valid['status']==bench['status']=='complete'
    a.output.mkdir(parents=True,exist_ok=False)
    r={'status':'running','sources':snapshot(a.output),'enabled_presets':a.presets,
       'validation_sha256':sha(a.validation/'results.json'),
       'benchmark_sha256':sha(a.benchmark/'results.json'),'pairs_per_shape':a.pairs,
       'load_start':process_load(),'results':{},
       'limitations':['Actual cached boundary tensors isolate zip cost only; this is not E2E.',
          'Full boundary includes scoring, selection/merge, positional indexing and required concatenation.',
          'Fused full boundary and separately compiled components have different scheduling; do not sum them.',
          'Identity dispatch is a timing floor, not an uncompressed prefix benchmark.',
          'Shared server; retain all outliers. No closed-loop accuracy claim.']}
    path=a.output/'results.json';save(path,r)
    m=load_model();r['model']=m.study_info
    bylabel={c['label']:c for c in cases()}
    chosen=[bylabel[c['label']] for c in bench['selected'][:4]]
    # Include an already-aligned input when present, to verify the real skip path.
    for c in bylabel.values():
        if int(m.tokenize(c['prompt']).shape[1])==63 and c['label'] not in {x['label'] for x in chosen}:
            chosen.append(c);break
    r['selected']=[c['label'] for c in chosen]
    def run(fn,args):
        mx.synchronize();start=time.perf_counter_ns();out=fn(*args);mx.eval(out);mx.synchronize()
        return (time.perf_counter_ns()-start)/1e6,out
    for c in chosen:
        label=c['label'];kw,_=sample(c,42)
        image,state=m.preprocess(kw['images'],kw['state'])
        hidden=m._encode_text(m.tokenize(kw['prompt']),kw['prompt']);latent=m.vae(image)
        mx.eval(hidden,latent,state);nt=hidden.shape[1]+1
        for name in a.presets:
            cfg=PRESETS[name].resolved(nt)
            cache,positions,scores,captured=build_prefill(m,PRESETS[name],capture_cut=True,capture_layout=True)(hidden,latent,state)
            mx.eval(cache,positions,scores,captured)
            with np.load(a.validation/name/(label+'_s42.npz')) as ref:
                actual=m.denoise(cache,kw['noise'],10)[0];mx.eval(actual)
                assert np.array(actual.astype(mx.float32)).tobytes()==ref['normalized'].tobytes()
            early=cfg.cut_after<=5
            if early:
                features,q,k,txt,ti,ii=captured
                pe=tuple(mx.concatenate([t,i],axis=2) for t,i in zip(ti,ii));mx.eval(pe)
                context=(txt,*ti,*ii,*pe)
                def layout(sparse,pos,text,tc,ts,ic,iss,pc,ps):
                    image_pe=tuple(mx.take(t,pos,axis=2) for t in (ic,iss))
                    return mx.concatenate([text,sparse],axis=1),tuple(mx.concatenate([t,i],axis=2) for t,i in zip((tc,ts),image_pe))
                def identity(f,q,k,text,tc,ts,ic,iss,pc,ps):
                    return mx.concatenate([text,f],axis=1),(pc,ps)
                args=(features,q,k,*context)
            else:
                x,q,k,pe=captured;features=x[:,nt:];mx.eval(features)
                context=(x[:,:nt],*pe);mx.eval(context)
                def layout(sparse,pos,text,pc,ps):
                    ix=mx.concatenate([mx.arange(nt,dtype=mx.int32),nt+pos])
                    return mx.concatenate([text,sparse],axis=1),tuple(mx.take(t,ix,axis=2) for t in (pc,ps))
                def identity(x,q,k,text,pc,ps):return x,(pc,ps)
                args=(x,q,k,*context)
            def boundary(f,q,k,*context):
                if cfg.keep==288:return identity(f,q,k,*context)
                features=f if early else f[:,nt:]
                sparse,pos=select_tokens(features,incoming_scores(q,k,nt,cfg.score),cfg)
                return layout(sparse,pos,*context)
            full=mx.compile(boundary);noop=mx.compile(identity)
            operations={'full_boundary':(full,args),'identity_dispatch':(noop,args)}
            if cfg.keep<288:
                sparse,pos=select_tokens(features,scores,cfg);mx.eval(sparse,pos)
                assert np.array(pos).tobytes()==np.array(positions).tobytes()
                operations.update(
                    attention_score=(mx.compile(lambda q,k:incoming_scores(q,k,nt,cfg.score)),(q,k)),
                    selection_and_merge=(mx.compile(lambda f,s:select_tokens(f,s,cfg)),(features,scores)),
                    layout_and_position=(mx.compile(layout),(sparse,pos,*context)))
            expected=layout(sparse,pos,*context) if cfg.keep<288 else identity(*args)
            measured=full(*args);mx.eval(expected,measured)
            def assert_equal_tree(left,right):
                if isinstance(left,(tuple,list)):
                    assert len(left)==len(right)
                    for first,second in zip(left,right):assert_equal_tree(first,second)
                else:
                    assert left.shape==right.shape
                    assert np.array(left.astype(mx.float32)).tobytes()==np.array(right.astype(mx.float32)).tobytes()
            assert_equal_tree(expected,measured)
            costs={}
            for op,(fn,inputs) in operations.items():
                for _ in range(10):mx.eval(fn(*inputs))
                values=[run(fn,inputs)[0] for _ in range(a.pairs)]
                costs[op]={'mean_ms':float(np.mean(values)),'p50_ms':float(np.median(values)),
                           'p95_ms':float(np.percentile(values,95)),'samples_ms':values}
            rows=[];order=np.tile([0,1],a.pairs//2);np.random.default_rng(109).shuffle(order)
            for direction in order:
                row={'order':'AB' if direction==0 else 'BA'}
                calls=[('A',noop),('B',full)]
                for key,fn in (calls if direction==0 else calls[::-1]):row[key]=run(fn,args)[0]
                rows.append(row)
            pair=pair_stats(rows)
            r['results'].setdefault(name,{})[label]={'text_tokens':nt-1,'visual_keep':cfg.keep,
                'skip_zip':cfg.keep==288,'cut_after':cfg.cut_after,'costs':costs,
                'full_boundary_minus_identity_mean_ms':-pair['mean_saving_ms'],
                'full_boundary_minus_identity_ci95_ms':[-x for x in pair['block_bootstrap95_saving_ms'][::-1]],
                'paired_rows':rows,'denoise_byte_equal_validation':True,'boundary_output_verified':True}
            save(path,r);print('ZIP',name,label,cfg.keep,{k:round(v['mean_ms'],4) for k,v in costs.items()},flush=True)
            del operations,cache,captured,actual
    r.update(status='complete',load_end=process_load());save(path,r);print('COMPLETE',flush=True)


if __name__=='__main__':main()
