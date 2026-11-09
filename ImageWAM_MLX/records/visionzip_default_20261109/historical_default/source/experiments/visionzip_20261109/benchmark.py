"""Multi-observation paired E2E, paired module profile, cold-text and zip costs.

End-to-end and synchronized module runs are separate. Per-case outputs must
match saved validation outputs byte for byte throughout every measurement.
All samples, including load spikes, remain in the reported distributions.
"""
import argparse
import json
import subprocess
from pathlib import Path
from common import *

def pair_stats(rows):
    a=np.array([r['A'] for r in rows]);b=np.array([r['B'] for r in rows]);delta=a-b
    # Blocks of five adjacent pairs; descriptive interval conditional on these inputs.
    blocks=delta.reshape(-1,5).mean(axis=1)
    boot=np.random.default_rng(20261109).choice(blocks,(20000,len(blocks)),replace=True).mean(axis=1)
    return {'pairs':len(rows),'baseline_mean_ms':float(a.mean()),'candidate_mean_ms':float(b.mean()),
        'baseline_p50_ms':float(np.median(a)),'candidate_p50_ms':float(np.median(b)),
        'baseline_p95_ms':float(np.percentile(a,95)),'candidate_p95_ms':float(np.percentile(b,95)),
        'mean_saving_ms':float(delta.mean()),'saving_pct':float(100*(1-b.mean()/a.mean())),
        'block_bootstrap95_saving_ms':np.percentile(boot,[2.5,97.5]).tolist()}

def process_load():
    text=subprocess.check_output(['ps','-axo','pid,pcpu,pmem,comm'],text=True)
    return [line.strip() for line in text.splitlines() if 'python' in line.lower()]

def allocator_usage():
    return {'active_bytes':mx.get_active_memory(),'cache_bytes':mx.get_cache_memory(),
            'process_peak_bytes':mx.get_peak_memory(),'memory_limit_bytes':mx.get_memory_limit()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--pairs',type=int,default=50)
    p.add_argument('--presets',nargs='+',choices=tuple(PRESETS),default=['merge252','late_align32','prune252'])
    a=p.parse_args();assert a.pairs>=20 and a.pairs%10==0
    assert len(a.presets)==len(set(a.presets)), 'Duplicate presets'
    valid=json.loads((a.validation/'results.json').read_text());assert valid['status']=='complete'
    a.output.mkdir(parents=True,exist_ok=False)
    r={'status':'running','sources':snapshot(a.output),'validation_sha256':sha(a.validation/'results.json'),
       'pairs_per_observation':a.pairs,'rounds':2,'module_pairs_per_observation':25,'enabled_presets':a.presets,
       'e2e':{},'modules':{},'cold_text':{},'micro':{},'load_start':process_load(),
       'limitations':['Shared server; no competing process stopped.','Observations are from train metadata.',
          'Eight selected inputs characterize tested shapes, not all real-world scenes.',
          'Module and microbench synchronizations change scheduling; they are not additive E2E components.',
          'Microbench uses already-computed intermediate tensors only to isolate zip operations; full E2E recomputes images/state/prefix each call.']}
    path=a.output/'results.json';save(path,r)
    m=load_model();r['model']=m.study_info;original=m.prefill
    cs=cases();bylabel={c['label']:c for c in cs}
    tokens={c['label']:int(m.tokenize(c['prompt']).shape[1]) for c in cs}
    selected=[]
    def add(c):
        if c['label'] not in {x['label'] for x in selected}:selected.append(c)
    add(cs[0]);add(min(cs,key=lambda c:tokens[c['label']]));add(max(cs,key=lambda c:tokens[c['label']]))
    for name in a.presets:
        worst=max([x for x in valid['variants'][name]['rows'] if x['cohort']=='new_observations'],key=lambda x:x['normalized']['rmse'])
        add(bylabel[worst['case'].rsplit('_s',1)[0]])
    for i in [len(cs)//3,2*len(cs)//3,len(cs)-1,*range(len(cs))]:
        if len(selected)>=8:break
        add(cs[i])
    assert len(selected)==8
    r['selected']=[{'label':c['label'],'cohort':c['cohort'],'tokens':tokens[c['label']],'seed':42} for c in selected]
    prepared={}
    for c in selected:
        kw,_=sample(c,42);key=c['label']+'_s42';outputs={}
        for name in ('baseline',*a.presets):
            with np.load(a.validation/name/(key+'.npz')) as f:outputs[name]=(f['action'].copy(),f['normalized'].copy())
        prepared[c['label']]=(kw,outputs)
    save(path,r)

    def checked(fn,kw,expected,profile=False,clear=False):
        m.prefill=fn
        if clear:m.clear_text_cache()
        mx.synchronize();started=time.perf_counter_ns()
        actual,normalized=m.predict(**kw,profile=profile);mx.synchronize()
        elapsed=(time.perf_counter_ns()-started)/1e6
        assert actual.tobytes()==expected[0].tobytes() and normalized.tobytes()==expected[1].tobytes()
        return elapsed,dict(m.profile)

    for name in a.presets:
        install(m,name);candidate=m.prefill
        r['e2e'][name]={};r['modules'][name]={};r['cold_text'][name]={};r['micro'][name]={}
        # Two passes, second in reverse observation order, with balanced total AB/BA.
        orders={}
        for index,c in enumerate(selected):
            order=np.tile([0,1],a.pairs//2);np.random.default_rng(20261109+index).shuffle(order);orders[c['label']]=order
        for repeat in range(2):
            for c in (selected if repeat==0 else selected[::-1]):
                label=c['label'];kw,expected=prepared[label]
                m.prepare_text(kw['prompt'])
                for variant,fn in [('baseline',original),(name,candidate)]:
                    for _ in range(8 if repeat==0 else 2):checked(fn,kw,expected[variant])
                record=r['e2e'][name].setdefault(label,{'tokens':tokens[label],'rows':[]})
                memory_before=allocator_usage()
                for order in orders[label][repeat*a.pairs//2:(repeat+1)*a.pairs//2]:
                    row={'round':repeat,'order':'AB' if order==0 else 'BA'}
                    calls=[('A','baseline',original),('B',name,candidate)]
                    for key,variant,fn in (calls if order==0 else calls[::-1]):row[key]=checked(fn,kw,expected[variant])[0]
                    record['rows'].append(row)
                record.setdefault('allocator_by_round',{})[str(repeat)]={'before':memory_before,'after':allocator_usage()}
                record['summary']=pair_stats(record['rows']);save(path,r)
                print('E2E',name,repeat,label,record['summary']['saving_pct'],flush=True)
        r['e2e'][name]['summary']=pair_stats([row for c in selected for row in r['e2e'][name][c['label']]['rows']])
        r['e2e'][name]['round_summary']={str(i):pair_stats([row for c in selected for row in r['e2e'][name][c['label']]['rows'] if row['round']==i]) for i in range(2)}
        save(path,r)
        # Four shapes/scenes for module, cold-text and compression costs.
        for c in selected[:4]:
            label=c['label'];kw,expected=prepared[label];m.prepare_text(kw['prompt'])
            for variant,fn in [('baseline',original),(name,candidate)]:
                for _ in range(3):checked(fn,kw,expected[variant],profile=True)
            rows=[]
            order=np.arange(25)%2;np.random.default_rng(91).shuffle(order)
            for switch in order:
                row={'order':'AB' if switch==0 else 'BA'}
                calls=[('A','baseline',original),('B',name,candidate)]
                for key,variant,fn in (calls if switch==0 else calls[::-1]):
                    elapsed,stages=checked(fn,kw,expected[variant],profile=True)
                    row[key]=elapsed;row[key+'_stages']=stages
                rows.append(row)
            stage_names=rows[0]['A_stages'].keys()
            r['modules'][name][label]={'rows':rows,'summary':{stage:pair_stats([{'A':x['A_stages'][stage],'B':x['B_stages'][stage]} for x in rows]) for stage in stage_names}}
            print('MODULES',name,label,flush=True);save(path,r)
            # Cache misses are timed separately; cache clear itself is outside timer.
            rows=[]
            for index in range(10):
                row={'order':'AB' if index%2==0 else 'BA'}
                calls=[('A','baseline',original),('B',name,candidate)]
                for key,variant,fn in (calls if index%2==0 else calls[::-1]):row[key]=checked(fn,kw,expected[variant],clear=True)[0]
                rows.append(row)
            r['cold_text'][name][label]={'rows':rows,'summary':pair_stats(rows)}
            # Capture cut-boundary tensors outside measurements.
            image,state=m.preprocess(kw['images'],kw['state']);hidden=m._encode_text(m.tokenize(kw['prompt']),kw['prompt'])
            cache,pos,scores,captured=build_prefill(m,PRESETS[name],capture_cut=True)(hidden,m.vae(image),state)
            mx.eval(cache,pos,scores,captured);features,q,k=captured
            nt=hidden.shape[1]+1;cfg=PRESETS[name].resolved(nt)
            if cfg.keep==288:
                r['micro'][name][label]={'tokens':nt-1,'visual_keep':288,'costs':{},'skip_zip':True,
                    'note':'Already aligned; production skips scoring/selection/layout. No synthetic zip cost reported.'}
                del cache,captured,features,q,k
                save(path,r);continue
            score_fn=mx.compile(lambda q,k:incoming_scores(q,k,nt,cfg.score))
            selection_fn=mx.compile(lambda f,s:select_tokens(f,s,cfg))
            combined_fn=mx.compile(lambda f,q,k:select_tokens(f,incoming_scores(q,k,nt,cfg.score),cfg))
            from imagewam_mlx.ops import flux_rope,D
            ids=np.zeros((1,18,16,4),np.float32);ids[...,0]=10;ids[...,1]=np.arange(18)[:,None];ids[...,2]=np.arange(16)[None,:]
            pe=flux_rope(mx.array(ids.reshape(1,288,4)).astype(D));mx.eval(pe)
            def with_rope(f,q,k,pe):
                f,positions=select_tokens(f,incoming_scores(q,k,nt,cfg.score),cfg)
                return f,positions,tuple(mx.take(x,positions,axis=2) for x in pe)
            zipped_fn=mx.compile(with_rope)
            operations=[('attention_score',score_fn,(q,k)),('selection_and_merge',selection_fn,(features,scores)),
                        ('score_and_selection',combined_fn,(features,q,k)),('score_selection_visual_rope_gather',zipped_fn,(features,q,k,pe))]
            cost={}
            for op,fn,inputs in operations:
                for _ in range(10):mx.eval(fn(*inputs))
                mx.synchronize();values=[]
                for _ in range(100):
                    start=time.perf_counter_ns();mx.eval(fn(*inputs));mx.synchronize();values.append((time.perf_counter_ns()-start)/1e6)
                cost[op]={'mean_ms':float(np.mean(values)),'p50_ms':float(np.median(values)),'p95_ms':float(np.percentile(values,95)),'samples_ms':values}
            r['micro'][name][label]={'tokens':nt-1,'visual_keep':cfg.keep,'costs':cost,
                'note':'Visual RoPE gather only; prefix split/concat and earlier/later blocks excluded. Subgraph fusion differs from whole prefill.'}
            del cache,captured,features,q,k,pe
            save(path,r);print('MICRO',name,label,{key:round(val['mean_ms'],4) for key,val in cost.items()},flush=True)
        for category in ('modules','cold_text'):
            records=[r[category][name][c['label']] for c in selected[:4]]
            rows=[row for record in records for row in record['rows']]
            if category=='modules':
                r[category][name]['summary']={stage:pair_stats([{'A':x['A_stages'][stage],'B':x['B_stages'][stage]} for x in rows]) for stage in rows[0]['A_stages']}
            else:r[category][name]['summary']=pair_stats(rows)
        save(path,r)
    # Trace the largest errors separately; these calls are never included in timing.
    r['worst_case_traces']={};trace_dir=a.output/'traces';trace_dir.mkdir()
    for name in a.presets:
        rows=valid['variants'][name]['rows'];worst=[]
        for metric in ('rmse','max_abs'):
            row=max(rows,key=lambda x:x['normalized'][metric])
            if row['case'] not in worst:worst.append(row['case'])
        r['worst_case_traces'][name]={}
        for key in worst:
            label,seed=key.rsplit('_s',1);c=bylabel[label];kw,_=sample(c,int(seed))
            m.prefill=original;ba,bn=m.predict(**kw,trace=True);baseline_trace=dict(m.trace)
            install(m,name);ca,cn=m.predict(**kw,trace=True);candidate_trace=dict(m.trace)
            for mode,actual,normalized in [('baseline',ba,bn),(name,ca,cn)]:
                with np.load(a.validation/mode/(key+'.npz')) as ref:
                    assert actual.tobytes()==ref['action'].tobytes() and normalized.tobytes()==ref['normalized'].tobytes()
            common={field:baseline_trace[field].tobytes()==candidate_trace[field].tobytes() for field in ('image','state','text','latent','token_ids')}
            assert all(common.values())
            early={str(i):baseline_trace[f'cache_k_{i}'].tobytes()==candidate_trace[f'cache_k_{i}'].tobytes()
                   for i in (0,4,5,24) if i<PRESETS[name].cut_after}
            assert all(early.values())
            steps={str(i):metrics(candidate_trace[f'action_step_{i}'][0],baseline_trace[f'action_step_{i}'][0]) for i in range(10)}
            diff=np.abs(cn.astype(np.float64)-bn.astype(np.float64));ix=np.unravel_index(np.argmax(diff),diff.shape)
            r['worst_case_traces'][name][key]={'instruction':c['instruction'],'tokens':int(m.tokenize(c['prompt']).shape[1]),
                'unchanged_features_byte_equal':common,'early_prefix_keys_byte_equal':early,'denoise_steps':steps,
                'largest_final_error':{'action_step':int(ix[0]),'dimension':int(ix[1]),'baseline_normalized':float(bn[ix]),
                    'candidate_normalized':float(cn[ix]),'absolute_normalized_difference':float(diff[ix]),'absolute_raw_difference':float(abs(ca[ix]-ba[ix]))}}
            np.savez_compressed(trace_dir/(name+'_'+key+'.npz'),**{prefix+str(i):trace[f'action_step_{i}']
                for prefix,trace in [('baseline_',baseline_trace),('candidate_',candidate_trace)] for i in range(10)})
            save(path,r);print('TRACE',name,key,flush=True)
    install(m,'off');r.update(status='complete',all_timed_outputs_byte_equal=True,load_end=process_load())
    save(path,r);print('COMPLETE',flush=True)

if __name__=='__main__':main()
