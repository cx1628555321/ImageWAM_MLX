import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from common import *
r=ROOT/'screen';r.mkdir(exist_ok=False)
b=ImageWAM('weights/bf16','assets',profile='speed',text_cache_size=0);modes=['baseline','visual_selected','split128','split256','split1024'];models={k:install(b,k) for k in modes};args=inputs()
def run(k):
 mx.synchronize();t=time.perf_counter();a,n=models[k].predict(**args,steps=10);mx.synchronize();return a,n,(time.perf_counter()-t)*1000
errors={}
for k in modes:
 for _ in range(5):a,n,dt=run(k)
 if k=='baseline':ref=a.copy();norm=n.copy()
 errors[k]=dict(exact=bool(np.array_equal(a,ref) and np.array_equal(n,norm)),max_abs=float(np.abs(a-ref).max()))
 print('warm',k,errors[k],flush=True)
rows={k:[] for k in modes};rng=np.random.default_rng(5115)
for i in range(60):
 for k in rng.permutation(modes):a,n,dt=run(k);rows[k].append(dt)
 if i%20==0:print('round',i,flush=True)
res={k:dict(mean_ms=float(np.mean(v)),p95_ms=float(np.percentile(v,95)),samples_ms=v,**errors[k]) for k,v in rows.items()}
(r/'performance.json').write_text(json.dumps(res,indent=2));print('PERF', {k:{a:b for a,b in v.items() if a!='samples_ms'} for k,v in res.items()},flush=True)
# Correctness across the existing 51 real inputs plus dynamic horizon/long text.
refpath=Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full');cases=json.loads((refpath/'comparison.json').read_text())['cases'];validation=[]
for c in cases:
 label=c['label'];raw=np.load(refpath/('base_inputs.npz' if label.startswith('length_') else label+'_inputs.npz'));args=dict(images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')},state=raw['state'],prompt=c['prompt'],noise=inputs()['noise']);a,n,_=run('baseline');errs={}
 for k in modes[1:]:
  out,z,_=run(k);errs[k]=dict(exact=bool(np.array_equal(a,out) and np.array_equal(n,z)),max_abs=float(np.abs(a-out).max()),rmse=float(np.sqrt(np.mean((a-out)**2))))
  np.savez(r/(label+'_'+k+'.npz'),action=out,normalized=z)
 validation.append(dict(label=label,tokens=int(b.tokenize(c['prompt']).shape[1]),errors=errs))
(r/'validation.json').write_text(json.dumps(validation,indent=2));print('VALIDATION',len(validation),flush=True)
# Trace runs are deliberately separated from performance measurements.
trace=[]
for h,longtext,alter in [(1,False,False),(16,True,False),(64,False,False),(16,False,True),(16,False,False)]:
 args=inputs();args['noise']=np.resize(args['noise'],(1,h,14))
 if longtext:args['prompt']+=' Please carefully follow the instruction.'*25
 if alter:args['images']['cam_high']=args['images']['cam_high'][::2,::2].copy();args['state']=args['state'].copy();args['state'][0]+=.1
 a,n=b.predict(**args,steps=10,trace=True);gold={k:v.copy() for k,v in b.trace.items()};checks={}
 for k in modes[1:]:
  out,z=models[k].predict(**args,steps=10,trace=True);checks[k]=dict(action_exact=bool(np.array_equal(a,out) and np.array_equal(n,z)),trace_exact={name:bool(np.array_equal(v,models[k].trace[name])) for name,v in gold.items()})
  np.savez(r/f'dynamic_{len(trace)}_{k}.npz',action=out,normalized=z)
 trace.append(dict(horizon=h,tokens=int(b.tokenize(args['prompt']).shape[1]),altered=alter,checks=checks))
(r/'dynamic.json').write_text(json.dumps(trace,indent=2));print('DYNAMIC DONE',flush=True)
