import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,time,json,copy,functools
from pathlib import Path
sys.path.insert(0,str(Path.cwd()));sys.path.insert(0,str(Path('experiments/roadmap_20261105').resolve()))
import numpy as np
import mlx.core as mx
from common import inputs
from candidate import fused
from imagewam_mlx import ImageWAM
from imagewam_mlx import action_kernel as ak
from imagewam_mlx.workspace import workspace_step
r=Path(os.environ.get('WAM_RECORD_DIR','records/split_main_20261105'));r.mkdir(exist_ok=False)
p=Path('records/roadmap_20261105/capture');d=mx.load(str(p/'projection_inputs.npz'));kw=json.loads((p/'projection_parameters.json').read_text());a=[d['x'],[d[f'part_{i}'] for i in range(3)],d['residual'],d['gate'],d['bias']]
modes={'baseline':ak.fused,'unroll':functools.partial(fused,mode='unroll'),'edges':functools.partial(fused,mode='edges'),'combined':functools.partial(fused,mode='combined')}
def make(fn):
 c=mx.compile(lambda x,w,s,q,r,g,b:fn(x,[w,s,q],r,g,b,**kw))
 return lambda:c(a[0],*a[1],*a[2:])
fns={k:make(fn) for k,fn in modes.items()};ref=fns['baseline']();mx.eval(ref);micro={}
for k,fn in fns.items():
 for _ in range(15):mx.eval(fn())
 micro[k]={'exact':bool(mx.array_equal(fn(),ref).item()),'ms':[]}
rng=np.random.default_rng(5118)
for _ in range(200):
 for k in rng.permutation(list(fns)):
  t=time.perf_counter();mx.eval(fns[k]());micro[k]['ms'].append((time.perf_counter()-t)*1000)
(r/'micro.json').write_text(json.dumps(micro,indent=2));print('micro',{k:(float(np.mean(v['ms'])),v['exact']) for k,v in micro.items()},flush=True)
b=ImageWAM('weights/bf16','assets',profile='speed',text_cache_size=0)
class Action(type(b.action)):
 def residual_linear(self,x,p,residual,gate):
  if self.kind=='w1' and p in self.packed and x.shape[0]==1 and x.shape[1]>13 and x.shape[-1]>4096:
   bias=self.w.get(p+'.bias',mx.zeros((residual.shape[-1],),dtype=x.dtype));return self.fn(x,self.packed[p],residual,gate,bias,target=self.split_target,tile=((32,64,2,2) if self.split_adaptive and x.shape[1]>16 else self.split_tile))
  return residual+gate*self.linear(x,p)
models={}
for k,fn in modes.items():
 m=copy.copy(b);act=Action.__new__(Action);act.__dict__=dict(b.action.__dict__);act.fn=fn;m.action=act
 m._compiled_workspace_step=mx.compile(lambda c,x,t,d,p,model=m:workspace_step(model,c,x,t,d,p));models[k]=m
args=inputs();perf={k:[] for k in modes};exact={k:True for k in modes}
def run(k):
 mx.synchronize();t=time.perf_counter();out,n=models[k].predict(**args,steps=10);mx.synchronize();return out,n,(time.perf_counter()-t)*1000
for k in modes:
 for _ in range(5):out,n,dt=run(k)
 if k=='baseline':gold=out.copy();gn=n.copy()
 exact[k]=bool(np.array_equal(out,gold) and np.array_equal(n,gn));print('warm',k,exact[k],flush=True)
for i in range(60):
 for k in rng.permutation(list(modes)):
  out,n,dt=run(k);perf[k].append(dt);exact[k]&=bool(np.array_equal(out,gold) and np.array_equal(n,gn))
 if i%20==0:print('round',i,flush=True)
result={k:dict(mean_ms=float(np.mean(v)),p95_ms=float(np.percentile(v,95)),exact=exact[k],samples_ms=v) for k,v in perf.items()};(r/'performance.json').write_text(json.dumps(result,indent=2));print({k:{a:b for a,b in v.items() if a!='samples_ms'} for k,v in result.items()},flush=True)
# Validate mixed text and horizons separately; no cached features.
checks=[]
for h,extra in [(1,False),(16,True),(64,False)]:
 args=inputs();args['noise']=np.resize(args['noise'],(1,h,14))
 if extra:args['prompt']+=' Please carefully follow the instruction.'*25
 gold,gn,_=run('baseline')
 for k in modes:
  out,n,_=run(k);checks.append(dict(horizon=h,long_text=extra,mode=k,exact=bool(np.array_equal(out,gold) and np.array_equal(n,gn)),max_abs=float(np.max(np.abs(out-gold)))))
  np.savez(r/f'h{h}_{k}.npz',action=out,normalized=n)
(r/'dynamic.json').write_text(json.dumps(checks,indent=2));print('done',flush=True)
