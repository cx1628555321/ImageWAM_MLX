import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,json,copy,functools
from pathlib import Path
sys.path.insert(0,str(Path.cwd()));sys.path.insert(0,str(Path('experiments/roadmap_20261105').resolve()))
import numpy as np
import mlx.core as mx
from common import inputs
from candidate import fused
from imagewam_mlx import ImageWAM
from imagewam_mlx import action_kernel as ak
from imagewam_mlx.workspace import workspace_step
modes={'baseline':ak.fused,'unroll':functools.partial(fused,mode='unroll'),'edges':functools.partial(fused,mode='edges'),'combined':functools.partial(fused,mode='combined')}
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

r=Path(os.environ.get('WAM_RECORD_DIR','records/split_main_20261105'))/'validation';r.mkdir(exist_ok=False)
source=Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full');cases=json.loads((source/'comparison.json').read_text())['cases'];rows=[]
for i,c in enumerate(cases):
 label=c['label'];raw=np.load(source/('base_inputs.npz' if label.startswith('length_') else label+'_inputs.npz'));args=dict(images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')},state=raw['state'],prompt=c['prompt'],noise=inputs()['noise'])
 a,n=models['baseline'].predict(**args,steps=10);np.savez(r/f'{i:03d}_baseline.npz',action=a,normalized=n)
 for k in list(modes)[1:]:
  out,z=models[k].predict(**args,steps=10);np.savez(r/f'{i:03d}_{k}.npz',action=out,normalized=z);rows.append(dict(label=label,mode=k,exact=bool(np.array_equal(a,out) and np.array_equal(n,z)),max_abs=float(np.max(np.abs(a-out)))))
 if i%10==0:print('case',i,flush=True)
(r/'results.json').write_text(json.dumps(rows,indent=2));print('validation',len(cases),all(x['exact'] for x in rows),flush=True)
