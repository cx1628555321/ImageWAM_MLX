import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,time,json,copy
from pathlib import Path
sys.path.insert(0,str(Path.cwd()));sys.path.insert(0,str(Path('experiments/roadmap_20261105').resolve()))
from common import inputs
import numpy as np
import mlx.core as mx
from imagewam_mlx import ImageWAM
from imagewam_mlx.model import ImageWAM as Base
from imagewam_mlx.workspace import workspace_step
mode=sys.argv[1];r=Path('records/audit4_20261105')/mode;r.mkdir(exist_ok=False)
m=ImageWAM('weights/bf16','assets',profile='speed',text_cache_size=0);counts={'prefill':0,'step':0}
def prefill(h,l,s):counts['prefill']+=1;return Base.prefill(m,h,l,s)
def step(c,x,t,d,p):counts['step']+=1;return workspace_step(m,c,x,t,d,p)
m.prefill=mx.compile(prefill,shapeless=mode=='shapeless_prefill');m._compiled_workspace_step=mx.compile(step,shapeless=mode=='shapeless_step')
source=Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full');cases=json.loads((source/'comparison.json').read_text())['cases'];bylen={int(m.tokenize(c['prompt']).shape[1]):c['prompt'] for c in cases};base=inputs();prompts=[bylen[31],base['prompt'],bylen[79],base['prompt']+' Please carefully follow the instruction.'*25]
samples=[dict(base,prompt=p) for p in prompts];prewarm=[]
if mode=='prewarmed':prewarm=m.warmup(samples[:3])
rows=[];errors=[]
# Repeated mixed lengths, including an unseen longer prompt. Noise remains explicit and unchanged.
sequence=[0,1,2,3]*3+np.random.default_rng(5117).integers(0,4,20).tolist()
for i,idx in enumerate(sequence):
 args=samples[idx];old=dict(counts);mx.synchronize();t=time.perf_counter()
 try:
  a,n=m.predict(**args,steps=10);mx.synchronize();ms=(time.perf_counter()-t)*1000
  np.savez(r/f'{i:03d}.npz',action=a,normalized=n)
  row=dict(index=i,input=idx,tokens=int(m.tokenize(args['prompt']).shape[1]),ms=ms,trace_delta={k:counts[k]-old[k] for k in counts})
  if mode!='baseline':
   ref=np.load(r.parent/'baseline'/f'{i:03d}.npz');row['exact']=bool(np.array_equal(a,ref['action']) and np.array_equal(n,ref['normalized']));row['max_abs']=float(np.abs(a-ref['action']).max())
  rows.append(row)
 except Exception as e:errors.append(dict(index=i,input=idx,error=repr(e),counts=dict(counts)));break
(r/'results.json').write_text(json.dumps(dict(mode=mode,rows=rows,errors=errors,counts=counts,prewarm=prewarm,measurement='diagnostic tracing wrappers shared by all modes; full requests, not production speed claim'),indent=2));print(mode,'rows',len(rows),'errors',errors,flush=True)
