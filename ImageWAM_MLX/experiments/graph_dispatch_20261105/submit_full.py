import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,copy,time,json
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
from imagewam_mlx.model import ImageWAM as Base
from imagewam_mlx import split_quant_visual as sq
pv=sq.pv
root=Path('records/graph_dispatch_20261105/submit');root.mkdir(exist_ok=False)
import types
from submit import denoise
base=ImageWAM('weights/bf16','assets',profile='speed');original=base.denoise
models={'baseline':base,'xonly':base,'every2':base,'every5':base}
intervals={'xonly':1,'every2':2,'every5':5};kernels={k:None for k in intervals}
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
def run(name):
 if name=='baseline':base.denoise=original
 else:base.submit_interval=intervals[name];base.denoise=types.MethodType(denoise,base)
 mx.synchronize();t=time.perf_counter();a,n=models[name].predict({k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')},raw['state'],prompt,noise);mx.synchronize();return a,n,(time.perf_counter()-t)*1000
outs={}
for name in models:
 for _ in range(5):a,n,dt=run(name)
 outs[name]=(a,n);np.savez(root/(name+'.npz'),action=a,normalized=n)
 assert np.array_equal(a,outs['baseline'][0]) and np.array_equal(n,outs['baseline'][1]),name
 print('warmup exact',name,flush=True)
rows={k:[] for k in models};rng=np.random.default_rng(94)
for i in range(120):
 for name in rng.permutation(list(models)):
  a,n,dt=run(name);rows[name].append(dt)
 if i%20==0:print(i,{k:float(np.mean(v)) for k,v in rows.items()},flush=True)
res={}
for name,v in rows.items():
 d=np.array(rows['baseline'])-v;ci=np.percentile(np.mean(d[rng.integers(0,len(d),(10000,len(d)))],1),[2.5,97.5]);res[name]=dict(mean_ms=float(np.mean(v)),saving_ms=float(d.mean()),ci95=ci.tolist(),samples_ms=v)
(root/'results.json').write_text(json.dumps(res,indent=2));print({k:{a:b for a,b in v.items() if a!='samples_ms'} for k,v in res.items()},flush=True)
cases=json.loads(Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full/comparison.json').read_text())['cases'];ref=Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full');checks=[]
for case in cases:
 label=case['label'];raw=np.load(ref/('base_inputs.npz' if label.startswith('length_') else label+'_inputs.npz'));prompt=case['prompt'];a,n,_=run('baseline')
 for name in kernels:
  b,z,_=run(name);assert np.array_equal(a,b) and np.array_equal(n,z),(label,name);np.savez(root/(label+'_'+name+'.npz'),action=b,normalized=z)
 checks.append(label)
(root/'validation.json').write_text(json.dumps(dict(exact_cases=checks,all_exact=True),indent=2));print('ALL EXACT',len(checks),flush=True)
