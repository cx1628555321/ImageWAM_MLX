import os
os.environ.setdefault('MLX_MAX_OPS_PER_BUFFER','256')
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import mlx.core as mx
from imagewam_mlx import ImageWAM
from imagewam_mlx.ops import D
from common import MODES,install
r=Path('records/action_condition_20261105/screen');r.mkdir(parents=True,exist_ok=False)
base=ImageWAM('weights/bf16','assets',profile='speed',text_cache_size=0,device_visual=True);models={k:install(base,k) for k in MODES};p=Path('examples/robotwin_ep0_frame0');raw=np.load(p/'inputs.npz');prompt=(p/'prompt.txt').read_text();noise=np.load(p/'initial_noise_fp32.npy')
def run(mode):
 mx.synchronize();t=time.perf_counter();a,n=models[mode].predict({k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')},raw['state'],prompt,noise,steps=10);mx.synchronize();return a,n,(time.perf_counter()-t)*1000
errors={}
for mode in MODES:
 for _ in range(5):a,n,_=run(mode)
 if mode=='baseline':ref=a.copy();norm=n.copy()
 errors[mode]=dict(exact=bool(np.array_equal(a,ref) and np.array_equal(n,norm)),max_abs=float(np.abs(a-ref).max()),rmse=float(np.sqrt(np.mean((a-ref)**2))))
 np.savez(r/(mode+'.npz'),action=a,normalized=n);print('warm',mode,errors[mode],flush=True)
rows={k:[] for k in MODES};rng=np.random.default_rng(5011)
for i in range(80):
 for k in rng.permutation(MODES):a,n,dt=run(k);rows[k].append(dt)
 if i%20==0:print(i,{k:round(float(np.mean(v)),3) for k,v in rows.items()},flush=True)
res={}
for mode,v in rows.items():
 d=np.array(rows['baseline'])-v;ci=np.percentile(np.mean(d[rng.integers(0,len(d),(5000,len(d)))],1),[2.5,97.5]);res[mode]=dict(mean_ms=float(np.mean(v)),saving_ms=float(d.mean()),ci95=ci.tolist(),samples_ms=v,**errors[mode])
(r/'results.json').write_text(json.dumps(res,indent=2));print('PERF DONE', {k:{a:b for a,b in v.items() if a!='samples_ms'} for k,v in res.items()},flush=True)
refpath=Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full');cases=json.loads((refpath/'comparison.json').read_text())['cases'];checks=[]
for c in cases:
 label=c['label'];raw=np.load(refpath/('base_inputs.npz' if label.startswith('length_') else label+'_inputs.npz'));prompt=c['prompt'];a,n,_=run('baseline');row={'label':label,'errors':{}}
 for mode in MODES[1:]:
  b,z,_=run(mode);row['errors'][mode]=dict(exact=bool(np.array_equal(a,b) and np.array_equal(n,z)),max_abs=float(np.abs(a-b).max()),rmse=float(np.sqrt(np.mean((a-b)**2))))
  np.savez(r/(label+'_'+mode+'.npz'),action=b,normalized=z)
 checks.append(row)
(r/'validation.json').write_text(json.dumps(checks,indent=2));print('VALIDATION DONE',len(checks),flush=True)
