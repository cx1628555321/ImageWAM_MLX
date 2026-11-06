import os
os.environ.setdefault('MLX_MAX_OPS_PER_BUFFER','256')
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import mlx.core as mx
from imagewam_mlx import ImageWAM
from common import install,predict,MODES
r=Path('records/remaining_20261105/screen');r.mkdir(parents=True,exist_ok=False)
base=ImageWAM('weights/bf16','assets',profile='speed');models={mode:install(base,mode) for mode in MODES}
p=Path('examples/robotwin_ep0_frame0');raw=np.load(p/'inputs.npz');noise=np.load(p/'initial_noise_fp32.npy')
def args(raw,prompt):return dict(images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')},state=raw['state'],prompt=prompt,noise=noise)
a=args(raw,(p/'prompt.txt').read_text());rows={k:[] for k in MODES};errors={}
def run(mode):
 m,s=models[mode];mx.synchronize();t=time.perf_counter();x,z=predict(m,mode,s,a);mx.synchronize();return x,z,(time.perf_counter()-t)*1000
for mode in MODES:
 for _ in range(5):x,z,dt=run(mode)
 if mode=='baseline':ref=x.copy();norm=z.copy()
 np.savez(r/(mode+'.npz'),action=x,normalized=z)
 errors[mode]=dict(exact=bool(np.array_equal(x,ref) and np.array_equal(z,norm)),max_abs=float(np.max(np.abs(x-ref))),rmse=float(np.sqrt(np.mean((x-ref)**2))))
 if not mode.startswith('steps'):assert errors[mode]['exact'],mode
 print('ready',mode,errors[mode],flush=True)
rng=np.random.default_rng(2005)
for i in range(50):
 for mode in rng.permutation(MODES):x,z,t=run(mode);rows[mode].append(t)
 if i%10==0:print(i,{k:round(float(np.mean(v)),3) for k,v in rows.items()},flush=True)
res={}
for k,v in rows.items():
 d=np.asarray(rows['baseline'])-v;ci=np.percentile(np.mean(d[rng.integers(0,len(d),(5000,len(d)))],1),[2.5,97.5]);res[k]=dict(mean_ms=float(np.mean(v)),p95_ms=float(np.percentile(v,95)),saving_ms=float(d.mean()),ci95=ci.tolist(),samples_ms=v,**errors[k])
(r/'results.json').write_text(json.dumps(res,indent=2));print('SCREEN DONE',flush=True)
refpath=Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full');cases=json.loads((refpath/'comparison.json').read_text())['cases'];checks=[]
for c in cases:
 label=c['label'];raw=np.load(refpath/('base_inputs.npz' if label.startswith('length_') else label+'_inputs.npz'));a=args(raw,c['prompt']);ref,norm,_=run('baseline');row={'label':label,'tokens':int(base.tokenize(c['prompt']).shape[1]),'errors':{}}
 for mode in MODES[1:]:
  x,z,_=run(mode);d=x-ref;row['errors'][mode]=dict(exact=bool(np.array_equal(x,ref) and np.array_equal(z,norm)),max_abs=float(np.abs(d).max()),rmse=float(np.sqrt(np.mean(d*d))),finite=bool(np.isfinite(x).all()))
  if not mode.startswith('steps'):assert row['errors'][mode]['exact'],(label,mode)
  np.savez(r/(label+'_'+mode+'.npz'),action=x,normalized=z)
 np.savez(r/(label+'_baseline.npz'),action=ref,normalized=norm);checks.append(row)
(r/'validation.json').write_text(json.dumps(checks,indent=2));print('VALIDATION DONE',len(checks),flush=True)
