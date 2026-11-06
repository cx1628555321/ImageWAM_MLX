import sys,time,json,copy
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
from imagewam_mlx.runtime import RopeWeights
from imagewam_mlx.model import ImageWAM as Base
from staged_int8 import prepare,project
root=Path('records/runtime_tuning_20261104/native_confirm');root.mkdir(exist_ok=False)
from imagewam_mlx.staged_visual import StagedVisual
base=ImageWAM('weights/bf16','assets',profile='speed',staged_visual=False);c=copy.copy(base);c.visual=StagedVisual(base.visual);c.prefill=mx.compile(lambda h,l,s:Base.prefill(c,h,l,s));models={'baseline':base,'staged':c}
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
def run(name):
 mx.synchronize();t=time.perf_counter();a,n=models[name].predict(images,raw['state'],prompt,noise);mx.synchronize();assert np.isfinite(a).all();return a,n,(time.perf_counter()-t)*1000
for name in models:
 for _ in range(5):run(name)
rows={k:[] for k in models};outs={};rng=np.random.default_rng(83)
for i in range(120):
 for name in rng.permutation(list(models)):
  a,n,dt=run(name);rows[name].append(dt);outs[name]=(a,n)
 if i%20==0:print(i,{k:float(np.mean(v)) for k,v in rows.items()},flush=True)
results={}
for name in models:
 a,n=outs[name];np.savez(root/(name+'.npz'),action=a,normalized=n);d=np.array(rows['baseline'])-rows[name];ci=np.percentile(np.mean(d[rng.integers(0,len(d),(10000,len(d)))],1),[2.5,97.5]);diff=a-outs['baseline'][0];results[name]=dict(mean_ms=float(np.mean(rows[name])),samples_ms=rows[name],saving_ms=float(d.mean()),ci95=ci.tolist(),max_abs=float(abs(diff).max()),rmse=float(np.sqrt(np.mean(diff**2))))
(root/'results.json').write_text(json.dumps(results,indent=2));print({k:{a:b for a,b in v.items() if a!='samples_ms'} for k,v in results.items()},flush=True)
# Dynamic boundary checks; new input and noise each case, no feature caches.
extra=[];original_prompt=prompt
for prompt,horizon in [(original_prompt,1),(original_prompt+' Please carefully follow the instruction.'*25,16),(original_prompt,64)]:
 noise=np.resize(np.load(ex/'initial_noise_fp32.npy'),(1,horizon,14));a,n,_=run('baseline')
 for name in models:
  b,z,_=run(name);diff=b-a;idx=len(extra);np.savez(root/f'extra{idx}.npz',baseline=a,candidate=b,normalized=z);extra.append(dict(name=name,tokens=int(base.tokenize(prompt).shape[1]),horizon=horizon,max_abs=float(abs(diff).max()),rmse=float(np.sqrt(np.mean(diff**2)))))
(root/'extra.json').write_text(json.dumps(extra,indent=2));print('EXTRA COMPLETE',flush=True)

ref=Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full');cases=json.loads((ref/'comparison.json').read_text())['cases'];validation=[]
for case in cases:
 label=case['label'];raw=np.load(ref/('base_inputs.npz' if label.startswith('length_') else label+'_inputs.npz'));images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=case['prompt'];noise=np.load(ex/'initial_noise_fp32.npy');a,n,_=run('baseline');b,z,_=run('staged');d=b-a
 np.savez(root/(label+'.npz'),baseline=a,staged=b,normalized=z,token_ids=base.tokenize(prompt));validation.append(dict(label=label,tokens=int(base.tokenize(prompt).shape[1]),max_abs=float(abs(d).max()),rmse=float(np.sqrt(np.mean(d*d)))))
(root/'cases.json').write_text(json.dumps(validation,indent=2));print('FULL CASES',len(validation),'MAX',max(x['max_abs'] for x in validation),flush=True)
