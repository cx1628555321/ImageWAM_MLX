import sys,json,time
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
from candidate import install,synchronize
root=Path('records/dual_stream_20261104/validation');root.mkdir(exist_ok=False)
base=ImageWAM('weights/bf16','assets',profile='speed');serial=install(base,False);dual=install(base,True)
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
traces={};outputs={}
for name,m in [('baseline',base),('serial',serial),('dual',dual)]:
 a,n=m.predict(images,raw['state'],prompt,noise,trace=True);synchronize(m);traces[name]=dict(m.trace);outputs[name]=(a,n);np.savez(root/(name+'_trace.npz'),action=a,normalized=n,**m.trace)
checks={}
for name in ('baseline','serial'):
 checks[name]={k:dict(exact=bool(np.array_equal(v,traces['dual'][k])),max_abs=float(np.max(np.abs(v.astype(np.float32)-traces['dual'][k].astype(np.float32))))) for k,v in traces[name].items()};assert all(x['exact'] for x in checks[name].values()),checks[name]
(root/'trace_comparison.json').write_text(json.dumps(checks,indent=2));print('TRACE EXACT',flush=True)
ref=Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full');cases=json.loads((ref/'comparison.json').read_text())['cases'];rows=[]
for case in cases:
 label=case['label'];raw=np.load(ref/('base_inputs.npz' if label.startswith('length_') else label+'_inputs.npz'));images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=case['prompt']
 a,n=base.predict(images,raw['state'],prompt,noise);b,z=dual.predict(images,raw['state'],prompt,noise);synchronize(dual);exact=np.array_equal(a,b) and np.array_equal(n,z);assert np.isfinite(b).all();assert exact,label
 np.savez(root/(label+'.npz'),baseline=a,dual=b,normalized=z,token_ids=base.tokenize(prompt));rows.append(dict(label=label,tokens=int(base.tokenize(prompt).shape[1]),exact=bool(exact)))
(root/'cases.json').write_text(json.dumps(rows,indent=2));print('FULL CASES',len(rows),flush=True)
raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompts=[(ex/'prompt.txt').read_text(),min(cases,key=lambda c:base.tokenize(c['prompt']).shape[1])['prompt'],max(cases,key=lambda c:base.tokenize(c['prompt']).shape[1])['prompt']];prompts.append(prompts[-1]+' Please carefully follow the instruction.'*20);extra=[]
for ci,prompt in enumerate(prompts):
 for horizon in (1,16,64):
  noise=np.resize(np.load(ex/'initial_noise_fp32.npy'),(1,horizon,14));a,n=base.predict(images,raw['state'],prompt,noise);b,z=dual.predict(images,raw['state'],prompt,noise);synchronize(dual);exact=np.array_equal(a,b) and np.array_equal(n,z);assert exact,(ci,horizon);assert np.isfinite(b).all()
  np.savez(root/f'extra{ci}_h{horizon}.npz',baseline=a,dual=b,normalized=z,token_ids=base.tokenize(prompt));extra.append(dict(tokens=int(base.tokenize(prompt).shape[1]),horizon=horizon,exact=bool(exact)))
(root/'extra.json').write_text(json.dumps(extra,indent=2));print('EXTRA EXACT',len(extra),flush=True)
(root/'streams.json').write_text(json.dumps(dict(text=str(dual.text_stream),image=str(dual.image_stream)),indent=2))
