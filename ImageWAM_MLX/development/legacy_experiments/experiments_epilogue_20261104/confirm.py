import sys,time,json
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
import candidates
from split import fused
candidates.fused=fused
from candidates import install
root=Path('records/epilogue_20261104/split_confirm_paired');root.mkdir(exist_ok=False)
t=time.perf_counter();base=ImageWAM('weights/bf16','assets',profile='speed');candidate=install(base);mx.synchronize();load=time.perf_counter()-t
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
models={'base':base,'fused':candidate}
def run(mode,prompt=prompt,noise=noise,steps=10,profile=False):
 mx.synchronize();t=time.perf_counter();a,n=models[mode].predict(images,raw['state'],prompt,noise,steps=steps,profile=profile);mx.synchronize();assert np.isfinite(a).all();return a,n,(time.perf_counter()-t)*1000
first={}
for mode in models:
 a,n,ms=run(mode);first[mode]=ms
 for _ in range(3):run(mode)
rows=[];maxerr=0
for i in range(120):
 pair={}
 for mode in (('base','fused') if i%2==0 else ('fused','base')):pair[mode]=run(mode)
 maxerr=max(maxerr,float(np.max(np.abs(pair['base'][0]-pair['fused'][0]))));rows.append([pair['base'][2],pair['fused'][2]])
 if i%10==0:print('pair',i,rows[-1],maxerr,flush=True)
for mode in models:
 a,n,_=run(mode);np.savez(root/(mode+'_output.npz'),action=a,normalized=n)
x=np.array(rows);d=x[:,0]-x[:,1];rng=np.random.default_rng(42);ci=np.percentile(np.mean(d[rng.integers(0,len(d),(10000,len(d)))],axis=1),[2.5,97.5])
r=dict(load_s=load,first_ms=first,pairs=rows,base_ms=float(x[:,0].mean()),fused_ms=float(x[:,1].mean()),saving_ms=float(d.mean()),ci95=ci.tolist(),max_abs=maxerr,modules={},dynamic=[])
for mode in models:
 r['modules'][mode]=[]
 for _ in range(3):run(mode,profile=True);r['modules'][mode].append(dict(models[mode].profile))
for i,(pr,h) in enumerate([(prompt,1),(prompt,8),(prompt,32),(prompt,64),('Move carefully. '*60,16)]):
 z=np.resize(noise,(1,h,14));outs={}
 for mode in models:
  a,n,ms=run(mode,pr,z,3);outs[mode]=a;np.savez(root/f'dynamic_{i}_{mode}.npz',action=a,normalized=n,noise=z)
 diff=outs['base']-outs['fused'];r['dynamic'].append(dict(tokens=int(base.tokenize(pr).shape[1]),horizon=h,steps=3,max_abs=float(abs(diff).max()),rmse=float(np.sqrt((diff*diff).mean()))))
(root/'results.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k not in ('pairs','modules')},flush=True)
