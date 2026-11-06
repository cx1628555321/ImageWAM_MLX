import sys,time,json
from pathlib import Path
import mlx.core as mx
import numpy as np
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
from candidate import install,synchronize
root=Path('records/dual_stream_20261104/paired');root.mkdir(parents=True,exist_ok=False)
t=time.perf_counter();base=ImageWAM('weights/bf16','assets',profile='speed');load=time.perf_counter()-t
models={'baseline':base,'serial_control':install(base,False),'dual':install(base,True)}
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
def run(name):
 m=models[name];synchronize(m);t=time.perf_counter();a,n=m.predict(images,raw['state'],prompt,noise);synchronize(m);dt=(time.perf_counter()-t)*1000;assert np.isfinite(a).all();return a,n,dt
first={}
for name in models:
 _,_,first[name]=run(name)
 for _ in range(5):run(name)
rows={n:[] for n in models};outs={};rng=np.random.default_rng(172)
for i in range(120):
 for name in rng.permutation(list(models)):
  a,n,ms=run(name);rows[name].append(ms);outs[name]=(a,n)
 if i%20==0:print(i,{n:float(np.mean(r)) for n,r in rows.items()},flush=True)
results={}
for name in models:
 a,n=outs[name];np.savez(root/f'{name}.npz',action=a,normalized=n);d=np.array(rows['baseline'])-rows[name];ci=np.percentile(np.mean(d[rng.integers(0,len(d),(10000,len(d)))],axis=1),[2.5,97.5]);diff=a-outs['baseline'][0]
 results[name]=dict(mean_ms=float(np.mean(rows[name])),median_ms=float(np.median(rows[name])),p95_ms=float(np.percentile(rows[name],95)),samples_ms=rows[name],saving_ms=float(d.mean()),ci95=ci.tolist(),max_abs=float(abs(diff).max()),rmse=float(np.sqrt(np.mean(diff*diff))),first_ms=first[name])
d=np.array(rows['serial_control'])-rows['dual'];ci=np.percentile(np.mean(d[rng.integers(0,len(d),(10000,len(d)))],axis=1),[2.5,97.5]);result=dict(load_s=load,results=results,dual_vs_serial=dict(saving_ms=float(d.mean()),ci95=ci.tolist()),device=mx.device_info(),peak_bytes=mx.get_peak_memory());(root/'results.json').write_text(json.dumps(result,indent=2));print({n:{k:v for k,v in r.items() if k!='samples_ms'} for n,r in results.items()},flush=True);print(result['dual_vs_serial'],flush=True)
