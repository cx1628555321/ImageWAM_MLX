import sys,time,json
from pathlib import Path
import mlx.core as mx
import numpy as np
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
import candidates
from split import fused
candidates.fused=fused
root=Path('records/epilogue_20261104/action_paired');root.mkdir(exist_ok=False)
m=ImageWAM('weights/bf16','assets',profile='speed');c=candidates.install(m)
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
x,st=m.preprocess(images,raw['state']);cache=m.prefill(m.qwen(mx.array(m.tokenize(prompt))),m.vae(x),st);mx.eval(cache)
models={'base':m,'fused':c}
def run(mode):
 mx.synchronize();t=time.perf_counter();out=models[mode].denoise(cache,noise);mx.eval(out);mx.synchronize();ms=(time.perf_counter()-t)*1000;return out,ms
for _ in range(5):run('base');run('fused')
rows=[]
for i in range(120):
 pair={}
 for mode in (('base','fused') if i%2==0 else ('fused','base')):pair[mode]=run(mode)
 rows.append([pair['base'][1],pair['fused'][1]])
 if i%30==0:print(i,rows[-1],flush=True)
a=np.array(pair['base'][0].astype(mx.float32));b=np.array(pair['fused'][0].astype(mx.float32));np.savez(root/'outputs.npz',base=a,fused=b)
x=np.array(rows);d=x[:,0]-x[:,1];rng=np.random.default_rng(42);ci=np.percentile(np.mean(d[rng.integers(0,len(d),(10000,len(d)))],1),[2.5,97.5]);r=dict(pairs=rows,base_ms=float(x[:,0].mean()),fused_ms=float(x[:,1].mean()),saving_ms=float(d.mean()),ci95=ci.tolist(),max_abs=float(abs(a-b).max()),rmse=float(np.sqrt(np.mean((a-b)**2))))
(root/'results.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k!='pairs'},flush=True)
