import sys,time,json
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
from tile_candidates import install
root=Path('records/split_tune_20261104/tiles');root.mkdir(exist_ok=False)
m=ImageWAM('weights/bf16','assets',profile='speed');tiles={0:(32,64,2,2),1:(16,64,1,2),2:(16,128,1,4),3:(32,128,2,2)};models={t:install(m,512,tile) for t,tile in tiles.items()}
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
x,st=m.preprocess(images,raw['state']);cache=m.prefill(m.qwen(mx.array(m.tokenize(prompt))),m.vae(x),st);mx.eval(cache)
def run(t):
 mx.synchronize();start=time.perf_counter();out=models[t].denoise(cache,noise);mx.eval(out);mx.synchronize();return out,(time.perf_counter()-start)*1000
for t in models:
 for _ in range(4):run(t)
rows={t:[] for t in models};outs={};rng=np.random.default_rng(43)
for i in range(40):
 for t in rng.permutation(list(models)):
  out,ms=run(int(t));rows[int(t)].append(ms);outs[int(t)]=out
 if i%10==0:print(i,{t:round(np.mean(v),3) for t,v in rows.items()},flush=True)
reference=np.array(outs[0].astype(mx.float32));results={}
for t in models:
 out=np.array(outs[t].astype(mx.float32));assert np.isfinite(out).all();np.save(root/f'action_target{t}.npy',out)
 d=out-reference;results[t]=dict(tile=tiles[t],ms=rows[t],mean=float(np.mean(rows[t])),median=float(np.median(rows[t])),max_abs=float(abs(d).max()),rmse=float(np.sqrt(np.mean(d*d))))
(root/'sweep.json').write_text(json.dumps(results,indent=2));print({t:{k:v for k,v in r.items() if k!='ms'} for t,r in results.items()},flush=True)
