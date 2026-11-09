import sys,time,json
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
from candidate import install,synchronize
root=Path('records/dual_stream_20261104/encoders');root.mkdir(exist_ok=False)
base=ImageWAM('weights/bf16','assets',profile='speed');models={'serial':install(base,False),'dual':install(base,True)}
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text()
def run(m):
 synchronize(m);start=time.perf_counter();ids=m.tokenize(prompt)
 with mx.stream(m.text_stream):h=m.qwen(mx.array(ids));mx.async_eval(h)
 with mx.stream(m.image_stream):x,st=m.preprocess(images,raw['state']);l=m.vae(x);mx.async_eval(l,st)
 mx.eval(h,l,st);synchronize(m);return h,l,st,(time.perf_counter()-start)*1000
for m in models.values():
 for _ in range(5):run(m)
rows={k:[] for k in models};outs={};rng=np.random.default_rng(63)
for i in range(100):
 for name in rng.permutation(list(models)):
  *out,dt=run(models[name]);outs[name]=out;rows[name].append(dt)
d=np.array(rows['serial'])-rows['dual'];ci=np.percentile(np.mean(d[rng.integers(0,100,(10000,100))],1),[2.5,97.5]);exact=all(bool(mx.array_equal(a,b).item()) for a,b in zip(outs['serial'],outs['dual']));result=dict(samples_ms=rows,means_ms={k:float(np.mean(v)) for k,v in rows.items()},saving_ms=float(d.mean()),ci95=ci.tolist(),exact=exact);(root/'results.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='samples_ms'},flush=True)
