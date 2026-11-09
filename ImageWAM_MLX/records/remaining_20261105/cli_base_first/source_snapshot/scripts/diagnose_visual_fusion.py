import sys,time,json,copy,types,faulthandler
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
from imagewam_mlx.model import ImageWAM as Base
from next_optimization_candidates import FullFusedWeights
out=Path('records/next_optimizations_20261103/visual_fusion_diagnosis');out.mkdir(exist_ok=False)
faulthandler.dump_traceback_later(60,repeat=True)
m=ImageWAM('weights/bf16','assets');ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text()
x,st=m.preprocess(images,raw['state']);hidden=m.qwen(mx.array(m.tokenize(prompt)));latent=m.vae(x);mx.eval(hidden,latent,st)
c=copy.copy(m);c.visual=FullFusedWeights(m.visual.w)
r={}
for tag,fn in [('baseline',m.prefill),('eager',types.MethodType(Base.prefill,c)),('compiled',mx.compile(types.MethodType(Base.prefill,c)))]:
 print('START',tag,flush=True);start=time.perf_counter();kv=fn(hidden,latent,st);mx.eval(kv);mx.synchronize();r[tag]={'ms':(time.perf_counter()-start)*1000};print(tag,r[tag],flush=True)
 arrays={f'{j}_{kind}':np.array(v.astype(mx.float32)) for j,pair in enumerate(kv) for kind,v in zip(('k','v'),pair)}
 np.savez(out/(tag+'.npz'),**arrays)
 if tag=='baseline':ref=arrays
 else:r[tag]['max_abs']=max(float(np.max(np.abs(v-ref[k]))) for k,v in arrays.items())
 (out/'results.json').write_text(json.dumps(r,indent=2))
faulthandler.cancel_dump_traceback_later()
