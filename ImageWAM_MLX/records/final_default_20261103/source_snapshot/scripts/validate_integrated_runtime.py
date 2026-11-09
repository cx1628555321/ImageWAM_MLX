"""Compare packaged default against previous experimental optimized path."""
import json,time,sys,argparse,shutil
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM,LegacyImageWAM
from imagewam_mlx.ops import Weights
from visual_rope_experiments import VisualRopeCandidates
from compile_experiments import stats,error
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
new=ImageWAM('weights/bf16','assets');old=LegacyImageWAM.__new__(LegacyImageWAM);old.__dict__=dict(new.__dict__)
old.__dict__.pop('prefill',None);old.__dict__.pop('_compiled_step',None);old.action=Weights(new.action.w);old.visual=Weights(new.visual.w)
c=VisualRopeCandidates(old);c.set('visual_rope')
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};pr=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
def once(m,n=noise,steps=10,trace=False):
 mx.synchronize();t=time.perf_counter();out,_=m.predict(images,raw['state'],pr,n,steps=steps,trace=trace);mx.synchronize();return out,(time.perf_counter()-t)*1000
for _ in range(8):once(new);once(old)
r={'rows':[],'boundary':[]};times={'default':[],'previous':[]}
for i in range(30):
 outputs={}
 for label,m in ((('default',new),('previous',old)) if i%2==0 else (('previous',old),('default',new))):
  x,ms=once(m);times[label].append(ms);outputs[label]=x;r['rows'].append({'index':i,'mode':label,'ms':ms});np.save(a.output/f'{label}_{i:03d}.npy',x)
 assert np.array_equal(outputs['default'],outputs['previous'])
for horizon,steps in ((1,1),(8,3),(16,7),(32,10),(64,2)):
 n=np.random.default_rng(horizon).standard_normal((1,horizon,14)).astype(np.float32);x,_=once(new,n,steps);y,_=once(old,n,steps);r['boundary'].append({'horizon':horizon,'steps':steps,'error':error(x,y)});assert np.array_equal(x,y)
x,_=once(new,trace=True);y,_=once(new,trace=False);assert np.array_equal(x,y)
r.update(status='completed',times={k:stats(v) for k,v in times.items()},trace_equal=True,paired_all_bitwise=True)
(a.output/'results.json').write_text(json.dumps(r,indent=2));shutil.copyfile(__file__,a.output/Path(__file__).name);print(r['times'])
