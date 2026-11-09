"""Isolate prefix evaluation; encoder features resident, NOT end-to-end inference."""
import argparse,json,time,sys,shutil
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
from selective_kv_experiments import SelectiveCandidates
from compile_experiments import stats
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
m=ImageWAM('weights/bf16','assets');c=SelectiveCandidates(m);ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');im={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};image,st=m.preprocess(im,raw['state']);ids=m.tokenize((ex/'prompt.txt').read_text());hidden=m.qwen(mx.array(ids));latent=m.vae(image);mx.eval(hidden,latent,st)
def once(mode):
 c.set(mode);mx.synchronize();t=time.perf_counter();cache=m.prefill(hidden,latent,st);mx.eval(cache);mx.synchronize();return cache,(time.perf_counter()-t)*1000
for _ in range(8):once('baseline');once('selective_kv')
r={'status':'running','measurement':'prefix only, resident encoder features; excludes encoders/action/CPU preprocessing; not end-to-end','rows':[]};times={'baseline':[],'selective_kv':[]};out={}
for i in range(100):
 for mode in (('baseline','selective_kv') if i%2==0 else ('selective_kv','baseline')):
  cache,ms=once(mode);out[mode]=cache;times[mode].append(ms);r['rows'].append({'mode':mode,'index':i,'ms':ms})
 for j in range(25):
  for k in range(2):
   # Checks outside timing; explicit evaluation above materializes all returned K/V.
   assert np.array_equal(np.array(out['baseline'][j][k].astype(mx.float32)),np.array(out['selective_kv'][j][k].astype(mx.float32)))
r.update(status='completed',times={k:stats(v) for k,v in times.items()},all_kv_bitwise_equal=True)
for mode in times:
 for i,label in enumerate(('k','v')):np.save(a.output/f'{mode}_last_{label}.npy',np.array(out[mode][-1][i].astype(mx.float32)))
shutil.copyfile(__file__,a.output/Path(__file__).name);(a.output/'results.json').write_text(json.dumps(r,indent=2));print(r['times'])
