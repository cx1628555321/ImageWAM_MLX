"""Separate-process cold-shape / explicit-prewarm experiment, real prompt corpus."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['cold','warm','first'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
m=ImageWAM('weights/bf16','assets');ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};noise=np.load(ex/'initial_noise_fp32.npy')
inv=json.loads(Path('../ImageWAM/development_records/m5pro_robotwin/20261103_dynamic_optimization/text_lengths.json').read_text())['examples_by_length']
prompts=[inv[k]['prompt'] for k in sorted(inv,key=int)]+['Move carefully. '*60]
samples=[dict(images=images,state=raw['state'],prompt=pr,noise=noise) for pr in prompts]
r={'mode':a.mode,'cases':[],'warmup':[],'note':'First means first measured request per shape in this process; not cold driver/disk. Warm mode prewarms the listed corpus only.'}
def save():(a.output/'results.json').write_text(json.dumps(r,indent=2))
if a.mode!='cold':
 r['warmup']=m.warmup(samples if a.mode=='warm' else samples[:1]);save();print('PREWARM DONE',flush=True)
for i,s in enumerate(samples+[dict(samples[0],prompt='Move the object safely. '*29)]+[samples[0]]):
 times=[];outs=[]
 for _ in range(3):
  mx.synchronize();start=time.perf_counter();out,n=m.predict(**s);mx.synchronize();times.append((time.perf_counter()-start)*1000);outs.append(out)
  assert np.isfinite(out).all()
 np.savez(a.output/f'case_{i}.npz',action=out,normalized=n)
 row={'case':i,'tokens':m.tokenize(s['prompt']).shape[1],'prewarmed':(a.mode=='warm' and i!=len(samples)) or (a.mode=='first' and i in (0,len(samples)+1)),'times_ms':times,'repeats_exact':all(x.tobytes()==outs[0].tobytes() for x in outs)}
 r['cases'].append(row);save();print(row,flush=True)
r['status']='completed';save()
