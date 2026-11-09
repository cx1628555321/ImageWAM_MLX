"""Sequential paired evaluation; record rejected candidates as well as successes."""
import argparse,json,time,sys,traceback,shutil
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
from remaining_candidates import BUILDERS
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=list(BUILDERS),required=True);p.add_argument('--pairs',type=int,default=40);p.add_argument('--baseline-mode',choices=['overlap']);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
shutil.copyfile(Path(__file__).with_name('remaining_candidates.py'),a.output/'candidate_source.py')
base=ImageWAM('weights/bf16','assets',fused_vae=True,overlap=False,qwen_rope=False,vae_group_inv=False)
if a.baseline_mode:base=BUILDERS[a.baseline_mode](base)
candidate=BUILDERS[a.mode](base)
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
r={'mode':a.mode,'baseline_mode':a.baseline_mode,'status':'running','pairs':[],'checks':[]}
def save():(a.output/'results.json').write_text(json.dumps(r,indent=2))
def once(m,pr=prompt,n=noise,steps=10):
 mx.synchronize();t=time.perf_counter();out,_=m.predict(images,raw['state'],pr,n,steps=steps);mx.synchronize();return out,(time.perf_counter()-t)*1000
try:
 print('WARMUP BASE',flush=True)
 for _ in range(5):once(base)
 print('FIRST CANDIDATE',flush=True)
 x,_=once(base);y,first=once(candidate);r['first_ms']=first;print('FIRST MS',first,flush=True);r['initial_max_abs']=float(np.max(np.abs(x-y)))
 assert np.isfinite(y).all()
 for _ in range(5):once(candidate)
 times={'base':[],'candidate':[]}
 for i in range(a.pairs):
  outs={}
  for tag,m in ((('base',base),('candidate',candidate)) if i%2==0 else (('candidate',candidate),('base',base))):
   out,ms=once(m);outs[tag]=out;times[tag].append(ms)
  if i%5==0:print('PAIR',i,times['base'][-1],times['candidate'][-1],flush=True)
  r['pairs'].append({'base_ms':times['base'][-1],'candidate_ms':times['candidate'][-1],'equal':bool(np.array_equal(outs['base'],outs['candidate']))})
 for tag,out in outs.items():np.save(a.output/(tag+'_action.npy'),out)
 inv=json.loads(Path('../ImageWAM/development_records/m5pro_robotwin/20261103_dynamic_optimization/text_lengths.json').read_text())['examples_by_length']
 cases=[(inv[str(l)]['prompt'],noise,10) for l in (31,79,48,31)]+[('Move carefully. '*60,noise,10)]
 cases += [(prompt,np.random.default_rng(h).standard_normal((1,h,14)).astype(np.float32),steps) for h,steps in ((1,1),(8,3),(16,7),(32,10),(64,2))]
 for i,(pr,n,steps) in enumerate(cases):
  x,_=once(base,pr,n,steps);y,_=once(candidate,pr,n,steps)
  np.savez(a.output/f'check_{i}.npz',baseline=x,candidate=y)
  r['checks'].append({'tokens':base.tokenize(pr).shape[1],'horizon':n.shape[1],'steps':steps,'equal':bool(np.array_equal(x,y)),'max_abs':float(np.max(np.abs(x-y)))})
 d=np.array(times['base'])-np.array(times['candidate']);rng=np.random.default_rng(42)
 r.update(status='completed',baseline_ms=float(np.mean(times['base'])),candidate_ms=float(np.mean(times['candidate'])),saving_ms=float(np.mean(d)),saving_ci95=np.percentile(rng.choice(d,(10000,len(d)),replace=True).mean(1),[2.5,97.5]).tolist(),all_equal=all(x['equal'] for x in r['pairs']+r['checks']))
 save();print(json.dumps({k:v for k,v in r.items() if k not in ('pairs','checks')}),flush=True)
except BaseException:
 r['status']='failed';r['exception']=traceback.format_exc();save();raise
