import sys,time,json,copy,gc
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
from imagewam_mlx.quantized import PackedRopeWeights
from imagewam_mlx.workspace import workspace_step
root=Path('records/best_native_20261104/validation');root.mkdir(exist_ok=False)
t=time.perf_counter();best=ImageWAM('weights/bf16','assets',profile='speed');mx.synchronize();load=time.perf_counter()-t;assert best.action_epilogue
old=copy.copy(best);old.action=PackedRopeWeights.__new__(PackedRopeWeights);old.action.__dict__=dict(best.action.__dict__);old.action_epilogue=False;old._compiled_workspace_step=mx.compile(lambda cache,x,t,d,pe:workspace_step(old,cache,x,t,d,pe))
sys.path.insert(0,str(Path.cwd()/'development/legacy_experiments/experiments_split_tune_20261104'))
from tile_candidates import install
exp=install(old,512,(16,64,1,2),adaptive=True)
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
def run(model,steps=10,profile=False):
 mx.synchronize();t=time.perf_counter();a,n=model.predict(images,raw['state'],prompt,noise,steps=steps,profile=profile);mx.synchronize();assert np.isfinite(a).all();return a,n,(time.perf_counter()-t)*1000
first={}
for name,m in [('old',old),('best',best)]:
 a,n,first[name]=run(m)
 for _ in range(4):run(m)
rows=[]
for i in range(100):
 pair={}
 for name in (('old','best') if i%2==0 else ('best','old')):pair[name]=run(old if name=='old' else best)
 rows.append([pair['old'][2],pair['best'][2]]);(root/'pairs.json').write_text(json.dumps(rows))
 if i%20==0:print('pair',i,rows[-1],flush=True)
for name,m in [('old',old),('best',best)]:
 a,n,_=run(m);np.savez(root/f'{name}.npz',action=a,normalized=n)
x=np.array(rows);d=x[:,0]-x[:,1];rng=np.random.default_rng(42);ci=np.percentile(np.mean(d[rng.integers(0,len(d),(10000,len(d)))],1),[2.5,97.5]);modules={}
for name,m in [('old',old),('best',best)]:
 modules[name]=[]
 for _ in range(5):run(m,profile=True);modules[name].append(dict(m.profile))
r=dict(load_s=load,first_ms=first,old_ms=float(x[:,0].mean()),best_ms=float(x[:,1].mean()),saving_ms=float(d.mean()),ci95=ci.tolist(),modules=modules,device=mx.device_info());(root/'performance.json').write_text(json.dumps(r,indent=2));print(r,flush=True)
ref=Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full');cases=json.loads((ref/'comparison.json').read_text())['cases'];results=[]
for case in cases:
 label=case['label'];raw=np.load(ref/('base_inputs.npz' if label.startswith('length_') else label+'_inputs.npz'));images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=case['prompt']
 a,n,_=run(exp);b,c,_=run(best);assert np.array_equal(a,b) and np.array_equal(n,c),label
 np.savez(root/(label+'.npz'),experimental=a,native=b,normalized=c);results.append(dict(label=label,tokens=int(best.tokenize(prompt).shape[1]),exact=True))
(root/'cases.json').write_text(json.dumps(results,indent=2))
extra=[]
for h in (1,8,14,15,16,32,64):
 noise=np.resize(noise,(1,h,14));a,n,_=run(exp,3);b,c,_=run(best,3);assert np.array_equal(a,b) and np.array_equal(n,c),h;extra.append(dict(horizon=h,exact=True))
(root/'extra.json').write_text(json.dumps(extra,indent=2))
# Release all compiled closures before the accuracy model.
del best,old,exp,m;gc.collect();mx.clear_cache()
acc=ImageWAM('weights/bf16','assets',profile='accuracy');assert not acc.action_epilogue
raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
a,n,_=run(acc);reference=np.load('records/final_default_20261103/reference_action.npy');assert np.array_equal(a,reference)
np.savez(root/'accuracy.npz',action=a,normalized=n)
(root/'completed.json').write_text(json.dumps(dict(cases=len(results),native_exact=True,accuracy_unchanged=True),indent=2));print('VALIDATION COMPLETE',flush=True)
