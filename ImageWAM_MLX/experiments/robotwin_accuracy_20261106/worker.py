import os,sys,json,time,importlib.metadata
from pathlib import Path
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
root=Path.cwd();sys.path.insert(0,str(root));sys.path.insert(0,str(root/'experiments/background_20261105'))
import numpy as np
import mlx.core as mx
import imagewam_mlx.runtime as rt
variant=sys.argv[1];out=Path(sys.argv[2]);out.mkdir(exist_ok=False)
version=importlib.metadata.version('mlx');expected='0.32.4' if variant in ('accuracy','stock') else '0.32.5.dev20261106';assert version==expected
if variant not in ('accuracy','stock'):
 source=Path(rt.__file__).read_text();guard="version('mlx')=='0.32.4'";assert guard in source;exec(compile(source.replace(guard,'True'),rt.__file__,'exec'),rt.__dict__)
t=time.perf_counter()
if variant=='accuracy':m=rt.ImageWAM('weights/bf16','assets',profile='accuracy',text_cache_size=0,vae_asymmetric=False)
else:
 m=rt.ImageWAM('weights/bf16','assets',profile='speed',prefix_precision='all_int8',text_cache_size=1,vae_asymmetric=False)
 if variant.startswith('native-'):
  from candidates import install
  install(m,variant.replace('-','_'))
 elif variant=='action-bf16':
  from imagewam_mlx.workspace import workspace_step
  m.action=rt.RopeWeights(mx.load('weights/bf16/action.safetensors'));mx.eval(m.action.w)
  m._compiled_workspace_step=mx.compile(lambda c,x,t,d,p:workspace_step(m,c,x,t,d,p))
mx.synchronize();load=time.perf_counter()-t
r=root/'records/robotwin_accuracy_20261106';manifest=json.loads((r/'manifest.json').read_text());rows=[]
for case in manifest['cases']:
 raw=np.load(r/'inputs'/(case['label']+'.npz'))
 for seed in manifest['seeds']:
  key=case['label']+f'_s{seed}';noise=np.load(r/'inputs'/f'noise_{seed}.npy')
  a,n=m.predict(images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')},state=raw['state'],prompt=case['prompt'],noise=noise,steps=10)
  assert a.shape==n.shape==(16,14) and np.isfinite(a).all() and np.isfinite(n).all()
  np.savez(out/(key+'.npz'),action=a,normalized=n)
  rows.append(dict(key=key,episode=case['episode'],frame=case['frame'],seed=seed,tokens=int(m.tokenize(case['prompt']).shape[1])))
 print('CASE',variant,case['label'],len(rows),flush=True)
report=dict(status='completed',variant=variant,mlx=version,load_s=load,cases=rows,count=len(rows),cache=m.text_cache_stats(),peak_bytes=mx.get_peak_memory())
(out/'results.json').write_text(json.dumps(report,indent=2));print('DONE',variant,len(rows),flush=True)
