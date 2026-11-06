import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,json,copy
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from common import *
r=ROOT/'integrated_validation';r.mkdir(exist_ok=False)
m=ImageWAM('weights/bf16','assets',profile='speed',text_cache_size=0,visual_shape_dispatch=True);assert m.visual_shape_dispatch
source=Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full');cases=json.loads((source/'comparison.json').read_text())['cases'];rows=[]
for c in cases:
 label=c['label'];raw=np.load(source/('base_inputs.npz' if label.startswith('length_') else label+'_inputs.npz'));args=inputs();args.update(images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')},state=raw['state'],prompt=c['prompt']);a,n=m.predict(**args,steps=10);ref=np.load(ROOT/'screen'/(label+'_visual_selected.npz'));exact=bool(np.array_equal(a,ref['action']) and np.array_equal(n,ref['normalized']));rows.append(dict(label=label,exact=exact));assert exact,label
 np.savez(r/(label+'.npz'),action=a,normalized=n)
# Integrated candidate versus explicit original visual path, shared immutable weights.
b=copy.copy(m);b.visual=DeviceVisual(m.visual);b.prefill=mx.compile(lambda h,l,s:Base.prefill(b,h,l,s));dynamic=[]
for h,longtext,alter in [(1,False,False),(16,True,False),(64,False,False),(16,False,True)]:
 args=inputs();args['noise']=np.resize(args['noise'],(1,h,14))
 if longtext:args['prompt']+=' Please carefully follow the instruction.'*25
 if alter:args['images']['cam_high']=args['images']['cam_high'][::2,::2].copy();args['state']=args['state'].copy();args['state'][0]+=.1
 a,n=b.predict(**args,steps=10,trace=True);ref={k:v.copy() for k,v in b.trace.items()};out,z=m.predict(**args,steps=10,trace=True);checks={k:bool(np.array_equal(v,m.trace[k])) for k,v in ref.items()};exact=bool(np.array_equal(a,out) and np.array_equal(n,z));assert exact and all(checks.values())
 dynamic.append(dict(horizon=h,tokens=int(m.tokenize(args['prompt']).shape[1]),altered=alter,action_exact=exact,trace_exact=checks))
 np.savez(r/f'dynamic_{len(dynamic)}_trace.npz',**m.trace)
(r/'results.json').write_text(json.dumps(dict(cases=rows,dynamic=dynamic),indent=2));print('integrated 51 cases + 4 dynamic traces exact',flush=True)
