import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import ast,copy,json,time,sys
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
from imagewam_mlx.model import ImageWAM as Base
from imagewam_mlx import prepacked_visual as pv
from imagewam_mlx.ops import gated
sv=pv.sv
root=Path('records/visual_epilogue_20261104');root.mkdir(exist_ok=False)
tree=ast.parse(Path(pv.__file__).read_text());call=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='kernel' for t in n.targets));src=ast.literal_eval(next(k.value for k in call.keywords if k.arg=='source'))
needle='y[size_t(r) * N + c] =\n                static_cast<T>(float(frag[e]) * x_scale[r] * w_scale[c]);'
replacement='T yy = static_cast<T>(float(frag[e]) * x_scale[r] * w_scale[c]); if(HAS_BIAS) yy=T(float(yy)+float(bias[c])); T scaled=T(float(yy)*float(alpha[c])); y[size_t(r)*N+c]=T(float(residual[size_t(r)*N+c])+float(scaled));'
assert src.count(needle)==2;src=src.replace(needle,replacement)
kernel=mx.fast.metal_kernel(name='wam_visual_residual_epilogue',input_names=['w','scales','biases','w_inv','w_scale','x8','x_scale','x_exp','alpha','residual','bias'],output_names=['y'],header=sv.header,source=src)
kernels={'epilogue':None}
class Candidate(pv.PrepackedVisual):
 def __init__(self,orig):self.w=orig.w;self.packed=orig.packed
 def single_post(self,x,att,mlp,p,m):
  key=p+'.linear2'
  if key not in self.packed:return super().single_post(x,att,mlp,p,m)
  z=mx.concatenate([att,gated(mlp)],axis=-1).reshape(-1,att.shape[-1]+mlp.shape[-1]//2);rows,k=z.shape;parts=self.packed[key];n=parts[0].shape[0];a,s,e=sv.quantize(z);bias=self.w.get(key+'.bias');has=bias is not None
  if bias is None:bias=mx.zeros((n,),x.dtype)
  y=kernel(inputs=[*parts,a,s,e,m[2].reshape(-1),x.reshape(rows,n),bias],template=[('T',x.dtype),('group_size',64),('bits',4),('x_span',0),('XK',k),('ROWS',rows),('HAS_BIAS',has)],grid=(((n+63)//64)*128,(rows+63)//64,1),threadgroup=(128,1,1),output_shapes=[(rows,n)],output_dtypes=[x.dtype])[0].reshape(x.shape)
  return y
base=ImageWAM('weights/bf16','assets',profile='speed',fused_concat=False);c=copy.copy(base);c.visual=Candidate(base.visual);c.prefill=mx.compile(lambda h,l,s:Base.prefill(c,h,l,s));models={'baseline':base,'epilogue':c}
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
def run(name):
 mx.synchronize();t=time.perf_counter();a,n=models[name].predict({k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')},raw['state'],prompt,noise);mx.synchronize();return a,n,(time.perf_counter()-t)*1000
outs={}
for name in models:
 for _ in range(5):a,n,dt=run(name)
 outs[name]=(a,n);np.savez(root/(name+'.npz'),action=a,normalized=n)
 assert np.array_equal(a,outs['baseline'][0]) and np.array_equal(n,outs['baseline'][1]),name
 print('warmup exact',name,flush=True)
rows={k:[] for k in models};rng=np.random.default_rng(94)
for i in range(160):
 for name in rng.permutation(list(models)):
  a,n,dt=run(name);rows[name].append(dt)
 if i%20==0:print(i,{k:float(np.mean(v)) for k,v in rows.items()},flush=True)
res={}
for name,v in rows.items():
 d=np.array(rows['baseline'])-v;ci=np.percentile(np.mean(d[rng.integers(0,len(d),(10000,len(d)))],1),[2.5,97.5]);res[name]=dict(mean_ms=float(np.mean(v)),saving_ms=float(d.mean()),ci95=ci.tolist(),samples_ms=v)
(root/'results.json').write_text(json.dumps(res,indent=2));print({k:{a:b for a,b in v.items() if a!='samples_ms'} for k,v in res.items()},flush=True)
cases=json.loads(Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full/comparison.json').read_text())['cases'];ref=Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full');checks=[]
for case in cases:
 label=case['label'];raw=np.load(ref/('base_inputs.npz' if label.startswith('length_') else label+'_inputs.npz'));prompt=case['prompt'];a,n,_=run('baseline')
 for name in kernels:
  b,z,_=run(name);assert np.array_equal(a,b) and np.array_equal(n,z),(label,name);np.savez(root/(label+'_'+name+'.npz'),action=b,normalized=z)
 checks.append(label)
(root/'validation.json').write_text(json.dumps(dict(exact_cases=checks,all_exact=True),indent=2));print('ALL EXACT',len(checks),flush=True)
