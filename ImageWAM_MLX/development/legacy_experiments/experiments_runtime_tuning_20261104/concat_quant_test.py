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
root=Path('records/concat_quant_20261104');root.mkdir(exist_ok=False)
tree=ast.parse(Path(sv.__file__).read_text());call=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='qx' for t in n.targets))
src=eval(compile(ast.Expression(next(k.value for k in call.keywords if k.arg=='source')),'source','eval'))
src=src.replace('const int K=x_shape[1],M=x_shape[0],max_exp=0,x_span=0;', 'const int KA=att_shape[1],KB=mlp_shape[1],K=KA+KB,M=att_shape[0],max_exp=0,x_span=0;')
needle='x += size_t(row) * K + lid * BK;';assert needle in src
src=src.replace(needle,'const device T* x = lid*BK < KA ? att + size_t(row)*KA + lid*BK : mlp + size_t(row)*KB + lid*BK-KA;')
qx=mx.fast.metal_kernel(name='wam_split_input_quant',input_names=['att','mlp'],output_names=['x8','x_scale','x_exp'],header=sv.header,source=src)
kernels={'split_quant':None}
class Candidate(pv.PrepackedVisual):
 def __init__(self,orig):self.w=orig.w;self.packed=orig.packed
 def single_post(self,x,att,mlp,p,m):
  key=p+'.linear2'
  if key not in self.packed:return super().single_post(x,att,mlp,p,m)
  g=gated(mlp);a0=att.reshape(-1,att.shape[-1]);g=g.reshape(-1,g.shape[-1]);rows=a0.shape[0];k=a0.shape[1]+g.shape[1];mp=(rows+63)//64*64;nb=k//64;threads=(nb+31)//32*32;assert a0.shape[1]%64==0 and g.shape[1]%64==0
  a,s,e=qx(inputs=[a0,g],template=[('T',a0.dtype)],grid=(threads,mp,1),threadgroup=(threads,1,1),output_shapes=[(mp,k),(rows,),(mp,nb+1)],output_dtypes=[mx.int8,mx.float32,mx.uint8])
  parts=self.packed[key];n=parts[0].shape[0]
  y=pv.kernel(inputs=[*parts,a,s,e],template=[('T',x.dtype),('group_size',64),('bits',4),('x_span',0),('XK',k),('ROWS',rows)],grid=(((n+63)//64)*128,(rows+63)//64,1),threadgroup=(128,1,1),output_shapes=[(rows,n)],output_dtypes=[x.dtype])[0].reshape(*x.shape[:-1],n)
  bias=self.w.get(key+'.bias')
  if bias is not None:y=y+bias
  return x+m[2]*y
base=ImageWAM('weights/bf16','assets',profile='speed',fused_concat=False);c=copy.copy(base);c.visual=Candidate(base.visual);c.prefill=mx.compile(lambda h,l,s:Base.prefill(c,h,l,s));models={'baseline':base,'split_quant':c}
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
