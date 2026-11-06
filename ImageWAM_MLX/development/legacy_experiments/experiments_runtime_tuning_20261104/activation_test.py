import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import copy,json,time,sys
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
from imagewam_mlx.model import ImageWAM as Base
from imagewam_mlx import prepacked_visual as pv
sv=pv.sv
root=Path('records/activation_int8_20261104');root.mkdir(exist_ok=True)
source=r"""
 uint row=threadgroup_position_in_grid.y,lid=thread_index_in_threadgroup;
 uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
 const int M=x_shape[0],K=x_shape[1],nb=K/64;
 threadgroup float shared[33];
 float v=0.0f;
 if(row<M){for(uint j=lid;j<K;j+=THREADS)v=max(v,fabs(float(x[row*K+j])));}
 v=simd_max(v);if(lane==0)shared[sg]=v;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(lid==0){float a=shared[0];for(int i=1;i<THREADS/32;i++)a=max(a,shared[i]);shared[32]=max(a*(1.0f/127.0f),1e-12f);if(row<M)s[row]=shared[32];}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float inv=1.0f/shared[32];
 for(uint j=lid;j<K;j+=THREADS){
  int z=row<M ? clamp(int(round(float(x[row*K+j])*inv)),-127,127):0;
  q[((row/64)*nb+j/64)*4096+(row%64)*64+j%64]=char(z);
 }
"""
qx=mx.fast.metal_kernel(name='wam_activation_coalesced',input_names=['x'],output_names=['q','s'],source=source)
kernels={'q128':128,'q256':256}
class Candidate(pv.PrepackedVisual):
 def __init__(self,orig,mode):self.w=dict(orig.w);self.packed=orig.packed;self.threads=kernels[mode]
 def linear(self,x,p):
  if p not in self.packed:return sv.RopeWeights.linear(self,x,p)
  shape=x.shape;x=x.reshape(-1,shape[-1]);m,k=x.shape;parts=self.packed[p];n=parts[0].shape[0];mp=(m+63)//64*64
  a,s=qx(inputs=[x],template=[('T',x.dtype),('THREADS',self.threads)],grid=(self.threads,mp,1),threadgroup=(self.threads,1,1),output_shapes=[(mp,k),(m,)],output_dtypes=[mx.int8,mx.float32]);e=mx.zeros((mp,k//64+1),mx.uint8)
  y=pv.kernel(inputs=[*parts,a,s,e],template=[('T',x.dtype),('group_size',64),('bits',4),('x_span',0),('XK',k),('ROWS',m)],grid=(((n+63)//64)*128,(m+63)//64,1),threadgroup=(128,1,1),output_shapes=[(m,n)],output_dtypes=[x.dtype])[0].reshape(*shape[:-1],n)
  bias=self.w.get(p+'.bias');return y if bias is None else y+bias
base=ImageWAM('weights/bf16','assets',profile='speed');models={'baseline':base}
for mode in kernels:
 c=copy.copy(base);c.visual=Candidate(base.visual,mode);c.prefill=mx.compile(lambda h,l,s,c=c:Base.prefill(c,h,l,s));models[mode]=c
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
for i in range(100):
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
