import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,copy,time,json
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
from imagewam_mlx.model import ImageWAM as Base
from imagewam_mlx import split_quant_visual as sq
pv=sq.pv
root=Path('records/gate_quant_20261104/full');root.mkdir(exist_ok=False)
q=mx.fast.metal_kernel(name="wam_gate_silu_vector",input_names=["att","mlp","lut"],output_names=["x8","x_scale","x_exp"],source='\n const uint row=threadgroup_position_in_grid.y,lid=thread_index_in_threadgroup;\n const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;\n const int M=att_shape[0],KA=att_shape[1],KB=mlp_shape[1]/2,K=XK,nb=K/64;\n constexpr int VPT=(XK+THREADS-1)/THREADS;\n threadgroup float shared[33];float vals[VPT];float maxv=0.0f;\n for(int i=0;i<VPT;i++){\n  int j=lid+i*THREADS;float v=0.0f;\n  if(row<M && j<K){\n   if(j<KA)v=float(att[row*att_strides[0]+j*att_strides[1]]);\n   else {int idx=j-KA;T u=mlp[row*mlp_strides[0]+idx*mlp_strides[1]];T a=lut[as_type<ushort>(u)];v=float(T(float(a)*float(mlp[row*mlp_strides[0]+(idx+KB)*mlp_strides[1]])));}\n  }\n  vals[i]=v;maxv=max(maxv,fabs(v));\n }\n maxv=simd_max(maxv);if(lane==0)shared[sg]=maxv;\n threadgroup_barrier(mem_flags::mem_threadgroup);\n if(lid==0){float m=shared[0];for(int i=1;i<THREADS/32;i++)m=max(m,shared[i]);shared[32]=max(m*(1.0f/127.0f),1e-12f);if(row<M)x_scale[row]=shared[32];}\n threadgroup_barrier(mem_flags::mem_threadgroup);\n float inv=1.0f/shared[32];\n for(int i=0;i<VPT;i++){int j=lid+i*THREADS;if(j<K)x8[((row/64)*nb+j/64)*4096+(row%64)*64+j%64]=char(clamp(int(round(vals[i]*inv)),-127,127));}\n for(int j=lid;j<nb+1;j+=THREADS)x_exp[row*(nb+1)+j]=0;\n',ensure_row_contiguous=False)
class Candidate(sq.SplitQuantVisual):
 def __init__(self,orig):
  super().__init__(orig);keys=mx.arange(65536,dtype=mx.uint16).view(mx.bfloat16);self.lut=keys*mx.sigmoid(keys);mx.eval(self.lut)
 def single_post(self,x,att,mlp,p,m):
  key=p+'.linear2'
  if key not in self.packed:return super().single_post(x,att,mlp,p,m)
  a0=att.reshape(-1,att.shape[-1]);g=mlp.reshape(-1,mlp.shape[-1]);rows=a0.shape[0];k=a0.shape[1]+g.shape[1]//2;mp=(rows+63)//64*64;nb=k//64
  a,s,e=q(inputs=[a0,g,self.lut],template=[('T',x.dtype),('XK',k),('THREADS',1024)],grid=(1024,mp,1),threadgroup=(1024,1,1),output_shapes=[(mp,k),(rows,),(mp,nb+1)],output_dtypes=[mx.int8,mx.float32,mx.uint8])
  parts=self.packed[key];n=parts[0].shape[0]
  y=pv.kernel(inputs=[*parts,a,s,e],template=[('T',x.dtype),('group_size',64),('bits',4),('x_span',0),('XK',k),('ROWS',rows)],grid=(((n+63)//64)*128,(rows+63)//64,1),threadgroup=(128,1,1),output_shapes=[(rows,n)],output_dtypes=[x.dtype])[0].reshape(x.shape)
  bias=self.w.get(key+'.bias')
  if bias is not None:y=y+bias
  return x+m[2]*y
base=ImageWAM('weights/bf16','assets',profile='speed',fused_gate=False);c=copy.copy(base);c.visual=Candidate(base.visual);c.prefill=mx.compile(lambda h,l,s:Base.prefill(c,h,l,s));models={'baseline':base,'gate':c};kernels={'gate':None}
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
for i in range(200):
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
