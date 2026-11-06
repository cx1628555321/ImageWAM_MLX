import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import ast,sys,time,json
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import split_quant_visual as sq
from imagewam_mlx.ops import gated
r=Path('records/gate_quant_20261104/vector');r.mkdir(exist_ok=False)
source=r"""
 const uint row=threadgroup_position_in_grid.y,lid=thread_index_in_threadgroup;
 const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
 const int M=att_shape[0],KA=att_shape[1],KB=mlp_shape[1]/2,K=XK,nb=K/64;
 constexpr int VPT=(XK+THREADS-1)/THREADS;
 threadgroup float shared[33];float vals[VPT];float maxv=0.0f;
 for(int i=0;i<VPT;i++){
  int j=lid+i*THREADS;float v=0.0f;
  if(row<M && j<K){
   if(j<KA)v=float(att[row*att_strides[0]+j*att_strides[1]]);
   else {int idx=j-KA;T u=mlp[row*mlp_strides[0]+idx*mlp_strides[1]];T sig=lut[as_type<ushort>(u)];T a=T(float(u)*float(sig));v=float(T(float(a)*float(mlp[row*mlp_strides[0]+(idx+KB)*mlp_strides[1]])));}
  }
  vals[i]=v;maxv=max(maxv,fabs(v));
 }
 maxv=simd_max(maxv);if(lane==0)shared[sg]=maxv;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(lid==0){float m=shared[0];for(int i=1;i<THREADS/32;i++)m=max(m,shared[i]);shared[32]=max(m*(1.0f/127.0f),1e-12f);if(row<M)x_scale[row]=shared[32];}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float inv=1.0f/shared[32];
 for(int i=0;i<VPT;i++){int j=lid+i*THREADS;if(j<K)x8[((row/64)*nb+j/64)*4096+(row%64)*64+j%64]=char(clamp(int(round(vals[i]*inv)),-127,127));}
 for(int j=lid;j<nb+1;j+=THREADS)x_exp[row*(nb+1)+j]=0;
"""
q=mx.fast.metal_kernel(name='wam_gate_lut_vector',input_names=['att','mlp','lut'],output_names=['x8','x_scale','x_exp'],source=source,ensure_row_contiguous=False)
M=328;KA=3072;KB=9216;K=KA+KB;mp=(M+63)//64*64;nb=K//64;threads=(nb+31)//32*32
lut=mx.sigmoid(mx.arange(65536,dtype=mx.uint16).view(mx.bfloat16));mx.eval(lut)
def quant(kernel,a,b,extra=[]):return kernel(inputs=[a,b,*extra],template=[('T',a.dtype)],grid=(threads,mp,1),threadgroup=(threads,1,1),output_shapes=[(mp,K),(M,),(mp,nb+1)],output_dtypes=[mx.int8,mx.float32,mx.uint8])
fns={'baseline':mx.compile(lambda a,b:quant(sq.qx,a,gated(b)))}
for nt in [128,256,512]:
 fns['v'+str(nt)]=mx.compile(lambda a,b,nt=nt:q(inputs=[a,b,lut],template=[('T',a.dtype),('XK',K),('THREADS',nt)],grid=(nt,mp,1),threadgroup=(nt,1,1),output_shapes=[(mp,K),(M,),(mp,nb+1)],output_dtypes=[mx.int8,mx.float32,mx.uint8]))
mx.random.seed(61);a=mx.random.normal((M,KA*2)).astype(mx.bfloat16)[:,::2];b=mx.random.normal((M,KA+KB*2)).astype(mx.bfloat16)[:,KA:];mx.eval(a,b)
ref=fns['baseline'](a,b);mx.eval(ref);checks={}
for name,f in fns.items():
 out=f(a,b);mx.eval(out);checks[name]=[dict(exact=bool(np.array_equal(np.array(x),np.array(y))),different=int(np.count_nonzero(np.array(x)!=np.array(y))),max_abs=float(np.max(np.abs(np.array(x).astype(float)-np.array(y).astype(float))))) for x,y in zip(ref,out)]
 for i in range(20):mx.eval(f(a,b))
print('CHECKS',checks,flush=True)
rows={k:[] for k in fns};rng=np.random.default_rng(74)
for i in range(500):
 for name in rng.permutation(list(fns)):
  mx.synchronize();t=time.perf_counter();out=fns[name](a,b);mx.eval(out);mx.synchronize();rows[name].append((time.perf_counter()-t)*1000)
res={}
for name,v in rows.items():
 d=np.array(rows['baseline'])-v;ci=np.percentile(np.mean(d[rng.integers(0,len(d),(10000,len(d)))],1),[2.5,97.5]);res[name]=dict(mean_ms=float(np.mean(v)),saving_ms=float(d.mean()),ci95=ci.tolist(),samples_ms=v,checks=checks[name])
(r/'micro.json').write_text(json.dumps(res,indent=2));print({k:{a:b for a,b in v.items() if a!='samples_ms'} for k,v in res.items()},flush=True)
