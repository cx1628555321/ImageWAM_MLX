import mlx.core as mx
from imagewam_mlx.gate_quant_visual import GateQuantVisual
from imagewam_mlx import prepacked_visual as pv
from imagewam_mlx.ops import norm
q=mx.fast.metal_kernel(name='wam_norm_mod_quant',input_names=['x','modscale','shift'],output_names=['x8','x_scale','x_exp'],source='\n const uint row=threadgroup_position_in_grid.y,lid=thread_index_in_threadgroup;\n const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;\n const int M=x_shape[0],K=XK,nb=K/64;\n constexpr int VPT=(XK+THREADS-1)/THREADS;\n threadgroup float shared[33];float vals[VPT];float maxv=0.0f;\n\n float sum=0.0f,ss=0.0f;\n for(int i=0;i<VPT;i++){int j=lid+i*THREADS;float v=(row<M && j<K)?float(x[row*x_strides[0]+j*x_strides[1]]):0.0f;vals[i]=v;sum+=v;ss+=v*v;}\n threadgroup float squares[32];\n sum=simd_sum(sum);ss=simd_sum(ss);\n if(lane==0){shared[sg]=sum;squares[sg]=ss;}\n threadgroup_barrier(mem_flags::mem_threadgroup);\n if(lid==0){float s=0.0f,t=0.0f;for(int i=0;i<THREADS/32;i++){s+=shared[i];t+=squares[i];}float mean=s/float(K);shared[32]=mean;squares[0]=rsqrt(max(t/float(K)-mean*mean,0.0f)+1e-6f);}\n threadgroup_barrier(mem_flags::mem_threadgroup);\n float mean=shared[32],rn=squares[0];\n threadgroup_barrier(mem_flags::mem_threadgroup);\n for(int i=0;i<VPT;i++){\n  int j=lid+i*THREADS;float v=0.0f;\n  if(row<M && j<K){\n   T scale=T(1.0f+float(modscale[j]));\n   T product=T(float(scale)*float(T((vals[i]-mean)*rn)));\n   v=float(T(float(product)+float(shift[j])));\n  }\n  vals[i]=v;maxv=max(maxv,fabs(v));\n }\n maxv=simd_max(maxv);if(lane==0)shared[sg]=maxv;\n threadgroup_barrier(mem_flags::mem_threadgroup);\n if(lid==0){float m=shared[0];for(int i=1;i<THREADS/32;i++)m=max(m,shared[i]);shared[32]=max(m*(1.0f/127.0f),1e-12f);if(row<M)x_scale[row]=shared[32];}\n threadgroup_barrier(mem_flags::mem_threadgroup);\n float inv=1.0f/shared[32];\n for(int i=0;i<VPT;i++){int j=lid+i*THREADS;if(j<K)x8[((row/64)*nb+j/64)*4096+(row%64)*64+j%64]=char(clamp(int(round(vals[i]*inv)),-127,127));}\n for(int j=lid;j<nb+1;j+=THREADS)x_exp[row*(nb+1)+j]=0;\n',ensure_row_contiguous=False)
class Candidate(GateQuantVisual):
 def __init__(self,orig):
  self.__dict__.update(orig.__dict__)
 def single_prepare(self,x,p,m,pe):
  key=p+'.linear1'
  if key not in self.packed:return super().single_prepare(x,p,m,pe)
  inp=x.reshape(-1,x.shape[-1]);rows,k=inp.shape;mp=(rows+63)//64*64
  a,s,e=q(inputs=[inp,m[1].reshape(-1),m[0].reshape(-1)],template=[('T',x.dtype),('XK',k),('THREADS',256)],grid=(256,mp,1),threadgroup=(256,1,1),output_shapes=[(mp,k),(rows,),(mp,k//64+1)],output_dtypes=[mx.int8,mx.float32,mx.uint8])
  parts=self.packed[key];n=parts[0].shape[0]
  z=pv.kernel(inputs=[*parts,a,s,e],template=[('T',x.dtype),('group_size',64),('bits',4),('x_span',0),('XK',k),('ROWS',rows)],grid=(((n+63)//64)*128,(rows+63)//64,1),threadgroup=(128,1,1),output_shapes=[(rows,n)],output_dtypes=[x.dtype])[0].reshape(*x.shape[:-1],n)
  bias=self.w.get(key+'.bias')
  if bias is not None:z=z+bias
  q0,k0,v=self.qkv(z[...,:9216],p+'.norm',pe)
  return q0,k0,v,z[...,9216:]
