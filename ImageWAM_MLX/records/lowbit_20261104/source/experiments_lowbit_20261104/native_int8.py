"""Experimental genuine INT8 TensorOps, FP32 scales, BF16 output.
GEMM adapted from MIT Cider; see kernels/LICENSE.cider. No BF16 weight expansion.
"""
from pathlib import Path
import re
import mlx.core as mx
HEADER=re.sub(r'\bhalf\b','bfloat',(Path(__file__).parent/'kernels/cider_w8a8.metal').read_text())
HEADER=HEADER.replace('decltype(ct_a)', 'typename mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup>::template cooperative_tensor_left_input_t<int8_t, int8_t, int32_t>').replace('decltype(ct_b)', 'typename mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup>::template cooperative_tensor_right_input_t<int8_t, int8_t, int32_t>')
_desc=re.search(r'  constexpr auto desc = (mpp::tensor_ops::matmul2d_descriptor\(.*?\));',HEADER,re.S).group(1)
HEADER=re.sub(r'  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor\(.*?\);','',HEADER,flags=re.S)
HEADER=HEADER.replace('using namespace metal;', 'using namespace metal;\nconstant constexpr auto desc = '+_desc+';')
_QUANT=mx.fast.metal_kernel(name='wam_i8_row_quant',input_names=['x'],output_names=['q','scales'],source=r'''
 uint row=threadgroup_position_in_grid.x,tid=thread_position_in_threadgroup.x;
 threadgroup float vals[8];threadgroup float scale;
 float v=0;for(uint j=tid;j<K;j+=256)v=max(v,abs(float(x[row*K+j])));
 v=simd_max(v);if(tid%32==0)vals[tid/32]=v;threadgroup_barrier(mem_flags::mem_threadgroup);
 if(tid<32){float a=tid<8?vals[tid]:0;a=simd_max(a);if(tid==0){scale=max(a/127.0f,1e-12f);scales[row]=scale;}}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint j=tid;j<K;j+=256)q[row*K+j]=char(clamp(rint(float(x[row*K+j])/scale),-127.0f,127.0f));
''')
def quantize_rows(x):
 x=mx.contiguous(x.reshape(-1,x.shape[-1]));m,k=x.shape
 return tuple(_QUANT(inputs=[x],template=[('K',k)],grid=(m*256,1,1),threadgroup=(256,1,1),output_shapes=[x.shape,(m,)],output_dtypes=[mx.int8,mx.float32]))

def pad2(x,m,n):return mx.pad(x,[(0,m-x.shape[0]),(0,n-x.shape[1])])
_KERNELS={}
def int8_gemm(q,w,sx=None,sw=None,bias=None,raw=False):
 m,k=q.shape;n=w.shape[0];bm=32 if m<=64 else 128;threads=128 if bm==32 else 512
 bk=128 if k<512 else 512;kp=(k+bk-1)//bk*bk;npad=(n+127)//128*128;mpad=(m+bm-1)//bm*bm
 q=pad2(q,mpad,kp);w=pad2(w,npad,kp)
 key=(raw,bm,bk)
 if key not in _KERNELS:
  names=['q','w'] if raw else ['q','w','sx','sw','bias']
  fn='w8a8_gemm_int32_impl' if raw else 'w8a8_gemm_impl'
  middle='' if raw else 'sx,sw,bias,'
  src=f'{fn}<{bm},128,{bk},32,{bm//32},4>(q,w,out,M,N,K,{middle}0,(M+{bm}-1)/{bm},(N+127)/128,threadgroup_position_in_grid.xy,thread_position_in_threadgroup.x/32,thread_position_in_threadgroup.x%32);'
  _KERNELS[key]=mx.fast.metal_kernel(name=f'wam_native_i8_{raw}_{bm}_{bk}',input_names=names,output_names=['out'],header=HEADER,source=src)
 if not raw:
  sx=mx.pad(sx,[(0,max(0,16-sx.size))]);sw=mx.pad(sw,[(0,max(0,16-sw.size))]);bias=mx.pad(bias,[(0,max(0,16-bias.size))])
 inputs=[q,w] if raw else [q,w,sx,sw,bias]
 return _KERNELS[key](inputs=inputs,template=[('M',m),('N',n),('K',kp)],grid=(((n+127)//128)*threads,(m+bm-1)//bm,1),threadgroup=(threads,1,1),output_shapes=[(m,n)],output_dtypes=[mx.int32 if raw else mx.bfloat16])[0]

def pack_weight(w):
 q,s=quantize_rows(w);mx.eval(q,s);return q,s

def linear(x,weight,bias=None):
 q,s=quantize_rows(x);w,sw=weight;n=w.shape[0]
 b=mx.zeros((n,),dtype=mx.bfloat16) if bias is None else bias
 return int8_gemm(q,w,s,sw,b).reshape(*x.shape[:-1],n)

_BATCH={}
def batched_gemm(q,w,sx,sw):
 b,m,k=q.shape;_,n,_=w.shape;bm=32 if m<=64 else 128;threads=128 if bm==32 else 512;bk=128 if k<512 else 512
 kp=(k+bk-1)//bk*bk;npad=(n+127)//128*128;mpad=(m+bm-1)//bm*bm
 q=mx.pad(q,[(0,0),(0,mpad-m),(0,kp-k)]);w=mx.pad(w,[(0,0),(0,npad-n),(0,kp-k)]);bias=mx.zeros((max(n,16),),dtype=mx.bfloat16)
 key=(bm,bk)
 if key not in _BATCH:
  src=f'''uint z=threadgroup_position_in_grid.z;
  w8a8_gemm_impl<{bm},128,{bk},32,{bm//32},4>(q+z*MP*K,w+z*NP*K,out+z*M*N,M,N,K,sx+z*M,sw+z*N,bias,0,(M+{bm}-1)/{bm},(N+127)/128,threadgroup_position_in_grid.xy,thread_position_in_threadgroup.x/32,thread_position_in_threadgroup.x%32);'''
  _BATCH[key]=mx.fast.metal_kernel(name=f'wam_batched_i8_{bm}_{bk}',input_names=['q','w','sx','sw','bias'],output_names=['out'],header=HEADER,source=src)
 return _BATCH[key](inputs=[q,w,sx,sw,bias],template=[('M',m),('N',n),('K',kp),('MP',mpad),('NP',npad)],grid=(((n+127)//128)*threads,(m+bm-1)//bm,b),threadgroup=(threads,1,1),output_shapes=[(b,m,n)],output_dtypes=[mx.bfloat16])[0]
