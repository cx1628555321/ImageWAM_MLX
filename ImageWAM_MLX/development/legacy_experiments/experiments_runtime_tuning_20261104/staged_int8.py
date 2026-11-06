"""Experimental PR #4722 mode-2 kernels via MLX runtime JIT, not a patched wheel."""
from pathlib import Path
import mlx.core as mx
base=Path(__file__).parent
src=(base/'upstream_int8/mlx/backend/metal/kernels/quantized_nax.h').read_text()
header=(Path('imagewam_mlx/metal/epilogue.h')).read_text()
helper=src[src.index('// Fixed point bits'):src.index('// The scale of each row')]
header+= '\n'+helper

def body(name):
 start=src.index('{',src.index('[[kernel]] void '+name));depth=1;i=start+1
 while depth:
  depth+=(src[i]=='{')-(src[i]=='}');i+=1
 return src[start+1:i-1]
rs=mx.fast.metal_kernel(name='wam_pr4722_scales',input_names=['scales','biases'],output_names=['w_scale','w_inv'],header=header,source='uint tid=thread_position_in_grid.x; if(tid>=scales_shape[0])return; const int K_g=scales_shape[1];\n'+body('affine_qmm_i8_row_scales'))
qx=mx.fast.metal_kernel(name='wam_pr4722_x8',input_names=['x'],output_names=['x8','x_scale','x_exp'],header=header,source='uint3 tid=threadgroup_position_in_grid;uint lid=thread_index_in_threadgroup,simd_gid=simdgroup_index_in_threadgroup,simd_lid=thread_index_in_simdgroup,n_simd=simdgroups_per_threadgroup;const int K=x_shape[1],M=x_shape[0],max_exp=0,x_span=0;\n'+body('affine_qmm_i8_quantize_x'))
mm=mx.fast.metal_kernel(name='wam_pr4722_mm8',input_names=['w','scales','biases','w_inv','w_scale','x8','x_scale','x_exp'],output_names=['y'],header=header,source='uint3 tid=threadgroup_position_in_grid;uint lid=thread_index_in_threadgroup,simd_gid=simdgroup_index_in_threadgroup;const int K=XK,N=w_shape[0],M=ROWS;\n'+body('affine_qmm_t_nax_i8'))
def prepare(parts):
 w,s,b=parts;n=w.shape[0]
 ws,wi=rs(inputs=[s,b],template=[('T',s.dtype),('bits',4)],grid=(n,1,1),threadgroup=(min(256,n),1,1),output_shapes=[(n,),(n,)],output_dtypes=[mx.float32,mx.float32]);mx.eval(ws,wi)
 return w,s,b,wi,ws

def quantize(x):
 m,k=x.shape;mp=(m+63)//64*64;nb=k//64;threads=(nb+31)//32*32
 assert k%64==0 and nb<=1024
 return qx(inputs=[x],template=[('T',x.dtype)],grid=(threads,mp,1),threadgroup=(threads,1,1),output_shapes=[(mp,k),(m,),(mp,nb+1)],output_dtypes=[mx.int8,mx.float32,mx.uint8])

def project(x,parts):
 shape=x.shape;x=x.reshape(-1,shape[-1]);m,k=x.shape;n=parts[0].shape[0];a,s,e=quantize(x)
 y=mm(inputs=[*parts,a,s,e],template=[('T',x.dtype),('group_size',64),('bits',4),('x_span',0),('XK',k),('ROWS',m)],grid=(((n+63)//64)*128,(m+63)//64,1),threadgroup=(128,1,1),output_shapes=[(m,n)],output_dtypes=[x.dtype])[0]
 return y.reshape(*shape[:-1],n)
