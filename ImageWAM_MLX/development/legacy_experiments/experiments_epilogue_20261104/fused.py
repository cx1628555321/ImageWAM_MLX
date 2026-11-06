from pathlib import Path
import mlx.core as mx
kernel=mx.fast.metal_kernel(name='imagewam_w1_epilogue',input_names=['x','w','scales','qb','residual','gate','bias'],output_names=['out'],header=Path(__file__).with_name('epilogue.h').read_text(),source='''
threadgroup T Ws[64*72];
int M=residual_shape[1],N=residual_shape[2],K=x_shape[2];
epilogue_qmm<T,64,1,true,32,64,64,2,2>(w,scales,qb,x,out,residual,gate,bias,Ws,K,N,M,threadgroup_position_in_grid,thread_index_in_threadgroup,simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
''')
def fused(x,parts,residual,gate,bias):
 m,n=residual.shape[1:];return kernel(inputs=[x,*parts,residual,gate,bias],template=[('T',x.dtype)],grid=((n//64)*128,(m+31)//32,1),threadgroup=(128,1,1),output_shapes=[residual.shape],output_dtypes=[x.dtype])[0]
