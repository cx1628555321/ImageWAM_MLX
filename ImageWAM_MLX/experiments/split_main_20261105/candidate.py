import mlx.core as mx
from imagewam_mlx import action_kernel as ak
# Modify only split_qmm, not shared loader or ordinary qmm.
head,tail=ak.header.split('METAL_FUNC void split_qmm(',1)
kernels={}
for mode in ('unroll','edges','combined'):
 t=tail
 if mode in ('unroll','combined'):t=t.replace('STEEL_PRAGMA_NO_UNROLL','STEEL_PRAGMA_UNROLL')
 if mode in ('edges','combined'):
  t=t.replace('for (int k = 0; k < K/splits; k += BK) {\n        threadgroup_barrier(mem_flags::mem_threadgroup);','for (int k = 0; k < K/splits; k += BK) {\n        if (k > 0) threadgroup_barrier(mem_flags::mem_threadgroup);')
  t=t.replace('// Store results to device memory\n      threadgroup_barrier(mem_flags::mem_threadgroup);','// Accumulator stores use registers, no remaining shared-memory reads.')
 kernels[mode]=mx.fast.metal_kernel(name='wam_split_main_'+mode,input_names=['x','w','scales','qb'],output_names=['out'],header=head+'METAL_FUNC void split_qmm('+t,source="""
threadgroup T Ws[BN*72];
int M=x_shape[1],N=w_shape[0],K=x_shape[2];
split_qmm<T,64,1,true,BM,64,BN,WM,WN>(w,scales,qb,x,out,SPLITS,Ws,K,N,M,threadgroup_position_in_grid,thread_index_in_threadgroup,simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
""")
def fused(x,parts,residual,gate,bias,target=512,tile=(32,64,2,2),mode='unroll'):
 bm,bn,wm,wn=tile;threads=32*wm*wn
 m,n=residual.shape[1:];k=x.shape[-1];nt=(n+bn-1)//bn;mt=(m+bm-1)//bm;s=min(max(1,target//(nt*mt)),k//64)
 while s>1 and k%(s*64):s-=1
 p=kernels[mode](inputs=[x,*parts],template=[('T',x.dtype),('SPLITS',s),('BM',bm),('BN',bn),('WM',wm),('WN',wn)],grid=(nt*threads,mt,s),threadgroup=(threads,1,1),output_shapes=[(s,m,n)],output_dtypes=[mx.float32])[0]
 return ak.reduce(inputs=[p,residual,gate,bias],template=[('T',x.dtype),('SPLITS',s)],grid=(residual.size,1,1),threadgroup=(256,1,1),output_shapes=[residual.shape],output_dtypes=[x.dtype])[0]
