import mlx.core as mx
from imagewam_mlx import action_kernel as ak
source="""
uint i=thread_position_in_grid.x*VEC;
uint n=residual_shape[2],total=residual_shape[1]*n;
if(i>=total)return;
float sums[VEC];
#pragma unroll
for(int e=0;e<VEC;e++)sums[e]=0;
#pragma unroll
for(int j=0;j<SPLITS;j++){
#pragma unroll
 for(int e=0;e<VEC;e++)sums[e]+=partial[j*total+i+e];
}
#pragma unroll
for(int e=0;e<VEC;e++){
 T z=T(sums[e]);volatile float vb=float(z)+float(bias[(i+e)%n]);z=T(vb);
 volatile float pg=float(z)*float(gate[(i+e)%n]);z=T(pg);
 volatile float result=float(residual[i+e])+float(z);out[i+e]=T(result);
}
"""
reduce=mx.fast.metal_kernel(name='wam_reduce_unrolled',input_names=['partial','residual','gate','bias'],output_names=['out'],source=source)
def fused(x,parts,residual,gate,bias,target=512,tile=(32,64,2,2),vec=1,tg=256):
 if residual.shape[-1]%vec:return ak.fused(x,parts,residual,gate,bias,target,tile)
 bm,bn,wm,wn=tile;threads=32*wm*wn
 m,n=residual.shape[1:];k=x.shape[-1];nt=(n+bn-1)//bn;mt=(m+bm-1)//bm;s=min(max(1,target//(nt*mt)),k//64)
 while s>1 and k%(s*64):s-=1
 p=ak.kernel(inputs=[x,*parts],template=[('T',x.dtype),('SPLITS',s),('BM',bm),('BN',bn),('WM',wm),('WN',wn)],grid=(nt*threads,mt,s),threadgroup=(threads,1,1),output_shapes=[(s,m,n)],output_dtypes=[mx.float32])[0]
 return reduce(inputs=[p,residual,gate,bias],template=[('T',x.dtype),('SPLITS',s),('VEC',vec)],grid=(residual.size//vec,1,1),threadgroup=(tg,1,1),output_shapes=[residual.shape],output_dtypes=[x.dtype])[0]
