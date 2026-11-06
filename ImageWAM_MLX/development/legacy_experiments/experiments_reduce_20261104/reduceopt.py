import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'development/legacy_experiments/experiments_split_tune_20261104'))
import mlx.core as mx
from tiled import kernel
source='''
uint col=thread_position_in_grid.x*VEC,row=thread_position_in_grid.y;
uint n=residual_shape[2],total=residual_shape[1]*n;
if(col>=n)return;
uint i=row*n+col;
if constexpr(VEC==4){
 float4 sum=0;
 for(int j=0;j<SPLITS;j++)sum+=reinterpret_cast<const device float4*>(partial)[(j*total+i)/4];
 for(int e=0;e<4;e++){
  T z=T(sum[e]);volatile float vb=float(z)+float(bias[col+e]);z=T(vb);
  volatile float pg=float(z)*float(gate[col+e]);z=T(pg);
  volatile float result=float(residual[i+e])+float(z);out[i+e]=T(result);
 }
}else{
 float v=0;for(int j=0;j<SPLITS;j++)v+=partial[j*total+i];
 T z=T(v);volatile float vb=float(z)+float(bias[col]);z=T(vb);
 volatile float pg=float(z)*float(gate[col]);z=T(pg);
 volatile float result=float(residual[i])+float(z);out[i]=T(result);
}
'''
reduce=mx.fast.metal_kernel(name='imagewam_reduce_layout',input_names=['partial','residual','gate','bias'],output_names=['out'],source=source)
def fused(x,parts,residual,gate,bias,target=512,tile=(32,64,2,2),mode=4):
 bm,bn,wm,wn=tile;threads=32*wm*wn
 m,n=residual.shape[1:];k=x.shape[-1];nt=(n+bn-1)//bn;mt=(m+bm-1)//bm;s=min(max(1,target//(nt*mt)),k//64)
 while s>1 and k%(s*64):s-=1
 p=kernel(inputs=[x,*parts],template=[('T',x.dtype),('SPLITS',s),('BM',bm),('BN',bn),('WM',wm),('WN',wn)],grid=(nt*threads,mt,s),threadgroup=(threads,1,1),output_shapes=[(s,m,n)],output_dtypes=[mx.float32])[0]
 return reduce(inputs=[p,residual,gate,bias],template=[('T',x.dtype),('SPLITS',s),('VEC',mode)],grid=(n//mode,m,1),threadgroup=(128,1,1),output_shapes=[residual.shape],output_dtypes=[x.dtype])[0]
