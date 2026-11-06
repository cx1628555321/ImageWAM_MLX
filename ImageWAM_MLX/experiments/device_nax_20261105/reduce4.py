import mlx.core as mx
from imagewam_mlx.action_epilogue import FusedAction
from imagewam_mlx.action_kernel import kernel
reduce=mx.fast.metal_kernel(name='wam_split_epilogue_vector4',input_names=['partial','residual','gate','bias'],output_names=['out'],source='''
uint j4=thread_position_in_grid.x,i=j4*4;
uint n=residual_shape[2],total=residual_shape[1]*n;
if(i>=total)return;
float4 v=0;
for(int j=0;j<SPLITS;j++)v+=((const device float4*)partial)[j*total/4+j4];
vec<T,4> bv=((const device vec<T,4>*)bias)[(i%n)/4];
vec<T,4> gv=((const device vec<T,4>*)gate)[(i%n)/4];
vec<T,4> rv=((const device vec<T,4>*)residual)[j4];
vec<T,4> result;
for(int lane=0;lane<4;lane++){
 T z=T(v[lane]);volatile float vb=float(z)+float(bv[lane]);z=T(vb);
 volatile float pg=float(z)*float(gv[lane]);z=T(pg);
 volatile float r=float(rv[lane])+float(z);result[lane]=T(r);
}
((device vec<T,4>*)out)[j4]=result;
''')
def fused(x,parts,residual,gate,bias,target=512,tile=(32,64,2,2)):
 bm,bn,wm,wn=tile;threads=32*wm*wn
 m,n=residual.shape[1:];k=x.shape[-1];nt=(n+bn-1)//bn;mt=(m+bm-1)//bm;s=min(max(1,target//(nt*mt)),k//64)
 while s>1 and k%(s*64):s-=1
 assert n%4==0
 p=kernel(inputs=[x,*parts],template=[('T',x.dtype),('SPLITS',s),('BM',bm),('BN',bn),('WM',wm),('WN',wn)],grid=(nt*threads,mt,s),threadgroup=(threads,1,1),output_shapes=[(s,m,n)],output_dtypes=[mx.float32])[0]
 return reduce(inputs=[p,residual,gate,bias],template=[('T',x.dtype),('SPLITS',s)],grid=(residual.size//4,1,1),threadgroup=(256,1,1),output_shapes=[residual.shape],output_dtypes=[x.dtype])[0]
class Action4(FusedAction):
 def residual_linear(self,x,p,residual,gate):
  if self.kind=='w1' and p in self.packed and x.shape[0]==1 and x.shape[1]>13 and x.shape[-1]>4096 and residual.shape[-1]%4==0:
   n=residual.shape[-1];bias=self.w.get(p+'.bias',mx.zeros((n,),dtype=x.dtype))
   return fused(x,self.packed[p],residual,gate,bias,target=self.split_target,tile=((32,64,2,2) if self.split_adaptive and x.shape[1]>16 else self.split_tile))
  return super().residual_linear(x,p,residual,gate)
