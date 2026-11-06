from pathlib import Path
import mlx.core as mx
from imagewam_mlx.runtime import RopeWeights
from imagewam_mlx.ops import gated
header=(Path(__file__).parent/'gemm.h').read_text()
kernel=mx.fast.metal_kernel(name='visual_bf16_nax_epilogue',input_names=['x','w','bias','residual','gate'],output_names=['out'],header=header,source=r'''
using namespace mlx::steel;
constexpr short SM=BM/WM,SN=BN/WN,TM=SM/16,TN=SN/16;
static_assert(TM%2==0 || TN%2==0,"unsupported 1x1 fragment MMA");
int M=x_shape[1],K=x_shape[2],N=w_shape[0];
int row=threadgroup_position_in_grid.y*BM+(simdgroup_index_in_threadgroup/WN)*SM;
int col=threadgroup_position_in_grid.x*BN+(simdgroup_index_in_threadgroup%WN)*SN;
short rm=short(min(int(SM),M-row)),rn=short(min(int(SN),N-col));
NAXTile<float,TM,TN> tile;
dispatch_bool(rm==SM,[&](auto aligned){
 tile=gemm_loop<T,SM,SN,32,BK,false,true,aligned.value,true,true,float>(x+long(row)*K,w+long(col)*K,K,K,K,K/BK,rm,rn);
});
if(rm<=0)return;
for(short i=0;i<TM;i++) for(short j=0;j<TN;j++) for(short e=0;e<8;e++) {
 short2 c=BaseNAXFrag::get_coord(e);int r=row+i*16+c.y,n=col+j*16+c.x;
 if(r<M && n<N){
 T z=T(tile.frag_at(i,j)[e]);volatile float b=float(z)+float(bias[n]);z=T(b);
 if(FUSED){volatile float g=float(z)*float(gate[n]);z=T(g);volatile float v=float(residual[long(r)*N+n])+float(z);z=T(v);}
 out[long(r)*N+n]=z;
 }
}
''')
def project(x,w,bias,residual,gate,tile=(64,64,2,2,32),fused=True):
 bm,bn,wm,wn,bk=tile;m=x.shape[1];n=w.shape[0];assert n%bn==0 and x.shape[-1]%bk==0
 return kernel(inputs=[x,w,bias,residual,gate],template=[('T',x.dtype),('BM',bm),('BN',bn),('WM',wm),('WN',wn),('BK',bk),('FUSED',fused)],grid=((n//bn)*wm*wn*32,(m+bm-1)//bm,1),threadgroup=(wm*wn*32,1,1),output_shapes=[(1,m,n)],output_dtypes=[x.dtype])[0]
class Visual(RopeWeights):
 def single_post(self,x,att,mlp,p,m):
  y=mx.concatenate([att,gated(mlp)],axis=-1)
  return project(y,self.w[p+'.linear2.weight'],self.w.get(p+'.linear2.bias',mx.zeros((x.shape[-1],),dtype=x.dtype)),x,m[2],self.tile)
class InputVisual(RopeWeights):
 def linear(self,x,p):
  if p.startswith('single_blocks.') and p.endswith('.linear1'):
   w=self.w[p+'.weight'];b=self.w.get(p+'.bias',mx.zeros((w.shape[0],),dtype=x.dtype));dummy=mx.zeros((1,),dtype=x.dtype)
   return project(x,w,b,dummy,dummy,self.tile,fused=False)
  return super().linear(x,p)
