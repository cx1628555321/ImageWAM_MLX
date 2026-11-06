import copy
import mlx.core as mx
from imagewam_mlx.quantized import PackedRopeWeights
from imagewam_mlx.ops import gated,norm
from imagewam_mlx.workspace import workspace_step
from split import fused
class Epilogue(PackedRopeWeights):
 def residual_linear(self,x,p,residual,gate):
  # Preserve the official vector-dispatch range; only target large-K NAX shapes.
  if self.kind=='w1' and p in self.packed and x.shape[0]==1 and x.shape[1]>13 and x.shape[-1]>4096:
   n=residual.shape[-1];bias=self.w.get(p+'.bias',mx.zeros((n,),dtype=x.dtype))
   return fused(x,self.packed[p],residual,gate,bias,target=self.split_target)
  return residual+gate*self.linear(x,p)
 def double_post(self,x,att,p,m,stream='img'):
  x=self.residual_linear(att,p+'.'+stream+'_attn.proj',x,m[2])
  z=(1+m[4])*norm(x)+m[3]
  return self.residual_linear(gated(self.linear(z,p+'.'+stream+'_mlp.0')),p+'.'+stream+'_mlp.2',x,m[5])
 def single_post(self,x,att,mlp,p,m):
  return self.residual_linear(mx.concatenate([att,gated(mlp)],axis=-1),p+'.linear2',x,m[2])
def install(m,target=512):
 c=copy.copy(m);c.action=copy.copy(m.action);c.action.__class__=Epilogue;c.action.split_target=target
 c._compiled_workspace_step=mx.compile(lambda cache,x,t,delta,pe:workspace_step(c,cache,x,t,delta,pe))
 return c
