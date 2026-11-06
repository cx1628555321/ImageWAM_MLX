import copy,types
import mlx.core as mx
from imagewam_mlx.workspace import workspace_step
from imagewam_mlx.model import ImageWAM as Base

def fused(q,k,v,mask=None):
 y=mx.fast.scaled_dot_product_attention(q,k,v,scale=q.shape[-1]**-0.5,mask=mask,force_fused=True)
 return y.transpose(0,2,1,3).reshape(y.shape[0],y.shape[2],-1)

def install(base,mode):
 m=copy.copy(base)
 if mode in ('attention_action','attention_both'):
  step=types.FunctionType(workspace_step.__code__,dict(workspace_step.__globals__,attention=fused),workspace_step.__name__,workspace_step.__defaults__)
  m._compiled_workspace_step=mx.compile(lambda cache,x,t,d,pe:step(m,cache,x,t,d,pe))
 if mode in ('attention_visual','attention_both'):
  pf=types.FunctionType(Base.prefill.__code__,dict(Base.prefill.__globals__,attention=fused),Base.prefill.__name__,Base.prefill.__defaults__)
  m.prefill=mx.compile(lambda h,l,s:pf(m,h,l,s))
 return m
