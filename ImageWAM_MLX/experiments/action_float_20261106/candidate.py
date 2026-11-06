from pathlib import Path
import mlx.core as mx
import types
from imagewam_mlx import workspace as ws
from imagewam_mlx.ops import time_embedding,silu
from candidates import install as original_install
from imagewam_mlx.runtime import RopeWeights
from imagewam_mlx.workspace import workspace_step

def install(m,mode,raw=None,scales=None):
 if mode=='baseline':return original_install(m,'native_w1')
 if mode in ('bf16','fp16'):
  a=RopeWeights(mx.load(str(Path(m.weights_path)/'action.safetensors')))
 elif mode=='expanded_w1':
  original_install(m,'native_w1');q=m.action;w=dict(q.w)
  for name,parts in q.packed.items():
   w[name+'.weight']=mx.dequantize(*parts,group_size=q.group_size,bits=q.bits,mode=q.mode).astype(mx.bfloat16)
  mx.eval(w);a=RopeWeights(w)
 else:raise ValueError(mode)
 if mode=='fp16':
  a.w={k:v.astype(mx.float16) for k,v in a.w.items()}
  def embed_time(self,t):
   return self.linear(silu(self.linear(time_embedding(t).astype(mx.float16),'time_in.in_layer')),'time_in.out_layer')
  a.embed_time=types.MethodType(embed_time,a)
  glob=dict(ws.workspace_denoise.__globals__);glob['D']=mx.float16
  denoise=types.FunctionType(ws.workspace_denoise.__code__,glob)
  def run(self,cache,noise,steps=10,trace=False):
   cache=[(k.astype(mx.float16),v.astype(mx.float16)) for k,v in cache]
   return denoise(self,cache,noise,steps,trace)
  m.denoise=types.MethodType(run,m)
 mx.eval(a.w);m.action=a
 m._compiled_workspace_step=mx.compile(lambda c,x,t,d,p:workspace_step(m,c,x,t,d,p))
 return [dict(action_kind=mode,weight_bytes=sum(v.nbytes for v in a.w.values()),explicit_dequantize_in_forward=False)]
