import numpy as np
import mlx.core as mx
from imagewam_mlx.ops import D,flux_rope
from candidate import conditions

def prepare(self,cache,noise,u,hoist):
 x=noise.astype(D)
 sig=5*u/(1+4*u)
 times=(sig[:-1]*1000).astype(D);deltas=(sig[1:]-sig[:-1]).astype(D)
 h=x.shape[1]
 ids=mx.stack([mx.full((h,),2,dtype=D),mx.arange(h,dtype=D),mx.zeros((h,),dtype=D),mx.zeros((h,),dtype=D)],axis=-1)[None]
 pe=flux_rope(ids)
 cache=[(mx.pad(k,[(0,0),(0,0),(0,h),(0,0)]),mx.pad(v,[(0,0),(0,0),(0,h),(0,0)])) for k,v in cache]
 return x,cache,times,deltas,pe,conditions(self,times) if hoist else ()

def denoise(self,cache,noise,steps=10,trace=False):
 if noise.ndim!=3 or noise.shape[0]!=1 or noise.shape[2]!=14 or not 1<=noise.shape[1]<=64 or not isinstance(steps,int) or steps<1:
  raise ValueError('noise must be [1,T<=64,14]; steps must be a positive integer')
 x,cache,times,deltas,pe,c=self._prepared_request(cache,mx.array(noise),mx.linspace(1,0,steps+1,dtype=mx.float32))
 for i in range(steps):
  if self._hoist:
   x,cache=self._condition_step(cache,x,c[i],deltas[i],pe)
  else:
   t=(times[i:i+1]/1000).astype(D)
   x,cache=self._compiled_workspace_step(cache,x,t,deltas[i],pe)
  mx.async_eval(x,cache)
  if trace:self.trace[f'action_step_{i}']=np.array(x.astype(mx.float32))
 return x
