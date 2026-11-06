import numpy as np
import mlx.core as mx
from imagewam_mlx.ops import D,flux_rope
def denoise(self,cache,noise,steps=10,trace=False):
    if noise.ndim!=3 or noise.shape[0]!=1 or noise.shape[2]!=14 or not 1<=noise.shape[1]<=64 or not isinstance(steps,int) or steps<1:
        raise ValueError('noise must be [1,T<=64,14]; steps must be a positive integer')
    x=mx.array(noise).astype(D)
    u=mx.linspace(1,0,steps+1,dtype=mx.float32);sig=5*u/(1+4*u)
    times=(sig[:-1]*1000).astype(D);deltas=(sig[1:]-sig[:-1]).astype(D)
    ids=np.zeros((1,x.shape[1],4),np.float32);ids[...,0]=2;ids[0,:,1]=np.arange(x.shape[1])
    pe=flux_rope(mx.array(ids).astype(D))
    cache=[(mx.pad(k,[(0,0),(0,0),(0,x.shape[1]),(0,0)]),mx.pad(v,[(0,0),(0,0),(0,x.shape[1]),(0,0)])) for k,v in cache]
    for i in range(steps):
        t=(times[i:i+1]/1000).astype(D)
        x,cache=self._compiled_workspace_step(cache,x,t,deltas[i],pe)
        if (i+1)%self.submit_interval==0 or i+1==steps:
            mx.async_eval(x)
        if trace:self.trace[f'action_step_{i}']=np.array(x.astype(mx.float32))
    return x
