"""Per-request functional KV workspace. No cross-request feature cache.
MLX decides buffer reuse: slice_update is not a promise of in-place storage.
"""
import numpy as np
import mlx.core as mx
from imagewam_mlx.ops import D,attention,silu,norm,flux_rope

def workspace_step(self,cache,x,t,delta,pe):
    updated=[]
    a=self.action
    vec=a.embed_time(t)
    md=a.modulation(vec,'double_stream_modulation_img',True); ms=a.modulation(vec,'single_stream_modulation')
    z=a.linear(x,'action_encoder')
    for i in range(5):
        p=f'double_blocks.{i}'; q,k,v=a.double_prepare(z,p,md,pe)
        offset=cache[i][0].shape[2]-k.shape[2]
        k=mx.slice_update(cache[i][0],k,mx.array([offset]),axes=(2,)); v=mx.slice_update(cache[i][1],v,mx.array([offset]),axes=(2,))
        updated.append((k,v))
        z=a.double_post(z,attention(q,k,v),p,md)
    for i in range(20):
        p=f'single_blocks.{i}'; q,k,v,mlp=a.single_prepare(z,p,ms,pe)
        offset=cache[i+5][0].shape[2]-k.shape[2]
        k=mx.slice_update(cache[i+5][0],k,mx.array([offset]),axes=(2,)); v=mx.slice_update(cache[i+5][1],v,mx.array([offset]),axes=(2,))
        updated.append((k,v))
        z=a.single_post(z,attention(q,k,v),mlp,p,ms)
    shift,scale=mx.split(a.linear(silu(vec),'head.adaLN_modulation.1'),2,axis=-1)
    pred=a.linear((1+scale[:,None])*norm(z)+shift[:,None],'head.linear')
    x=x+pred*delta
    return x,updated

def workspace_denoise(self,cache,noise,steps=10,trace=False):
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
        x,_updated=self._compiled_workspace_step(cache,x,t,deltas[i],pe)
        mx.async_eval(x)
        if trace:self.trace[f'action_step_{i}']=np.array(x.astype(mx.float32))
    return x

