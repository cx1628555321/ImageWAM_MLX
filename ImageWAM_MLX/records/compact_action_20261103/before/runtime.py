"""Validated default MLX runtime. Frozen BF16 weights, dynamic text and horizon.

Set IMAGEWAM_MLX_OPTIMIZATION=legacy or optimization='legacy' for original path.
No dependency on experimental scripts, source extraction, exec, or monkeypatching.
"""
import os
import time
import numpy as np
import mlx.core as mx
from .model import ImageWAM as LegacyImageWAM
from .ops import D,Weights,rms,flux_rope,silu,norm,attention

_kernel=mx.fast.metal_kernel(name='imagewam_strided_rope',input_names=['x','c','s'],output_names=['out'],ensure_row_contiguous=False,source=r'''
 uint j=thread_position_in_grid.x;
 uint pairs=x_shape[3]/2;
 uint total=x_shape[0]*x_shape[1]*x_shape[2]*pairs;
 if(j>=total)return;
 uint d=j%pairs, t=(j/pairs)%x_shape[2], h=(j/pairs/x_shape[2])%x_shape[1], b=j/pairs/x_shape[2]/x_shape[1];
 size_t off=b*x_strides[0]+h*x_strides[1]+t*x_strides[2]+2*d*x_strides[3];
 size_t ci=(b%c_shape[0])*c_strides[0]+(h%c_shape[1])*c_strides[1]+t*c_strides[2]+d*c_strides[3];
 size_t si=(b%s_shape[0])*s_strides[0]+(h%s_shape[1])*s_strides[1]+t*s_strides[2]+d*s_strides[3];
 float a=float(x[off]), v=float(x[off+x_strides[3]]);
 volatile float ca=c[ci]*a, sv=s[si]*v, cv=c[ci]*v, sa=s[si]*a;
 out[2*j]=T(ca-sv);out[2*j+1]=T(cv+sa);
''')
def fused_rotate(x,pe):
 c,s=pe
 return _kernel(inputs=[x,c,s],template=[('T',x.dtype)],grid=(x.size//2,1,1),threadgroup=(256,1,1),output_shapes=[x.shape],output_dtypes=[x.dtype])[0]
class RopeWeights(Weights):
 def qkv(self,x,p,pe):
  q,k,v=mx.split(x.reshape(x.shape[0],x.shape[1],3,24,128).transpose(2,0,3,1,4),3,axis=0);q,k,v=q[0],k[0],v[0]
  q=rms(q,self.w[p+'.query_norm.scale']);k=rms(k,self.w[p+'.key_norm.scale'])
  return fused_rotate(q,pe),fused_rotate(k,pe),v

class ImageWAM(LegacyImageWAM):
    def __init__(self,weights,source,optimization=None):
        mode=optimization or os.environ.get('IMAGEWAM_MLX_OPTIMIZATION','default')
        if mode not in ('default','legacy'):raise ValueError('optimization must be default or legacy')
        super().__init__(weights,source)
        self.optimization=mode
        if mode=='default':
            self.visual=RopeWeights(self.visual.w)
            self.action=RopeWeights(self.action.w)
            self.prefill=mx.compile(self.prefill)
            self._compiled_step=mx.compile(self._step)

    def _step(self,cache,x,t,delta,pe):
        a=self.action
        vec=a.embed_time(t)
        md=a.modulation(vec,'double_stream_modulation_img',True); ms=a.modulation(vec,'single_stream_modulation')
        z=a.linear(x,'action_encoder')
        for i in range(5):
            p=f'double_blocks.{i}'; q,k,v=a.double_prepare(z,p,md,pe)
            k=mx.concatenate([cache[i][0],k],axis=2); v=mx.concatenate([cache[i][1],v],axis=2)
            z=a.double_post(z,attention(q,k,v),p,md)
        for i in range(20):
            p=f'single_blocks.{i}'; q,k,v,mlp=a.single_prepare(z,p,ms,pe)
            k=mx.concatenate([cache[i+5][0],k],axis=2); v=mx.concatenate([cache[i+5][1],v],axis=2)
            z=a.single_post(z,attention(q,k,v),mlp,p,ms)
        shift,scale=mx.split(a.linear(silu(vec),'head.adaLN_modulation.1'),2,axis=-1)
        pred=a.linear((1+scale[:,None])*norm(z)+shift[:,None],'head.linear')
        x=x+pred*delta
        return x

    def denoise(self,cache,noise,steps=10,trace=False):
        if self.optimization=='legacy':return super().denoise(cache,noise,steps,trace)
        if noise.ndim!=3 or noise.shape[0]!=1 or noise.shape[2]!=14 or not 1<=noise.shape[1]<=64 or not isinstance(steps,int) or steps<1:
            raise ValueError('noise must be [1,T<=64,14]; steps must be a positive integer')
        x=mx.array(noise).astype(D)
        u=mx.linspace(1,0,steps+1,dtype=mx.float32);sig=5*u/(1+4*u)
        times=(sig[:-1]*1000).astype(D);deltas=(sig[1:]-sig[:-1]).astype(D)
        ids=np.zeros((1,x.shape[1],4),np.float32);ids[...,0]=2;ids[0,:,1]=np.arange(x.shape[1])
        pe=flux_rope(mx.array(ids).astype(D))
        for i in range(steps):
            t=(times[i:i+1]/1000).astype(D)
            x=self._compiled_step(cache,x,t,deltas[i],pe)
            mx.async_eval(x)
            if trace:self.trace[f'action_step_{i}']=np.array(x.astype(mx.float32))
        return x

    def predict(self,images,state,prompt,noise,steps=10,profile=False,trace=False):
        if self.optimization=='legacy':return super().predict(images,state,prompt,noise,steps,profile,trace)
        self.trace={}; self.profile={}
        def run(name,fn):
            if profile:mx.synchronize()
            start=time.perf_counter(); y=fn()
            if profile:mx.eval(y); mx.synchronize(); self.profile[name]=(time.perf_counter()-start)*1000
            return y
        x,st=run('preprocess',lambda:self.preprocess(images,state))
        start=time.perf_counter(); ids=self.tokenize(prompt)
        if profile:self.profile['tokenization']=(time.perf_counter()-start)*1000
        hidden=run('text_encoder',lambda:self.qwen(mx.array(ids)))
        latent=run('image_encoder',lambda:self.vae(x))
        cache=run('visual_prefill',lambda:self.prefill(hidden,latent,st))
        # Prefix is evaluated through the action dependency; no forced boundary.
        out=run('action_denoise',lambda:self.denoise(cache,noise,steps,trace))
        start=time.perf_counter()
        normalized=np.array(out[0].astype(mx.float32))
        scale,offset=self.norms['action']; result=(normalized-offset)/scale
        if profile:self.profile['postprocess']=(time.perf_counter()-start)*1000
        if trace:
            self.trace.update(image=np.array(x.astype(mx.float32)),state=np.array(st.astype(mx.float32)),text=np.array(hidden.astype(mx.float32)),latent=np.array(latent.astype(mx.float32)),token_ids=ids)
            for i in (0,4,5,24):self.trace[f'cache_k_{i}']=np.array(cache[i][0].astype(mx.float32))
        return result,normalized
