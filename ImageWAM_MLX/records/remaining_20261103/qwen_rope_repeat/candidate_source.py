"""Candidates against compact + fused VAE. No cross-request feature caching."""
import copy,types,time
import numpy as np
import mlx.core as mx
from imagewam_mlx.runtime import ImageWAM
from imagewam_mlx.fused_vae import FusedVAE
from imagewam_mlx.ops import rms,silu,attention
from imagewam_mlx.qwen import Qwen

_inv=mx.fast.metal_kernel(name='wam_group_inv_reuse',input_names=['x','mean','inv','w','bias'],output_names=['out'],ensure_row_contiguous=False,source=r'''
 uint i=thread_position_in_grid.x;if(i>=x_shape[0]*x_shape[1]*x_shape[2]*x_shape[3])return;
 uint c=i%x_shape[3],b=i/(x_shape[1]*x_shape[2]*x_shape[3]);uint g=c/(x_shape[3]/32);
 volatile float centered=float(x[i])-mean[b*32+g];
 volatile float normalized=centered*inv[b*32+g];
 volatile float scaled=normalized*float(w[c]);T z=T(scaled+float(bias[c]));
 auto sy=1/(1+metal::precise::exp(metal::abs(z)));T sig=(z<0)?sy:1-sy;
 out[i]=T(float(z)*float(sig));
''')
class GroupVAE(FusedVAE):
 def gn_silu(self,x,p):
  b,h,w,c=x.shape;z=x.astype(mx.float32).reshape(b,h,w,32,c//32)
  mean=mx.mean(z,axis=(1,2,4),keepdims=True)
  var=mx.mean((z-mean)**2,axis=(1,2,4),keepdims=True)
  inv=mx.rsqrt(var+1e-6)
  return _inv(inputs=[mx.contiguous(x),mean,inv,self.w[p+'.weight'],self.w[p+'.bias']],template=[('T',x.dtype)],grid=(x.size,1,1),threadgroup=(256,1,1),output_shapes=[x.shape],output_dtypes=[x.dtype])[0]

_qrope=mx.fast.metal_kernel(name='wam_qwen_rope_exact',input_names=['x','c','s'],output_names=['out'],ensure_row_contiguous=False,source=r'''
 uint i=thread_position_in_grid.x;if(i>=x_shape[0]*x_shape[1]*x_shape[2]*128)return;
 uint d=i%128,t=(i/128)%x_shape[2],h=(i/128/x_shape[2])%x_shape[1],b=i/(128*x_shape[2]*x_shape[1]);
 size_t base=b*x_strides[0]+h*x_strides[1]+t*x_strides[2];
 T a=x[base+d*x_strides[3]],v=x[base+((d+64)%128)*x_strides[3]];
 if(d<64)v=T(-float(v));
 T left=T(float(a)*float(c[t*c_strides[2]+d*c_strides[3]]));
 T right=T(float(v)*float(s[t*s_strides[2]+d*s_strides[3]]));
 out[i]=T(float(left)+float(right));
''')
def qrope(x,c,s):
 return _qrope(inputs=[x,c,s],template=[('T',x.dtype)],grid=(x.size,1,1),threadgroup=(256,1,1),output_shapes=[x.shape],output_dtypes=[x.dtype])[0]

class RopeQwen(Qwen):
    def __call__(self,ids):
        x=self.w['embed_tokens.weight'][ids]
        n=ids.shape[1]; hidden=[]
        freq=1/(1000000**(mx.arange(0,128,2,dtype=mx.float32)/128))
        a=mx.arange(n,dtype=mx.float32)[:,None]*freq[None]
        c=mx.concatenate([mx.cos(a),mx.cos(a)],axis=-1).astype(x.dtype)[None,None]
        s=mx.concatenate([mx.sin(a),mx.sin(a)],axis=-1).astype(x.dtype)[None,None]
        def rope(z):
            return qrope(z,c,s)
        for i in range(27):
            p=f'layers.{i}'; y=rms(x,self.w[p+'.input_layernorm.weight'])
            q=self.linear(y,p+'.self_attn.q_proj').reshape(1,n,32,128).transpose(0,2,1,3)
            k=self.linear(y,p+'.self_attn.k_proj').reshape(1,n,8,128).transpose(0,2,1,3)
            v=self.linear(y,p+'.self_attn.v_proj').reshape(1,n,8,128).transpose(0,2,1,3)
            q=rope(rms(q,self.w[p+'.self_attn.q_norm.weight']))
            k=rope(rms(k,self.w[p+'.self_attn.k_norm.weight']))
            x=x+self.linear(attention(q,k,v,'causal'),p+'.self_attn.o_proj')
            y=rms(x,self.w[p+'.post_attention_layernorm.weight'])
            x=x+self.linear(silu(self.linear(y,p+'.mlp.gate_proj'))*self.linear(y,p+'.mlp.up_proj'),p+'.mlp.down_proj')
            if i+1 in (9,18,27):hidden.append(x)
        return mx.concatenate(hidden,axis=-1)

class OverlapModel(ImageWAM):
    def predict(self,images,state,prompt,noise,steps=10,profile=False,trace=False):
        if self.optimization=='legacy':return super().predict(images,state,prompt,noise,steps,profile,trace)
        self.trace={}; self.profile={}
        def run(name,fn):
            if profile:mx.synchronize()
            start=time.perf_counter(); y=fn()
            if profile:mx.eval(y); mx.synchronize(); self.profile[name]=(time.perf_counter()-start)*1000
            return y
        start=time.perf_counter(); ids=self.tokenize(prompt)
        if profile:self.profile['tokenization']=(time.perf_counter()-start)*1000
        hidden=run('text_encoder',lambda:self.qwen(mx.array(ids)))
        mx.async_eval(hidden)
        x,st=run('preprocess',lambda:self.preprocess(images,state))
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


def build(base,mode):
 m=copy.copy(base)
 if mode=='overlap':m.predict=types.MethodType(OverlapModel.predict,m)
 elif mode=='group_inv':m.vae=GroupVAE.from_vae(base.vae)
 elif mode=='qwen_rope':
  q=RopeQwen.__new__(RopeQwen);q.w=base.qwen.w;m.qwen=q
 return m
BUILDERS={name:(lambda b,n=name:build(b,n)) for name in ('overlap','group_inv','qwen_rope')}

def combined(base):
    # Bind predict last, to the object containing both modified encoders.
    return build(build(build(base,'group_inv'),'qwen_rope'),'overlap')
BUILDERS['combined']=combined
