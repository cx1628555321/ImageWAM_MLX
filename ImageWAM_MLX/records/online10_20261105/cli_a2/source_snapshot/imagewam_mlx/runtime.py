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
 def rotate_key(self,x,pe):return fused_rotate(x,pe)
 def qkv(self,x,p,pe):
  q,k,v=mx.split(x.reshape(x.shape[0],x.shape[1],3,24,128).transpose(2,0,3,1,4),3,axis=0);q,k,v=q[0],k[0],v[0]
  q=rms(q,self.w[p+'.query_norm.scale']);k=rms(k,self.w[p+'.key_norm.scale'])
  return fused_rotate(q,pe),fused_rotate(k,pe),v

class ImageWAM(LegacyImageWAM):
    def __init__(self,weights,source,optimization=None,compact=True,fused_vae=True,overlap=True,vae_group_inv=False,qwen_rope=False,vae_tail_compile=False,vae_pow_mode='original',profile='accuracy',quantization=None,speed_options=None,kv_workspace=None,action_epilogue=None,staged_visual=None,prepacked_visual=None,fused_concat=None,fused_gate=None,tail_visual=None,text_cache_size=0,vae_asymmetric=None):
        if not isinstance(text_cache_size,int) or text_cache_size<0:raise ValueError('text_cache_size must be a nonnegative integer')
        if text_cache_size and profile!='speed':raise ValueError('Text cache currently validated only for speed profile')
        self.text_cache_size=text_cache_size
        mode=optimization or os.environ.get('IMAGEWAM_MLX_OPTIMIZATION','default')
        if mode not in ('default','legacy'):raise ValueError('optimization must be default or legacy')
        if profile not in ('accuracy','speed'):raise ValueError('profile must be accuracy or speed')
        if profile=='accuracy' and (quantization is not None or speed_options is not None):raise ValueError('Speed configuration requires profile=speed')
        if profile=='speed' and mode=='legacy':raise ValueError('Speed profile requires default optimization')
        if vae_pow_mode not in ('original','fused','lookup'):raise ValueError('vae_pow_mode must be original, fused or lookup')
        super().__init__(weights,source,compact=compact and mode!='legacy')
        self.optimization=mode
        self.kv_workspace=(profile=='speed' if kv_workspace is None else bool(kv_workspace)) and mode=='default'
        self.inference_profile=profile
        self.staged_visual=False
        self.staged_visual_reason='disabled'
        if staged_visual and profile!='speed':raise ValueError('Staged visual requires profile=speed')
        self.action_epilogue=False
        self.action_epilogue_reason='disabled'
        if action_epilogue and profile!='speed':raise ValueError('Action epilogue requires profile=speed')
        self.quantization_info={}
        self.speed_options={}
        self.fused_vae=bool(fused_vae and mode=='default')
        self.overlap=bool(overlap and mode=='default')
        self.vae_group_inv=bool(vae_group_inv and self.fused_vae)
        self.qwen_rope=bool(qwen_rope and mode=='default')
        self.vae_asymmetric=False
        self.vae_pow_mode=vae_pow_mode if self.fused_vae and not self.vae_group_inv else 'original'
        self.vae_tail_compile=bool(vae_tail_compile and self.fused_vae and not self.vae_group_inv and self.vae_pow_mode=='original')
        if self.fused_vae:
            if self.vae_group_inv:
                from .fused_vae_inv import GroupInverseVAE
                self.vae=GroupInverseVAE.from_vae(self.vae)
            else:
                from .fused_vae import FusedVAE
                self.vae=FusedVAE.from_vae(self.vae)
        if self.vae_tail_compile:
            from .compiled_vae import CompiledResidualVAE
            self.vae=CompiledResidualVAE.from_vae(self.vae)
        if self.vae_pow_mode!='original':
            from .pow_vae import PowVAE,LookupPowVAE
            self.vae=(PowVAE if self.vae_pow_mode=='fused' else LookupPowVAE).from_vae(self.vae)
        if self.qwen_rope:
            from .fused_qwen_rope import FusedRopeQwen
            self.qwen=FusedRopeQwen.from_qwen(self.qwen)
        if mode=='default':
            self.visual=RopeWeights(self.visual.w)
            self.action=RopeWeights(self.action.w)
            if profile=='speed':
                from .quantized import SPEED_POLICY,SPEED_OPTIONS,configure
                opts=dict(SPEED_OPTIONS if speed_options is None else speed_options)
                if set(opts)-{'qwen_fused','vae'}:raise ValueError('Unknown speed option')
                if opts.get('vae','original') not in ('original','fast_bf16','fast_fp16'):raise ValueError('Invalid speed VAE')
                self.speed_options=opts
                self.quantization_info=configure(self,SPEED_POLICY if quantization is None else quantization,qwen_fused=opts.get('qwen_fused',False))
                if opts.get('vae','original')!='original':
                    from .speed_vae import SpeedVAE
                    from importlib.metadata import version
                    supported_asym=(opts['vae']=='fast_bf16' and mx.device_info().get('architecture')=='applegpu_g17s' and version('mlx')=='0.32.4')
                    if vae_asymmetric is not False and supported_asym:
                        from .asymmetric_vae import AsymmetricVAE as SpeedVAE
                        self.vae_asymmetric=True
                    self.vae=SpeedVAE.from_vae(self.vae,mx.float16 if opts['vae']=='fast_fp16' else mx.bfloat16)
            if profile=='speed' and action_epilogue is not False:
                from importlib.metadata import version
                supported=(mx.device_info().get('architecture')=='applegpu_g17s'
                           and version('mlx')=='0.32.4' and getattr(self.action,'kind',None)=='w1')
                if action_epilogue is True and not supported:
                    raise ValueError('Action epilogue validated only for applegpu_g17s / MLX 0.32.4 / action w1')
                self.action_epilogue_reason='validated_m5_w1' if supported else 'unsupported_environment_or_policy'
                if supported:
                    from .action_epilogue import FusedAction
                    self.action=FusedAction(self.action)
                    self.action_epilogue=True
            if profile=='speed' and staged_visual is not False:
                from importlib.metadata import version
                supported=(mx.device_info().get('architecture')=='applegpu_g17s'
                           and version('mlx')=='0.32.4'
                           and self.quantization_info.get('visual',{}).get('kind')=='bf16'
                           and getattr(self.action,'kind',None)=='w1')
                if staged_visual is True and not supported:
                    raise ValueError('Staged visual validated for applegpu_g17s / MLX 0.32.4 / visual BF16 / action W1')
                self.staged_visual_reason='validated_m5_mode2' if supported else 'unsupported_environment_or_policy'
                if supported:
                    from .staged_visual import StagedVisual
                    self.visual=StagedVisual(self.visual)
                    if prepacked_visual is not False:
                        from .prepacked_visual import PrepackedVisual
                        self.visual=PrepackedVisual(self.visual)
                        if fused_concat is not False:
                            from .split_quant_visual import SplitQuantVisual
                            self.visual=SplitQuantVisual(self.visual)
                            if fused_gate is not False:
                                from .gate_quant_visual import GateQuantVisual
                                self.visual=GateQuantVisual(self.visual)
                                if tail_visual is not False:
                                    from .tail_visual import TailVisual
                                    self.visual=TailVisual(self.visual)
                    self.quantization_info['visual']=dict(self.visual.info)
                    self.staged_visual=True
            self.prefill=mx.compile(self.prefill)
            self._compiled_step=mx.compile(self._step)
            if self.kv_workspace:
                from .workspace import workspace_step
                self._compiled_workspace_step=mx.compile(lambda cache,x,t,delta,pe: workspace_step(self,cache,x,t,delta,pe))

        if text_cache_size:
            from .text_cache import TextCache
            self.qwen=TextCache(self.qwen,capacity=text_cache_size)

    def clear_text_cache(self):
        if self.text_cache_size:self.qwen.clear()

    def text_cache_stats(self):
        if not self.text_cache_size:return {'enabled':False}
        return dict(enabled=True,capacity=self.qwen.capacity,max_feature_bytes=self.qwen.max_bytes,hits=self.qwen.hits,misses=self.qwen.misses,feature_bytes=self.qwen.bytes)

    def warmup(self,samples,repeats=1):
        """Opt-in prewarm of actual dynamic inputs; see imagewam_mlx.warmup."""
        from .warmup import warmup
        return warmup(self,samples,repeats)

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
        if self.kv_workspace:
            from .workspace import workspace_denoise
            return workspace_denoise(self,cache,noise,steps,trace)
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
        if not self.overlap:
            x,st=run('preprocess',lambda:self.preprocess(images,state))
        start=time.perf_counter(); ids=self.tokenize(prompt)
        if profile:self.profile['tokenization']=(time.perf_counter()-start)*1000
        hidden=run('text_encoder',lambda:self.qwen(mx.array(ids)))
        if self.overlap:
            # Submit GPU text work before CPU image resizing; no feature cache.
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
