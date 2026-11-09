"""ImageWAM action inference entirely on MLX; one observation per call."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import mlx.core as mx
from transformers import AutoTokenizer
from .ops import D,Weights,flux_rope,attention,silu,norm,rms,rotate
from .qwen import Qwen
from .vae import VAE

class ImageWAM:
    def __init__(self,weights,source,compact=True):
        weights,source=Path(weights),Path(source)
        original_manifest=(weights/'manifest.json').read_bytes()
        if compact and (weights/'action_only/manifest.json').exists():
            candidate=json.loads((weights/'action_only/manifest.json').read_text())
            if candidate.get('source_manifest_sha256')!=hashlib.sha256(original_manifest).hexdigest():
                raise ValueError('Action-only checkpoint source manifest mismatch')
            weights=weights/'action_only'
        manifest=json.loads((weights/'manifest.json').read_text())
        self.compact=manifest.get('layout')=='action_only_v1'
        self.weights_path=str(weights)

        def load(name):
            w=mx.load(str(weights/name)); expected=manifest['files'][name]['tensors']
            if set(w)!=set(expected) or any(list(w[k].shape)!=expected[k] for k in w):raise ValueError('Weight manifest mismatch: '+name)
            # Action inference returns only final visual K/V. Its output
            # projection is never called, including in the legacy path.
            # Validate the original checkpoint before filtering; keep it intact.
            if name=='visual.safetensors':
                w.pop('single_blocks.19.linear2.weight',None)
                w.pop('single_blocks.19.linear2.bias',None)
            mx.eval(w); return w
        self.visual=Weights(load('visual.safetensors')); self.action=Weights(load('action.safetensors'))
        self.proprio=Weights(load('proprio.safetensors')); self.vae=VAE(load('vae.safetensors'))
        qw={}
        for name in sorted(manifest['files']):
            if name.startswith('qwen-'):qw.update(load(name))
        self.qwen=Qwen(qw)
        self.tokenizer=AutoTokenizer.from_pretrained(source/'models/Qwen3-4B',local_files_only=True)
        self.max_text=json.loads((source/'models/Qwen3-4B/config.json').read_text())['max_position_embeddings']
        stats=json.loads((source/'models/robotwin_flux2_4b/dataset_stats.json').read_text())
        self.norms={}
        for group in ('state','action'):
            s=stats[group]['default']; mean=np.array(s['global_mean'],dtype=np.float32); std=np.array(s['global_std'],dtype=np.float32)
            self.norms[group]=(1/(std+np.float32(1e-8)),-mean/(std+np.float32(1e-8)))
        self.profile={}; self.trace={}
    def tokenize(self,prompt):
        text=self.tokenizer.apply_chat_template([{'role':'user','content':str(prompt)}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=self.tokenizer(text,padding=False,truncation=False)['input_ids']
        if len(ids)>self.max_text:raise ValueError('Text exceeds Qwen positional capacity; no truncation allowed')
        return np.array([ids],dtype=np.int32)
    def preprocess(self,images,state):
        if np.asarray(state).shape != (14,) or not np.isfinite(state).all():
            raise ValueError('state must be finite [14]')
        for key in ('cam_high','cam_left_wrist','cam_right_wrist'):
            im=np.asarray(images[key])
            if im.ndim!=3 or im.shape[-1]!=3 or im.dtype!=np.uint8 or min(im.shape[:2])<1:
                raise ValueError(f'{key} must be nonempty uint8 HWC RGB')
        names=('cam_high','cam_left_wrist','cam_right_wrist'); sizes=((256,192),(128,96),(128,96))
        imgs=[np.asarray(Image.fromarray(images[k]).resize(size,Image.Resampling.BILINEAR)) for k,size in zip(names,sizes)]
        rgb=np.concatenate([imgs[0],np.concatenate(imgs[1:],axis=1)],axis=0)
        x=(mx.array(rgb[None]).astype(mx.float32)*(2.0/255.0)).astype(D)
        x=x-1.0
        scale,offset=self.norms['state']; st=np.clip(np.asarray(state,dtype=np.float32)*scale+offset,-5,5)
        return x,mx.array(st.reshape(1,1,14)).astype(D)
    def prefill(self,hidden,latent,st):
        v=self.visual
        ctx=mx.concatenate([hidden,st@self.proprio.w['weight'].T+self.proprio.w['bias']],axis=1)
        txt=v.linear(ctx,'txt_in'); img=v.linear(latent.reshape(1,-1,128),'img_in')
        ids=np.zeros((1,txt.shape[1],4),np.float32); ids[0,:,3]=np.arange(txt.shape[1])
        ti=flux_rope(mx.array(ids).astype(D))
        h,w=latent.shape[1:3]; ids=np.zeros((1,h,w,4),np.float32); ids[...,0]=10; ids[...,1]=np.arange(h)[:,None]; ids[...,2]=np.arange(w)[None,:]
        ii=flux_rope(mx.array(ids.reshape(1,-1,4)).astype(D))
        pe=tuple(mx.concatenate([a,b],axis=2) for a,b in zip(ti,ii))
        vec=v.embed_time(mx.zeros((1,),dtype=D)); mi=v.modulation(vec,'double_stream_modulation_img',True); mt=v.modulation(vec,'double_stream_modulation_txt',True); ms=v.modulation(vec,'single_stream_modulation')
        cache=[]
        for i in range(5):
            p=f'double_blocks.{i}'
            tq,tk,tv=v.double_prepare(txt,p,mt,ti,'txt'); iq,ik,iv=v.double_prepare(img,p,mi,ii)
            q,k,val=[mx.concatenate(pair,axis=2) for pair in ((tq,iq),(tk,ik),(tv,iv))]
            cache.append((k,val)); out=attention(q,k,val)
            img=v.double_post(img,out[:,txt.shape[1]:],p,mi); txt=v.double_post(txt,out[:,:txt.shape[1]],p,mt,'txt')
        x=mx.concatenate([txt,img],axis=1)
        for i in range(20):
            p=f'single_blocks.{i}'
            if i==19 and self.compact:
                kv=v.linear((1+ms[1])*norm(x)+ms[0],p+'.kv')
                k,val=mx.split(kv.reshape(kv.shape[0],kv.shape[1],2,24,128).transpose(2,0,3,1,4),2,axis=0)
                k=v.rotate_key(rms(k[0],v.w[p+'.norm.key_norm.scale']),pe)
                cache.append((k,val[0])); break
            q,k,val,mlp=v.single_prepare(x,p,ms,pe); cache.append((k,val))
            if i<19:x=v.single_post(x,attention(q,k,val),mlp,p,ms)
        return cache
    def denoise(self,cache,noise,steps=10,trace=False):
        if not (1<=noise.shape[1]<=64) or noise.shape[0]!=1 or noise.shape[2]!=14:raise ValueError('noise must be [1,T<=64,14]')
        if steps<1:raise ValueError('steps must be positive')
        a=self.action; x=mx.array(noise).astype(D)
        u=mx.linspace(1,0,steps+1,dtype=mx.float32); sig=5*u/(1+4*u)
        times=(sig[:-1]*1000).astype(D); deltas=(sig[1:]-sig[:-1]).astype(D)
        ids=np.zeros((1,x.shape[1],4),np.float32); ids[...,0]=2; ids[0,:,1]=np.arange(x.shape[1]); pe=flux_rope(mx.array(ids).astype(D))
        for step in range(steps):
            t=(times[step:step+1]/1000).astype(D); vec=a.embed_time(t)
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
            x=x+pred*deltas[step]
            # Bound the lazy graph; keeps intermediate activation memory predictable.
            mx.eval(x)
            if trace:self.trace[f'action_step_{step}']=np.array(x.astype(mx.float32))
        return x
    def predict(self,images,state,prompt,noise,steps=10,profile=False,trace=False):
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
        mx.eval(cache)
        out=run('action_denoise',lambda:self.denoise(cache,noise,steps,trace))
        start=time.perf_counter()
        normalized=np.array(out[0].astype(mx.float32))
        scale,offset=self.norms['action']; result=(normalized-offset)/scale
        if profile:self.profile['postprocess']=(time.perf_counter()-start)*1000
        if trace:
            self.trace.update(image=np.array(x.astype(mx.float32)),state=np.array(st.astype(mx.float32)),text=np.array(hidden.astype(mx.float32)),latent=np.array(latent.astype(mx.float32)),token_ids=ids)
            for i in (0,4,5,24):self.trace[f'cache_k_{i}']=np.array(cache[i][0].astype(mx.float32))
        return result,normalized
