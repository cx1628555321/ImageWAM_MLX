"""Compute only final-layer visual K/V projection; experimental precision check."""
import argparse,json,time,sys,shutil,types
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
from imagewam_mlx.ops import D,flux_rope,attention,norm,rms
from imagewam_mlx.runtime import fused_rotate
from compile_experiments import stats,error

def selective_prefill(self,hidden,latent,st):
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
    for i in range(19):
        p=f'single_blocks.{i}'; q,k,val,mlp=v.single_prepare(x,p,ms,pe); cache.append((k,val))
        x=v.single_post(x,attention(q,k,val),mlp,p,ms)
    # Final visual block only provides K/V to the action expert.
    p='single_blocks.19'
    z=(1+ms[1])*norm(x)+ms[0]
    kv=z@v.w[p+'.linear1.weight'][3072:9216].T
    if p+'.linear1.bias' in v.w:kv=kv+v.w[p+'.linear1.bias'][3072:9216]
    k,val=mx.split(kv.reshape(kv.shape[0],kv.shape[1],2,24,128).transpose(2,0,3,1,4),2,axis=0)
    k=fused_rotate(rms(k[0],v.w[p+'.norm.key_norm.scale']),pe)
    cache.append((k,val[0]))
    return cache

class SelectiveCandidates:
 def __init__(self,model):
  self.model=model;self.original=model.prefill
  self.candidate=mx.compile(types.MethodType(selective_prefill,model))
 def set(self,mode):self.model.prefill=self.candidate if mode=='selective_kv' else self.original

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=30);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 shutil.copyfile(__file__,a.output/'selective_kv_experiments.py')
 model=ImageWAM('weights/bf16','assets');c=SelectiveCandidates(model)
 ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
 r={'status':'running','rows':[],'results':{}}
 def persist():(a.output/'results.json').write_text(json.dumps(r,indent=2))
 def once(mode,pr=prompt,n=noise,steps=10):
  c.set(mode);mx.synchronize();t=time.perf_counter();out,_=model.predict(images,raw['state'],pr,n,steps=steps);mx.synchronize();assert np.isfinite(out).all();return out,(time.perf_counter()-t)*1000
 try:
  for _ in range(8):once('baseline')
  for mode in ('selective_kv',):
   print('START',mode,flush=True);_,first=once(mode)
   for _ in range(6):once(mode)
   times={'baseline':[],mode:[]};outputs={}
   for i in range(a.iterations):
    for k in (('baseline',mode) if i%2==0 else (mode,'baseline')):
     x,ms=once(k);outputs[k]=x;times[k].append(ms);r['rows'].append({'candidate':mode,'mode':k,'index':i,'ms':ms});np.save(a.output/f'{mode}_{k}_{i:03d}.npy',x)
   row={'first_ms':first,'times':{k:stats(v) for k,v in times.items()},'error':error(outputs[mode],outputs['baseline']),'dynamic':[]}
   inv=json.loads(Path('../ImageWAM/development_records/m5pro_robotwin/20261103_dynamic_optimization/text_lengths.json').read_text())['examples_by_length']
   for length in (31,79,48,31):
    pr=inv[str(length)]['prompt'];ref,_=once('baseline',pr);x,ms=once(mode,pr);row['dynamic'].append({'tokens':length,'error':error(x,ref)})
   for horizon,steps in ((8,3),(16,7),(32,10)):
    n=np.random.default_rng(7).standard_normal((1,horizon,14)).astype(np.float32);ref,_=once('baseline',n=n,steps=steps);x,_=once(mode,n=n,steps=steps);row['dynamic'].append({'horizon':horizon,'steps':steps,'error':error(x,ref)})
   r['results'][mode]=row;persist();print('RESULT',mode,row,flush=True)
  r['status']='completed';persist()
 except BaseException:
  import traceback
  r['status']='failed';r['traceback']=traceback.format_exc();persist();raise
if __name__=='__main__':main()
