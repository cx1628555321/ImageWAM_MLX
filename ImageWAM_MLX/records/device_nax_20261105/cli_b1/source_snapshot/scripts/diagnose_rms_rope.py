"""Capture real compiled action Q/K, compare normalization stages; not a benchmark."""
import argparse,json,sys,copy,inspect,shutil
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
from imagewam_mlx.runtime import RopeWeights
from imagewam_mlx.ops import D,rms,rotate,flux_rope
import rms_rope_experiments as exp
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
source=inspect.getsource(exp).split("source=r'''",1)[1].split("''')",1)[0]
source+='''
 normal[row*128+2*d]=na;normal[row*128+2*d+1]=nv;
 weighted[row*128+2*d]=wa;weighted[row*128+2*d+1]=wv;
 if(d==0){meanout[row]=(sums[0]+sums[1])/128.0f;invout[row]=inv;}
'''
kernel=mx.fast.metal_kernel(name='rms_rope_diagnostic',input_names=['x','w','c','s'],output_names=['out','normal','weighted','meanout','invout'],ensure_row_contiguous=False,source=source)
def diag(x,w,pe):
 shape=x.shape;return kernel(inputs=[x,w,*pe],template=[('T',x.dtype)],grid=(x.size//2,1,1),threadgroup=(64,1,1),output_shapes=[shape,shape,shape,shape[:-1],shape[:-1]],output_dtypes=[D,D,D,mx.float32,mx.float32])
def arr(x):return np.array(x.astype(mx.float32))
def diff(x,y):
 d=abs(x-y);return {'different':int(np.count_nonzero(x!=y)),'max_abs':float(d.max())}
records=[]
class CaptureWeights(RopeWeights):
 def qkv(self,x,p,pe):
  q,k,v=mx.split(x.reshape(x.shape[0],x.shape[1],3,24,128).transpose(2,0,3,1,4),3,axis=0)
  records.extend([(p,'q',q[0],self.w[p+'.query_norm.scale'],pe),(p,'k',k[0],self.w[p+'.key_norm.scale'],pe)])
  return super().qkv(x,p,pe)
m=ImageWAM('weights/bf16','assets');probe=copy.copy(m);probe.action=CaptureWeights(m.action.w)
def captured_step(cache,x,t,delta,pe):
 records.clear();out=probe._step(cache,x,t,delta,pe)
 return out,tuple((r[2],r[3],*r[4]) for r in records)
compiled=mx.compile(captured_step)
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');im={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};inv=json.loads(Path('../ImageWAM/development_records/m5pro_robotwin/20261103_dynamic_optimization/text_lengths.json').read_text());pr=inv['examples_by_length']['31']['prompt'];noise=np.load(ex/'initial_noise_fp32.npy')
reference,_=m.predict(im,raw['state'],pr,noise,trace=True);refsteps={k:v.copy() for k,v in m.trace.items() if k.startswith('action_step_')}
image,st=m.preprocess(im,raw['state']);hidden=m.qwen(mx.array(m.tokenize(pr)));latent=m.vae(image);cache=m.prefill(hidden,latent,st);mx.eval(cache)
x=mx.array(noise).astype(D);u=mx.linspace(1,0,11,dtype=mx.float32);sig=5*u/(1+4*u);times=(sig[:-1]*1000).astype(D);delta=(sig[1:]-sig[:-1]).astype(D);ids=np.zeros((1,x.shape[1],4),np.float32);ids[...,0]=2;ids[0,:,1]=np.arange(x.shape[1]);pe=flux_rope(mx.array(ids).astype(D))
r={'status':'running','prompt':pr,'steps':[],'rows':[]};saved=False
for i in range(10):
 x,values=compiled(cache,x,(times[i:i+1]/1000).astype(D),delta[i],pe);mx.eval(x,values);r['steps'].append({'step':i,'capture_vs_default':diff(arr(x),refsteps[f'action_step_{i}'])})
 for j,(q,w,c,s) in enumerate(values):
  got=diag(q,w,(c,s));y=q.astype(mx.float32);mean=mx.mean(y*y,axis=-1);rs=mx.rsqrt(mean+1e-6);normal=(y*rs[...,None]).astype(D);weighted=normal*w;expected=[rotate(weighted,(c,s)),normal,weighted,mean,rs];direct=exp.fused(q,w,(c,s));mx.eval(got,expected,direct)
  row={'step':i,'layer':j//2,'qk':'q' if j%2==0 else 'k','stages':{name:diff(arr(v),arr(e)) for name,v,e in zip(['rotated','normal','weighted','mean','rsqrt'],got,expected)},'diagnostic_vs_fused':diff(arr(got[0]),arr(direct))};r['rows'].append(row)
  if not saved and row['stages']['rotated']['different']:
   mx.save_safetensors(str(a.output/'first_divergence.safetensors'),{'q':q,'w':w,'cos':c,'sin':s,**{f'actual_{name}':v for name,v in zip(['rotated','normal','weighted','mean','rsqrt'],got)},**{f'reference_{name}':v for name,v in zip(['rotated','normal','weighted','mean','rsqrt'],expected)}});r['first_divergence']=row;saved=True
 print('step',i,'capture',r['steps'][-1],'rotated differing rows',sum(z['stages']['rotated']['different']>0 for z in r['rows'] if z['step']==i),flush=True)
 (a.output/'diagnostic.json').write_text(json.dumps(r,indent=2))
r['status']='completed';r['summary']={name:{'rows_differing':sum(x['stages'][name]['different']>0 for x in r['rows']),'max_abs':max(x['stages'][name]['max_abs'] for x in r['rows'])} for name in ['mean','rsqrt','normal','weighted','rotated']};(a.output/'diagnostic.json').write_text(json.dumps(r,indent=2));shutil.copyfile(__file__,a.output/Path(__file__).name);print(r['summary'],flush=True)
