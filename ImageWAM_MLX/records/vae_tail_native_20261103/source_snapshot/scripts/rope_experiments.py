"""Strided interleaved RoPE Metal kernel; no input-contiguous materialization."""
import argparse,json,time,sys,shutil
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx.ops import Weights,rms,rotate
from graph_experiments import GraphCandidates
from compile_experiments import stats,error
import imagewam_mlx.model as mm

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
class RopeCandidates:
 def __init__(self,model):
  self.model=model;self.original=model.action;self.base=GraphCandidates(model)
  model.action=RopeWeights(self.original.w);self.fused=model.action;self.candidate=GraphCandidates(model);model.action=self.original
 def set(self,mode):
  self.model.action=self.fused if mode=='rope' else self.original
  (self.candidate if mode=='rope' else self.base).set('lazy_prefix')

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=30);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 shutil.copyfile(__file__,a.output/'rope_experiments.py')
 model=mm.ImageWAM('weights/bf16','assets');c=RopeCandidates(model)
 ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
 r={'status':'running','rows':[],'results':{}}
 def persist():(a.output/'results.json').write_text(json.dumps(r,indent=2))
 def once(mode,pr=prompt,n=noise,steps=10):
  c.set(mode);mx.synchronize();t=time.perf_counter();out,_=model.predict(images,raw['state'],pr,n,steps=steps);mx.synchronize();assert np.isfinite(out).all();return out,(time.perf_counter()-t)*1000
 try:
  for _ in range(8):once('baseline')
  for mode in ('rope',):
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
