"""Experimental RMSNorm+interleaved RoPE kernel, FP32 reduction / BF16 boundaries."""
import argparse,json,time,sys,shutil,types,copy
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
from imagewam_mlx.runtime import RopeWeights
from imagewam_mlx.ops import rms,rotate
from compile_experiments import stats,error
_kernel=mx.fast.metal_kernel(name='imagewam_rms_rope_precise',input_names=['x','w','c','s'],output_names=['out'],ensure_row_contiguous=False,source=r'''
 uint row=threadgroup_position_in_grid.x;
 uint d=thread_position_in_threadgroup.x;
 uint t=row%x_shape[2], h=(row/x_shape[2])%x_shape[1], b=row/x_shape[2]/x_shape[1];
 size_t off=b*x_strides[0]+h*x_strides[1]+t*x_strides[2]+2*d*x_strides[3];
 float a=float(x[off]),v=float(x[off+x_strides[3]]);
 volatile float aa=a*a,vv=v*v;
 float partial=simd_sum(aa+vv);
 threadgroup float sums[2];
 if((d%32)==0)sums[d/32]=partial;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float inv=metal::precise::rsqrt((sums[0]+sums[1])/128.0f+1e-6f);
 T na=T(a*inv),nv=T(v*inv);
 T wa=T(float(na)*float(w[2*d*w_strides[0]])),wv=T(float(nv)*float(w[(2*d+1)*w_strides[0]]));
 size_t ci=(b%c_shape[0])*c_strides[0]+(h%c_shape[1])*c_strides[1]+t*c_strides[2]+d*c_strides[3];
 size_t si=(b%s_shape[0])*s_strides[0]+(h%s_shape[1])*s_strides[1]+t*s_strides[2]+d*s_strides[3];
 volatile float ca=c[ci]*float(wa),sv=s[si]*float(wv),cv=c[ci]*float(wv),sa=s[si]*float(wa);
 out[row*128+2*d]=T(ca-sv);out[row*128+2*d+1]=T(cv+sa);
''')
def fused(x,w,pe):
 if x.shape[-1]!=128:raise ValueError('kernel requires head_dim=128')
 c,s=pe
 return _kernel(inputs=[x,w,c,s],template=[('T',x.dtype)],grid=(x.size//2,1,1),threadgroup=(64,1,1),output_shapes=[x.shape],output_dtypes=[x.dtype])[0]
class FusedWeights(RopeWeights):
 def qkv(self,x,p,pe):
  q,k,v=mx.split(x.reshape(x.shape[0],x.shape[1],3,24,128).transpose(2,0,3,1,4),3,axis=0)
  return fused(q[0],self.w[p+'.query_norm.scale'],pe),fused(k[0],self.w[p+'.key_norm.scale'],pe),v[0]
class FusedCandidates:
 def __init__(self,model):
  self.model=model;self.original=model._compiled_step
  frozen=copy.copy(model);frozen.action=FusedWeights(model.action.w);self.frozen=frozen
  self.compiled=mx.compile(frozen._step)
 def set(self,mode):self.model._compiled_step=self.compiled if mode=='rms_rope' else self.original

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=30);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 shutil.copyfile(__file__,a.output/'rms_rope_precise_experiments.py')
 model=ImageWAM('weights/bf16','assets');c=FusedCandidates(model)
 ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
 r={'status':'running','rows':[],'results':{}}
 def persist():(a.output/'results.json').write_text(json.dumps(r,indent=2))
 def once(mode,pr=prompt,n=noise,steps=10):
  c.set(mode);mx.synchronize();t=time.perf_counter();out,_=model.predict(images,raw['state'],pr,n,steps=steps);mx.synchronize();assert np.isfinite(out).all();return out,(time.perf_counter()-t)*1000
 try:
  for _ in range(8):once('baseline')
  for mode in ('rms_rope',):
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
