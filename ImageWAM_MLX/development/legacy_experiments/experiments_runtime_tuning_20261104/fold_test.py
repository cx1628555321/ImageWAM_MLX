import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import ast,copy,json,time,sys
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
from imagewam_mlx.model import ImageWAM as Base
from imagewam_mlx import staged_visual as sv

root=Path('records/staged_fold_20261104');root.mkdir(exist_ok=True)
tree=ast.parse(Path(sv.__file__).read_text())
call=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='mm' for t in n.targets))
src=eval(compile(ast.Expression(next(k.value for k in call.keywords if k.arg=='source')),'source','eval'))
coeff=mx.fast.metal_kernel(name='wam_fold_coeff',input_names=['scales','biases','w_inv'],output_names=['a','b'],header=sv.header,source='uint i=thread_position_in_grid.x; if(i>=scales_shape[0]*scales_shape[1])return; float inv=w_inv[i/scales_shape[1]]; a[i]=int(rint(float(scales[i])*inv)); b[i]=int(rint(float(biases[i])*inv))+(1<<(QMM_I8_FRAC-1));')
fold=mx.fast.metal_kernel(name='wam_fold_weights',input_names=['w','a','b'],output_names=['out'],header=sv.header,source='uint i=thread_position_in_grid.x; const int K=a_shape[1]*64; if(i>=w_shape[0]*K/4)return; uint row=i/(K/4),kk=(i%(K/4))*4; int4 q=qmm_i8_unpack4<4>((const device uint8_t*)w+row*K/2,kk/4); int4 v=(q*a[row*(K/64)+kk/64]+b[row*(K/64)+kk/64])>>QMM_I8_FRAC; ((device char4*)out)[i]=char4(v);')
cs=src.replace('int(rint(float(scales[kk / group_size]) * inv))','int(scales[kk / group_size])').replace('int(rint(float(biases[kk / group_size]) * inv)) +\n          (1 << (QMM_I8_FRAC - 1))','int(biases[kk / group_size])')
assert cs!=src
fs=cs.replace('size_t(n) * (K / pack_factor) * bytes_per_pack','size_t(n) * K')
fs=fs.replace('(kk / pack_factor) * bytes_per_pack','kk')
fs=fs.replace('const int4 q = qmm_i8_unpack4<bits>(p, i);\n        const int4 v = (q * a + b) >> QMM_I8_FRAC;','const int4 v = int4(*(const device char4*)(p + 4*i));')
kernels={k:mx.fast.metal_kernel(name='wam_fold_'+k,input_names=['w','scales','biases','w_inv','w_scale','x8','x_scale','x_exp'],output_names=['y'],header=sv.header,source=s) for k,s in [('coeff',cs),('int8',fs)]}
class Candidate(sv.StagedVisual):
 def __init__(self,orig,mode):
  self.w=dict(orig.w);self.packed={};self.mode=mode
  for key,(w,s,b,wi,ws) in orig.packed.items():
   a,z=coeff(inputs=[s,b,wi],grid=(s.size,1,1),threadgroup=(256,1,1),output_shapes=[s.shape,b.shape],output_dtypes=[mx.int32,mx.int32]);mx.eval(a,z)
   if mode=='int8':
    w=fold(inputs=[w,a,z],grid=(w.shape[0]*s.shape[1]*16,1,1),threadgroup=(256,1,1),output_shapes=[(w.shape[0],s.shape[1]*64)],output_dtypes=[mx.int8])[0];mx.eval(w)
   self.packed[key]=(w,a,z,wi,ws)
 def linear(self,x,p):
  if p not in self.packed:return sv.RopeWeights.linear(self,x,p)
  shape=x.shape;x=x.reshape(-1,shape[-1]);m,k=x.shape;parts=self.packed[p];n=parts[0].shape[0];a,s,e=sv.quantize(x)
  y=kernels[self.mode](inputs=[*parts,a,s,e],template=[('T',x.dtype),('group_size',64),('bits',4),('x_span',0),('XK',k),('ROWS',m)],grid=(((n+63)//64)*128,(m+63)//64,1),threadgroup=(128,1,1),output_shapes=[(m,n)],output_dtypes=[x.dtype])[0].reshape(*shape[:-1],n)
  bias=self.w.get(p+'.bias');return y if bias is None else y+bias
base=ImageWAM('weights/bf16','assets',profile='speed',prepacked_visual=False);models={'baseline':base}
for mode in kernels:
 c=copy.copy(base);c.visual=Candidate(base.visual,mode);c.prefill=mx.compile(lambda h,l,s,c=c:Base.prefill(c,h,l,s));models[mode]=c
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
def run(name):
 mx.synchronize();t=time.perf_counter();a,n=models[name].predict({k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')},raw['state'],prompt,noise);mx.synchronize();return a,n,(time.perf_counter()-t)*1000
outs={}
for name in models:
 for _ in range(5):a,n,dt=run(name)
 outs[name]=(a,n);np.savez(root/(name+'.npz'),action=a,normalized=n)
 assert np.array_equal(a,outs['baseline'][0]) and np.array_equal(n,outs['baseline'][1]),name
 print('warmup exact',name,flush=True)
rows={k:[] for k in models};rng=np.random.default_rng(94)
for i in range(100):
 for name in rng.permutation(list(models)):
  a,n,dt=run(name);rows[name].append(dt)
 if i%20==0:print(i,{k:float(np.mean(v)) for k,v in rows.items()},flush=True)
res={}
for name,v in rows.items():
 d=np.array(rows['baseline'])-v;ci=np.percentile(np.mean(d[rng.integers(0,len(d),(10000,len(d)))],1),[2.5,97.5]);res[name]=dict(mean_ms=float(np.mean(v)),saving_ms=float(d.mean()),ci95=ci.tolist(),samples_ms=v)
(root/'results.json').write_text(json.dumps(res,indent=2));print({k:{a:b for a,b in v.items() if a!='samples_ms'} for k,v in res.items()},flush=True)
cases=json.loads(Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full/comparison.json').read_text())['cases'];ref=Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full');checks=[]
for case in cases:
 label=case['label'];raw=np.load(ref/('base_inputs.npz' if label.startswith('length_') else label+'_inputs.npz'));prompt=case['prompt'];a,n,_=run('baseline')
 for name in kernels:
  b,z,_=run(name);assert np.array_equal(a,b) and np.array_equal(n,z),(label,name);np.savez(root/(label+'_'+name+'.npz'),action=b,normalized=z)
 checks.append(label)
(root/'validation.json').write_text(json.dumps(dict(exact_cases=checks,all_exact=True),indent=2));print('ALL EXACT',len(checks),flush=True)
