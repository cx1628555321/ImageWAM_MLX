"""Save capture, real operator inputs and isolated eager wall timings. Not kernel durations."""
import sys,json,time,functools,collections,os,shutil,types,traceback
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
from imagewam_mlx.model import ImageWAM as Base
import imagewam_mlx.ops as ops
import imagewam_mlx.model as mm
import imagewam_mlx.runtime as rt
out=Path('records/lowbit_20261104/profile');out.mkdir(parents=True,exist_ok=False)
m=ImageWAM('weights/bf16','assets');ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};pr=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
ref,_=m.predict(images,raw['state'],pr,noise)
x,st=m.preprocess(images,raw['state']);h=m.qwen(mx.array(m.tokenize(pr)));lat=m.vae(x);cache=m.prefill(h,lat,st);mx.eval(cache)
# Capture one representative action step; no full-model tensor dump.
cap={'status':'not_attempted'}
try:
 if shutil.disk_usage(out).free<20*1024**3:raise RuntimeError('Less than 20 GiB free; capture omitted')
 pe=ops.flux_rope(mx.zeros((1,16,4),dtype=mx.bfloat16));ax=mx.array(noise).astype(mx.bfloat16);t=mx.array([1.0],dtype=mx.bfloat16);d=mx.array(-.1,dtype=mx.bfloat16)
 mx.eval(m._compiled_step(cache,ax,t,d,pe));mx.synchronize()
 mx.metal.start_capture(str((out/'action_step.gputrace').resolve()))
 try:mx.eval(m._compiled_step(cache,ax,t,d,pe));mx.synchronize()
 finally:mx.metal.stop_capture()
 cap={'status':'captured','note':'GPU trace requires Xcode GUI analysis; no xctrace/Instruments installed.'}
except Exception:cap={'status':'unavailable','error':traceback.format_exc()}
(out/'capture.json').write_text(json.dumps(cap,indent=2));print('CAPTURE',cap['status'],flush=True)
rows=[];saved=[];phase='';iteration=0
selected={('visual','single_blocks.0.linear1'),('visual','double_blocks.0.img_attn.qkv'),('visual','double_blocks.0.txt_attn.qkv'),('action','single_blocks.0.linear1'),('action','single_blocks.0.linear2'),('qwen','layers.0.mlp.gate_proj')}
def arrays(x):
 if isinstance(x,mx.array):return [x]
 if isinstance(x,(tuple,list)):return sum((arrays(v) for v in x),[])
 if isinstance(x,dict):return sum((arrays(v) for v in x.values()),[])
 return []
def wrap(kind,fn,component=None):
 @functools.wraps(fn)
 def f(*args,**kw):
  global phase
  aa=arrays(args)+arrays(kw);mx.eval(*aa);mx.synchronize();start=time.perf_counter();y=fn(*args,**kw);mx.eval(*arrays(y));mx.synchronize();elapsed=(time.perf_counter()-start)*1000
  key=args[1] if kind in ('linear','conv') else ''
  rows.append(dict(phase=phase,kind=kind,component=component,key=key,ms=elapsed,shapes=[list(v.shape) for v in aa],iteration=iteration))
  tag=(component,key)
  if iteration==0 and kind=='linear' and tag in selected and tag not in saved:
   saved.append(tag);name=component+'__'+key;np.save(out/(name+'.npy'),np.array(args[0].astype(mx.float32)))
  if iteration==0 and kind=='attention' and phase=='action' and not (out/'attention.npz').exists():
   np.savez(out/'attention.npz',**{n:np.array(v.astype(mx.float32)) for n,v in zip(('q','k','v'),args[:3])})
  if iteration==0 and kind=='conv' and key in ('encoder.mid.attn_1.q','encoder.down.2.block.0.conv1') and ('vae',key) not in saved:
   saved.append(('vae',key));np.save(out/('vae__'+key+'.npy'),np.array(args[0].astype(mx.float32)))
  return y
 return f
for component in ('visual','action','qwen'):getattr(m,component).linear=wrap('linear',getattr(m,component).linear,component)
m.vae.conv=wrap('conv',m.vae.conv,'vae');m.vae.gn=wrap('groupnorm',m.vae.gn,'vae')
ops.norm=wrap('layernorm',ops.norm);mm.norm=rt.norm=ops.norm
ops.rms=wrap('rms',ops.rms);rt.rms=ops.rms
rt.fused_rotate=wrap('rope',rt.fused_rotate)
ops.gated=wrap('gated',ops.gated)
mm.attention=wrap('attention',mm.attention);rt.attention=wrap('attention',rt.attention)
m._compiled_step=m._step
for iteration in range(2):
 phase='visual';c=Base.prefill(m,h,lat,st);mx.eval(c)
 phase='action';z=m.denoise(c,noise);mx.eval(z)
 if iteration==0:
  phase='qwen';m.qwen(mx.array(m.tokenize(pr)))
  phase='vae';m.vae(x)
 for ph in ('visual','action'):
  print(iteration,ph,sum(v['ms'] for v in rows if v['iteration']==iteration and v['phase']==ph),flush=True)
(out/'operators.json').write_text(json.dumps({'note':'Input dependencies evaluated before each call; forced-sync eager timings, NOT production compiled kernel durations. Different wrapper levels can overlap (e.g. RMS/RoPE inside helpers).','capture':cap,'operators':rows,'saved_inputs':saved},indent=2))
