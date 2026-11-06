import ast,types,importlib.util
from pathlib import Path
import mlx.core as mx
from imagewam_mlx.prefix_int8 import ExtraVisual
from imagewam_mlx import staged_visual as sv
from imagewam_mlx.model import ImageWAM as Base
from imagewam_mlx.workspace import workspace_step
from imagewam_mlx.ops import norm,gated

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def bias_kernel(name):
 tree=ast.parse(Path('imagewam_mlx/device_visual.py').read_text())
 node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
 kw={k.arg:k.value for k in node.value.keywords};source=ast.literal_eval(kw['source']);old='static_cast<T>(float(frag[e]) * x_scale[r] * w_scale[c]);';assert source.count(old)>=2
 source=source.replace(old,'T(float(T(float(frag[e]) * x_scale[r] * w_scale[c])) + float(out_bias[c]));')
 return mx.fast.metal_kernel(name='bg_bias_'+name,input_names=ast.literal_eval(kw['input_names'])+['out_bias'],output_names=['y'],header=sv.header,source=source)
class FusedPrefix(ExtraVisual):
 def __init__(self,orig,mode):
  self.__dict__.update(orig.__dict__);self.fusion_mode=mode;self.bkernels=(bias_kernel('full_kernel'),bias_kernel('tail_kernel')) if mode in ('bias','both') else None
  self.qmod=load('bg_mod','experiments/mod_quant_20261104/candidate_norm.py').q if mode in ('norm','both') else None
 def project(self,x,p,qparts=None):
  shape=x.shape;flat=x.reshape(-1,shape[-1]);rows,k=flat.shape;parts=self.extra[p];n=parts[0].shape[0];q,s,e=sv.quantize(flat) if qparts is None else qparts
  kw=dict(inputs=[*parts,q,s,e],template=[('T',x.dtype),('group_size',64),('bits',4),('x_span',0),('XK',k),('ROWS',rows)],grid=((n+63)//64*128,(rows+63)//64,1),threadgroup=(128,1,1),output_shapes=[(rows,n)],output_dtypes=[x.dtype])
  bias=self.w.get(p+'.bias')
  if self.bkernels and bias is not None:
   kw['inputs'].append(bias);y=self.bkernels[int(0<rows%64<=32)](**kw)[0]
  else:
   y=self.mm(**kw)[0]
   if bias is not None:y=y+bias
  return y.reshape(*shape[:-1],n)
 def linear(self,x,p):return self.project(x,p) if p in self.extra else super().linear(x,p)
 def norm_project(self,x,p,scale,shift):
  if p not in self.extra:return self.linear((1+scale)*norm(x)+shift,p)
  flat=x.reshape(-1,x.shape[-1]);rows,k=flat.shape;mp=(rows+63)//64*64
  qs=self.qmod(inputs=[flat,scale.reshape(-1),shift.reshape(-1)],template=[('T',x.dtype),('XK',k),('THREADS',256)],grid=(256,mp,1),threadgroup=(256,1,1),output_shapes=[(mp,k),(rows,),(mp,k//64+1)],output_dtypes=[mx.int8,mx.float32,mx.uint8])
  return self.project(x,p,qs)
 def double_prepare(self,x,p,m,pe,stream='img'):
  if self.qmod is None:return super().double_prepare(x,p,m,pe,stream)
  return self.qkv(self.norm_project(x,p+'.'+stream+'_attn.qkv',m[1],m[0]),p+'.'+stream+'_attn.norm',pe)
 def double_post(self,x,att,p,m,stream='img'):
  if self.qmod is None:return super().double_post(x,att,p,m,stream)
  x=x+m[2]*self.linear(att,p+'.'+stream+'_attn.proj');z=self.norm_project(x,p+'.'+stream+'_mlp.0',m[4],m[3]);return x+m[5]*self.linear(gated(z),p+'.'+stream+'_mlp.2')

def install(m,mode,raw=None,scales=None):
 if mode in ('baseline','math_relaxed','math_fast'):return []
 if mode.startswith('fuse_'):
  m.visual=FusedPrefix(m.visual,mode[5:]);m.prefill=mx.compile(lambda h,l,s:Base.prefill(m,h,l,s));return m.visual.inventory
 from imagewam_mlx.quantized import PackedRopeWeights
 from imagewam_mlx.runtime import RopeWeights
 if mode.startswith('prefix_affine_'):
  bits=int(mode[-1]);raw=mx.load(str(Path(m.weights_path)/'visual.safetensors'));raw.pop('single_blocks.19.linear2.weight',None);raw.pop('single_blocks.19.linear2.bias',None)
  v=RopeWeights(dict(raw));packed={};selected=set(m.visual.extra)|set(m.visual.packed)
  for p in selected:
   packed[p]=mx.quantize(raw[p+'.weight'],group_size=64,bits=bits);mx.eval(packed[p]);del v.w[p+'.weight']
  old=v.linear
  def linear(self,x,p):
   if p not in packed:return old(x,p)
   w,s,b=packed[p];y=mx.qqmm(x,w,scales=s,biases=b,group_size=64,bits=bits,mode='affine');bias=self.w.get(p+'.bias');return y if bias is None else y+bias
  v.linear=types.MethodType(linear,v);m.visual=v;m.prefill=mx.compile(lambda h,l,s:Base.prefill(m,h,l,s));return [dict(prefix_affine_bits=bits,layers=len(packed))]
 raw=mx.load(str(Path(m.weights_path)/'action.safetensors'));mx.eval(raw)
 if mode.startswith('native_'):kind=mode[7:];a=PackedRopeWeights.from_weights(RopeWeights(raw),kind)
 elif mode.startswith('qq_'):
  kind=mode[3:];a=PackedRopeWeights.from_weights(RopeWeights(raw),kind);a.quantize_input=True
 elif mode.startswith('affine_'):
  kind=mode[7:];a=PackedRopeWeights.from_weights(RopeWeights(raw),kind);old=a.linear
  def linear(self,x,p):
   if p not in self.packed:return old(x,p)
   w,s,b=self.packed[p];y=mx.qqmm(x,w,scales=s,biases=b,group_size=self.group_size,bits=self.bits,mode='affine');bias=self.w.get(p+'.bias');return y if bias is None else y+bias
  a.linear=types.MethodType(linear,a)
 else:raise ValueError(mode)
 m.action=a;m._compiled_workspace_step=mx.compile(lambda c,x,t,d,p:workspace_step(m,c,x,t,d,p));return [dict(action_kind=kind,qq=mode.startswith(('qq_','affine_')))]
def action_int8(*args):raise NotImplementedError('Not scheduled')
