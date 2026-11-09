"""Read checkpoint headers and run small synthetic GPU capability checks only."""
import json,struct,math,collections,importlib.metadata
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
out=Path('records/quantization_support_20261103');out.mkdir(exist_ok=False)
root=Path('weights/bf16/action_only');manifest=json.loads((root/'manifest.json').read_text())
r={'mlx':importlib.metadata.version('mlx'),'device':mx.device_info(),'manifest':{k:v for k,v in manifest.items() if k!='files'},'files':{},'tests':[],
 'nn_quantized_classes':[k for k in dir(nn) if 'Quantized' in k], 'qqmm_doc':getattr(mx,'qqmm').__doc__}
for name in manifest['files']:
 with (root/name).open('rb') as f:
  header=json.loads(f.read(struct.unpack('<Q',f.read(8))[0]))
 dt=collections.Counter();size=0;linear=0;excluded=[]
 for k,v in header.items():
  if k=='__metadata__':continue
  n=math.prod(v['shape']);dt[v['dtype']]+=n;size+=v['data_offsets'][1]-v['data_offsets'][0]
  if len(v['shape'])==2 and k.endswith('weight') and 'embed_tokens' not in k and v['shape'][-1]%64==0:linear+=n
  elif len(v['shape'])==2:excluded.append({'name':k,'shape':v['shape']})
 r['files'][name]={'elements_by_dtype':dict(dt),'bytes':size,'eligible_linear_elements_g64':linear,'other_2d':excluded}
for mode,bits,group in [('affine',b,64) for b in (1,2,3,4,5,6,8)]+[('mxfp4',4,32),('mxfp8',8,32),('nvfp4',4,16)]:
 row={'mode':mode,'bits':bits,'group_size':group}
 try:
  w=mx.random.normal((128,128),key=mx.random.key(42)).astype(mx.bfloat16);parts=mx.quantize(w,group_size=group,bits=bits,mode=mode);mx.eval(parts)
  row['storage_dtypes']=[str(p.dtype) for p in parts];row['checks']=[]
  for length in (1,16,39,193):
   x=mx.random.normal((1,length,128),key=mx.random.key(length)).astype(mx.bfloat16)
   def run(x):return mx.quantized_matmul(x,*parts,transpose=True,group_size=group,bits=bits,mode=mode)
   y=run(x);z=mx.compile(run)(x);mx.eval(y,z)
   row['checks'].append({'M':length,'shape':list(y.shape),'dtype':str(y.dtype),'finite':bool(mx.all(mx.isfinite(y)).item()),'compiled_finite':bool(mx.all(mx.isfinite(z)).item())})
  row['supported']=True
 except Exception as e:row['supported']=False;row['error']=str(e)
 r['tests'].append(row);print(mode,bits,row['supported'],row.get('error',''),flush=True)
r['note']='Synthetic API execution is not an ImageWAM accuracy or speed benchmark. No model weights/configs modified.'
(out/'results.json').write_text(json.dumps(r,indent=2));print('FILES',r['files']);print('QQMM',r['qqmm_doc']);print('NN',r['nn_quantized_classes'])
