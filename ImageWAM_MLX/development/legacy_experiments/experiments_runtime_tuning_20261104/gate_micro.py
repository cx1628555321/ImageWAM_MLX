import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import ast,sys,time,json
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import split_quant_visual as sq
from imagewam_mlx.ops import gated
r=Path('records/gate_quant_20261104');r.mkdir(exist_ok=False)
tree=ast.parse(Path(sq.__file__).read_text());call=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='qx' for t in n.targets));src=ast.literal_eval(next(k.value for k in call.keywords if k.arg=='source'))
src=src.replace('KB=mlp_shape[1]','KB=mlp_shape[1]/2').replace('mlp + size_t(row)*KB + lid*BK-KA','mlp + size_t(row)*KB*2 + lid*BK-KA')
needle='vals[i] = float(x[i]);';assert src.count(needle)==1
kernels={}
for mode in ['exp','lut']:
 expr='T(1.0f/(1.0f+exp(-float(x[i]))))' if mode=='exp' else 'lut[as_type<ushort>(x[i])]'
 code=src.replace(needle,'if(lid*BK<KA){vals[i]=float(x[i]);} else {T sig='+expr+'; T a=T(float(x[i])*float(sig)); vals[i]=float(T(float(a)*float(x[i+KB])));}')
 kernels[mode]=mx.fast.metal_kernel(name='wam_gate_quant_'+mode,input_names=['att','mlp']+(['lut'] if mode=='lut' else []),output_names=['x8','x_scale','x_exp'],header=sq.sv.header,source=code)
M=328;KA=3072;KB=9216;K=KA+KB;mp=(M+63)//64*64;nb=K//64;threads=(nb+31)//32*32
lut=mx.sigmoid(mx.arange(65536,dtype=mx.uint16).view(mx.bfloat16));mx.eval(lut)
def quant(kernel,a,b,extra=[]):return kernel(inputs=[a,b,*extra],template=[('T',a.dtype)],grid=(threads,mp,1),threadgroup=(threads,1,1),output_shapes=[(mp,K),(M,),(mp,nb+1)],output_dtypes=[mx.int8,mx.float32,mx.uint8])
fns={'baseline':mx.compile(lambda a,b:quant(sq.qx,a,gated(b))),'exp':mx.compile(lambda a,b:quant(kernels['exp'],a,b)),'lut':mx.compile(lambda a,b:quant(kernels['lut'],a,b,[lut]))}
mx.random.seed(61);a=mx.random.normal((M,KA)).astype(mx.bfloat16);b=mx.random.normal((M,KB*2)).astype(mx.bfloat16);mx.eval(a,b)
ref=fns['baseline'](a,b);mx.eval(ref);checks={}
for name,f in fns.items():
 out=f(a,b);mx.eval(out);checks[name]=[dict(exact=bool(np.array_equal(np.array(x),np.array(y))),different=int(np.count_nonzero(np.array(x)!=np.array(y))),max_abs=float(np.max(np.abs(np.array(x).astype(float)-np.array(y).astype(float))))) for x,y in zip(ref,out)]
 for i in range(20):mx.eval(f(a,b))
print('CHECKS',checks,flush=True)
rows={k:[] for k in fns};rng=np.random.default_rng(74)
for i in range(500):
 for name in rng.permutation(list(fns)):
  mx.synchronize();t=time.perf_counter();out=fns[name](a,b);mx.eval(out);mx.synchronize();rows[name].append((time.perf_counter()-t)*1000)
res={}
for name,v in rows.items():
 d=np.array(rows['baseline'])-v;ci=np.percentile(np.mean(d[rng.integers(0,len(d),(10000,len(d)))],1),[2.5,97.5]);res[name]=dict(mean_ms=float(np.mean(v)),saving_ms=float(d.mean()),ci95=ci.tolist(),samples_ms=v,checks=checks[name])
(r/'micro.json').write_text(json.dumps(res,indent=2));print({k:{a:b for a,b in v.items() if a!='samples_ms'} for k,v in res.items()},flush=True)
