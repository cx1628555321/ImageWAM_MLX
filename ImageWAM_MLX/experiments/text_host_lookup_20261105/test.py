import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,time,json,copy,types
from pathlib import Path
sys.path.insert(0,str(Path.cwd()));sys.path.insert(0,str(Path('experiments/roadmap_20261105').resolve()))
import numpy as np
import mlx.core as mx
from common import inputs
from imagewam_mlx import ImageWAM
from imagewam_mlx.text_cache import TextCache
class HostCache(TextCache):
 def get(self,ids):
  a=np.asarray(ids);key=(a.dtype.str,a.shape,a.tobytes())
  if key in self.entries:
   self.hits+=1;y=self.entries.pop(key);self.entries[key]=y;return y
  self.misses+=1;y=self.encoder(mx.array(ids))
  if y.nbytes<=self.max_bytes:
   while self.entries and (len(self.entries)>=self.capacity or self.bytes+y.nbytes>self.max_bytes):
    _,v=self.entries.popitem(last=False);self.bytes-=v.nbytes
   self.entries[key]=y;self.bytes+=y.nbytes
  return y

def encode(self,ids,prompt):
 prompt=str(prompt)
 if self.text_cache_size==1 and prompt!=self._text_cache_prompt:self.qwen.clear()
 hidden=self.qwen.get(ids);self._text_cache_prompt=prompt;return hidden
r=Path(os.environ.get('WAM_RECORD_DIR','records/text_host_lookup_20261105'));r.mkdir(exist_ok=False)
m=ImageWAM('weights/bf16','assets',profile='speed');b=copy.copy(m);b.qwen=HostCache(m.qwen.encoder,capacity=1);b._encode_text=types.MethodType(encode,b)
models={'baseline':m,'host_lookup':b};args=inputs();checks=[]
for i,(suffix,alter) in enumerate([('',False),('',True),(' Please move carefully.',False),(' Please move carefully.',True),('',False),(' Please carefully follow the instruction.'*25,False)]):
 args=inputs();args['prompt']+=suffix
 if alter:args['images']['cam_high']=255-args['images']['cam_high'];args['state']=args['state'].copy();args['state'][0]+=.1
 out,n=m.predict(**args);y,z=b.predict(**args);exact=bool(np.array_equal(out,y) and np.array_equal(n,z));assert exact
 np.savez(r/f'case_{i}.npz',baseline=out,candidate=y,normalized=z)
 checks.append(dict(exact=exact,baseline=m.text_cache_stats(),candidate=b.text_cache_stats()))
args=inputs()
for model in models.values():
 model.prepare_text(args['prompt'])
 for _ in range(10):model.predict(**args)
rows={k:[] for k in models};rng=np.random.default_rng(5120)
for i in range(80):
 for k in rng.permutation(list(models)):
  mx.synchronize();t=time.perf_counter();out,n=models[k].predict(**args);mx.synchronize();rows[k].append((time.perf_counter()-t)*1000)
 if i%20==0:print('round',i,flush=True)
# Isolate lookup overhead with actual tokens/features; not additive to e2e.
ids=m.tokenize(args['prompt']);micro={k:[] for k in models}
for i in range(500):
 for k in rng.permutation(list(models)):
  t=time.perf_counter();models[k]._encode_text(ids,args['prompt']);micro[k].append((time.perf_counter()-t)*1000)
result=dict(checks=checks,performance={k:dict(mean_ms=float(np.mean(v)),p95_ms=float(np.percentile(v,95)),samples_ms=v) for k,v in rows.items()},lookup_mean_ms={k:float(np.mean(v)) for k,v in micro.items()})
(r/'results.json').write_text(json.dumps(result,indent=2));print({k:v['mean_ms'] for k,v in result['performance'].items()},result['lookup_mean_ms'],flush=True)
