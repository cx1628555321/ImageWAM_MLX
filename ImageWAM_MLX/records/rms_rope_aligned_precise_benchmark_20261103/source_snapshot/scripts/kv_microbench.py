"""Isolated KV concat / segmented-score attention probe, not end-to-end latency."""
import argparse,json,time,sys,shutil
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx.ops import D
from compile_experiments import stats,error

def concat(kp,vp,k,v):return mx.concatenate([kp,k],axis=2),mx.concatenate([vp,v],axis=2)
def standard(q,kp,vp,k,v):
 kk,vv=concat(kp,vp,k,v)
 return mx.fast.scaled_dot_product_attention(q,kk,vv,scale=128**-.5)
def segmented(q,kp,vp,k,v):
 # One joint softmax; never sum separately normalized attentions.
 logits=mx.concatenate([(q@kp.transpose(0,1,3,2)).astype(mx.float32),(q@k.transpose(0,1,3,2)).astype(mx.float32)],axis=-1)*128**-.5
 w=mx.softmax(logits,axis=-1);n=kp.shape[2]
 return (w[...,:n]@vp.astype(mx.float32)+w[...,n:]@v.astype(mx.float32)).astype(q.dtype)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);shutil.copyfile(__file__,a.output/'kv_microbench.py')
 fns={k:mx.compile(f) for k,f in [('concat',concat),('standard',standard),('segmented',segmented)]}
 r={'status':'running','note':'Synthetic resident inputs, synchronized per call; cache hot; not end-to-end cost or trace measurement. BF16 split score GEMMs may round differently from fused SDPA.','cases':[]}
 mx.random.seed(42)
 for n in (320,328,368):
  q,k,v=[mx.random.normal((1,24,16,128)).astype(D) for _ in range(3)];kp,vp=[mx.random.normal((1,24,n,128)).astype(D) for _ in range(2)];mx.eval(q,k,v,kp,vp)
  args={'concat':(kp,vp,k,v),'standard':(q,kp,vp,k,v),'segmented':(q,kp,vp,k,v)}
  for name,fn in fns.items():
   for _ in range(10):mx.eval(fn(*args[name]))
  times={k:[] for k in fns};out={}
  for i in range(100):
   for name in (list(fns) if i%2==0 else list(fns)[::-1]):
    mx.synchronize();t=time.perf_counter();out[name]=fns[name](*args[name]);mx.eval(out[name]);mx.synchronize();times[name].append((time.perf_counter()-t)*1000)
  x=np.array(out['standard'].astype(mx.float32));y=np.array(out['segmented'].astype(mx.float32));np.save(a.output/f'{n}_standard.npy',x);np.save(a.output/f'{n}_segmented.npy',y)
  row={'prefix_tokens':n,'times_ms':{k:stats(v) for k,v in times.items()},'raw_ms':times,'error':error(y,x)};r['cases'].append(row);print(n,row['times_ms'],row['error'],flush=True)
 r['status']='completed';(a.output/'results.json').write_text(json.dumps(r,indent=2))
if __name__=='__main__':main()
