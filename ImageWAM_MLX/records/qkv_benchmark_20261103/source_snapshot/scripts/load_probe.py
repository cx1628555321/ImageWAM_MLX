"""Fresh-process loading probe; filesystem/driver caches are not flushed."""
import json,sys,time
from pathlib import Path
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
out=Path(sys.argv[1]);rows=[];original=mx.load
def load(path,*args,**kwargs):
 t=time.perf_counter();v=original(path,*args,**kwargs);mx.eval(v);mx.synchronize();rows.append({'file':Path(path).name,'ms':(time.perf_counter()-t)*1000});return v
mx.load=load;t=time.perf_counter();m=ImageWAM('weights/bf16','assets');mx.synchronize();total=(time.perf_counter()-t)*1000
with out.open('x') as f:json.dump({'total_ms':total,'files':rows,'other_ms':total-sum(r['ms'] for r in rows),'scope':'fresh Python process, no inference, uncontrolled OS file cache; VAE layout transformations may be deferred to first inference'},f,indent=2)
print(total)
