import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,time,json,ctypes,cProfile,pstats
from pathlib import Path
sys.path.insert(0,str(Path.cwd()));sys.path.insert(0,str(Path('experiments/roadmap_20261105').resolve()))
from common import inputs
import numpy as np
import mlx.core as mx
from imagewam_mlx import ImageWAM
r=Path('records/audit4_20261105');r.mkdir(exist_ok=True,parents=True)
lib=ctypes.CDLL(str(Path('experiments/audit4_20261105/gpu_probe.dylib').resolve()));lib.wam_probe_begin.argtypes=[ctypes.c_char_p];assert lib.wam_probe_install()>=0
m=ImageWAM('weights/bf16','assets',profile='speed',text_cache_size=0);args=inputs()
for _ in range(10):m.predict(**args)
mx.synchronize();base=[]
for _ in range(30):
 t=time.perf_counter();a,n=m.predict(**args);mx.synchronize();base.append((time.perf_counter()-t)*1000)
assert lib.wam_probe_begin(str((r/'gpu_buffers.jsonl').resolve()).encode())==0
rows=[]
for i in range(30):
 lib.wam_probe_tag(i);t=time.perf_counter();b,z=m.predict(**args);mx.synchronize();rows.append((time.perf_counter()-t)*1000);assert np.array_equal(a,b) and np.array_equal(n,z)
pending=lib.wam_probe_end();np.savez(r/'probe_output.npz',action=b,normalized=z)
prof=cProfile.Profile();prof.enable()
for _ in range(10):m.predict(**args)
mx.synchronize();prof.disable();prof.dump_stats(str(r/'cpu.pstats'))
with (r/'cpu_profile.txt').open('w') as f:pstats.Stats(prof,stream=f).strip_dirs().sort_stats('cumulative').print_stats(50)
(r/'probe_host.json').write_text(json.dumps(dict(baseline_ms=base,instrumented_ms=rows,pending_callbacks=pending,output_exact=True),indent=2));print('probe complete',len(rows),pending,flush=True)
