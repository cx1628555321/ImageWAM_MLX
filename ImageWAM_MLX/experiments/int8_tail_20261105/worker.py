import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,json,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import mlx.core as mx
from imagewam_mlx import ImageWAM
from imagewam_mlx.model import ImageWAM as Base
from candidate import Candidate
mode,label=sys.argv[1:];load_start=time.perf_counter();m=ImageWAM('weights/bf16','assets',profile='speed')
if mode!='baseline':
 m.visual=Candidate(m.visual,mode);m.prefill=mx.compile(Base.prefill.__get__(m,type(m)))
p=Path('examples/robotwin_ep0_frame0');raw=np.load(p/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(p/'prompt.txt').read_text();noise=np.load(p/'initial_noise_fp32.npy')
load_ms=(time.perf_counter()-load_start)*1000
first_start=time.perf_counter();first_a,first_n=m.predict(images,raw['state'],prompt,noise);mx.synchronize();first_ms=(time.perf_counter()-first_start)*1000
for _ in range(10):a,n=m.predict(images,raw['state'],prompt,noise);mx.synchronize()
times=[]
for _ in range(100):
 mx.synchronize();t=time.perf_counter();a,n=m.predict(images,raw['state'],prompt,noise);mx.synchronize();times.append((time.perf_counter()-t)*1000)
root=Path('records/int8_tail_20261105/abba');root.mkdir(exist_ok=True)
assert not (root/(label+'.json')).exists(), 'Use a new label'
profiles=[]
for _ in range(5):
 m.predict(images,raw['state'],prompt,noise,profile=True);profiles.append(dict(m.profile))
np.savez(root/(label+'.npz'),action=a,normalized=n)
res=dict(load_ms=load_ms,first_ms=first_ms,module_profiles=profiles,mode=mode,mean_ms=float(np.mean(times)),median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),samples_ms=times,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(Path('imagewam_mlx').glob('*.py'))+list(Path('experiments/int8_tail_20261105').glob('*.py'))})
(root/(label+'.json')).write_text(json.dumps(res,indent=2));print(label,res['mean_ms'],flush=True)
