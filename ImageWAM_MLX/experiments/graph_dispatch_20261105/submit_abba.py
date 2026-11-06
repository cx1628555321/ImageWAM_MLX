import os
os.environ.setdefault('MLX_MAX_OPS_PER_BUFFER','256')
from thread_qos import apply
qos_record=apply(os.environ.get("IMAGEWAM_TEST_QOS","inherit"))
import sys,json,time,hashlib,resource,subprocess
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import mlx.core as mx
from imagewam_mlx import ImageWAM
from imagewam_mlx.model import ImageWAM as Base
from partial import Candidate
mode,label=sys.argv[1:];m=ImageWAM('weights/bf16','assets',profile='speed')
import types
from submit import denoise
assert mode in ('native','xonly','every2','every5')
if mode!='native':
 m.submit_interval={'xonly':1,'every2':2,'every5':5}[mode];m.denoise=types.MethodType(denoise,m)
p=Path('examples/robotwin_ep0_frame0');raw=np.load(p/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(p/'prompt.txt').read_text();noise=np.load(p/'initial_noise_fp32.npy')
for _ in range(10):a,n=m.predict(images,raw['state'],prompt,noise);mx.synchronize()
times=[];cpu_times=[]
usage_before=resource.getrusage(resource.RUSAGE_SELF)
swap_before=subprocess.check_output(["sysctl","vm.swapusage"],text=True)
for _ in range(100):
 mx.synchronize();t=time.perf_counter();cpu_start=time.process_time();a,n=m.predict(images,raw['state'],prompt,noise);mx.synchronize();times.append((time.perf_counter()-t)*1000);cpu_times.append((time.process_time()-cpu_start)*1000)
root=Path('records/graph_dispatch_20261105/submit_abba');root.mkdir(exist_ok=True)
assert not (root/(label+'.json')).exists(), 'Use a new label'
profiles=[]
for _ in range(3):
 m.predict(images,raw['state'],prompt,noise,profile=True);profiles.append(dict(m.profile))
np.savez(root/(label+'.npz'),action=a,normalized=n)
res=dict(qos=qos_record,cpu_ms=cpu_times,swap_before=swap_before,swap_after=subprocess.check_output(["sysctl","vm.swapusage"],text=True),pagefaults=dict(minor=resource.getrusage(resource.RUSAGE_SELF).ru_minflt-usage_before.ru_minflt,major=resource.getrusage(resource.RUSAGE_SELF).ru_majflt-usage_before.ru_majflt),active_memory=mx.get_active_memory(),peak_memory=mx.get_peak_memory(),max_ops=os.environ['MLX_MAX_OPS_PER_BUFFER'],module_profiles=profiles,mode=mode,mean_ms=float(np.mean(times)),median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),samples_ms=times,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(Path('imagewam_mlx').glob('*.py'))+list(Path('experiments/graph_dispatch_20261105').glob('*.py'))})
(root/(label+'.json')).write_text(json.dumps(res,indent=2));print(label,res['mean_ms'],flush=True)
