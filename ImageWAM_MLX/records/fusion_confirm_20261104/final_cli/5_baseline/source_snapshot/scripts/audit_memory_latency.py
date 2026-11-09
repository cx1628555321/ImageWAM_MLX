"""Read-only memory/latency observation; no wired limit or system setting changes."""
import sys,time,json,subprocess,resource
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
out=Path('records/memory_latency_audit_20261103');out.mkdir(exist_ok=False)
def snapshot():
 u=resource.getrusage(resource.RUSAGE_SELF)
 return {'vm_stat':subprocess.check_output(['vm_stat'],text=True),'swap':subprocess.check_output(['sysctl','vm.swapusage'],text=True),'process_major_faults':u.ru_majflt,'process_minor_faults':u.ru_minflt,'mlx_active':mx.get_active_memory(),'mlx_cache':mx.get_cache_memory(),'mlx_peak':mx.get_peak_memory()}
r={'device':mx.device_info(),'before_load':snapshot()};start=time.perf_counter();m=ImageWAM('weights/bf16','assets');mx.synchronize();r['load_s']=time.perf_counter()-start;r['after_load']=snapshot()
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
for _ in range(8):m.predict(images,raw['state'],prompt,noise)
mx.synchronize();r['before_steady']=snapshot();times=[]
for i in range(50):
 mx.synchronize();start=time.perf_counter();a,n=m.predict(images,raw['state'],prompt,noise);mx.synchronize();times.append((time.perf_counter()-start)*1000);assert np.isfinite(a).all()
r['after_steady']=snapshot();r['steady_ms']=times;r['summary']={'mean_ms':float(np.mean(times)),'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'max_ms':float(np.max(times))}
r['note']='System VM counters include all processes; process faults and global counters are not GPU hardware timestamps. No memory residency setting changed.'
np.save(out/'action.npy',a);(out/'results.json').write_text(json.dumps(r,indent=2));print(r['summary'],flush=True)
