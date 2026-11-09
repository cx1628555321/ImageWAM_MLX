"""One independent process per policy. Timings exclude conversion and disk I/O."""
import sys,json,time,argparse,traceback
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
p=argparse.ArgumentParser();p.add_argument('--policy',required=True,help='JSON component policy, or accuracy');p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=12);p.add_argument('--options',default='{}');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
r={'policy':a.policy,'options':json.loads(a.options),'status':'running','times_ms':[],'modules':[]}
def save():(a.output/'results.json').write_text(json.dumps(r,indent=2))
save()
try:
 start=time.perf_counter();m=ImageWAM('weights/bf16','assets',profile='accuracy' if a.policy=='accuracy' else 'speed',quantization=None if a.policy=='accuracy' else json.loads(a.policy),speed_options=None if a.policy=='accuracy' else json.loads(a.options));mx.synchronize();r['load_and_pack_s']=time.perf_counter()-start;r['quantization']=m.quantization_info;mx.clear_cache()
 r['active_after_load']=mx.get_active_memory();print('LOADED',r['load_and_pack_s'],r['active_after_load'],flush=True)
 ref=Path('records/final_default_20261103');raw=np.load(ref/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};pr=json.loads((ref/'input_manifest.json').read_text())['prompt'];noise=np.load(ref/'initial_noise_fp32.npy')
 def run(profile=False):
  mx.synchronize();t=time.perf_counter();out,n=m.predict(images,raw['state'],pr,noise,profile=profile);mx.synchronize();return out,n,(time.perf_counter()-t)*1000
 out,n,r['first_ms']=run()
 for _ in range(6):run()
 for i in range(a.iterations):
  out,n,ms=run();r['times_ms'].append(ms)
 for _ in range(3):
  _,_,ms=run(True);r['modules'].append(dict(m.profile,total_ms=ms))
 np.savez(a.output/'output.npz',action=out,normalized=n)
 r['finite']=bool(np.isfinite(out).all());refout=np.load(ref/'reference_action.npy')
 r['difference']={'max_abs':float(np.max(np.abs(out-refout))),'mae':float(np.mean(np.abs(out-refout))),'rmse':float(np.sqrt(np.mean((out-refout)**2)))} if r['finite'] else None
 r['mean_ms']=float(np.mean(r['times_ms']));r['p95_ms']=float(np.percentile(r['times_ms'],95));r['memory']={'active':mx.get_active_memory(),'peak':mx.get_peak_memory(),'cache':mx.get_cache_memory()};r['status']='completed';save();print(json.dumps({k:v for k,v in r.items() if k not in ('modules','times_ms','quantization')}),flush=True)
except Exception:
 r['status']='failed';r['error']=traceback.format_exc();save();raise
