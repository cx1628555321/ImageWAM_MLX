from pathlib import Path
import hashlib,json,subprocess,shutil,sys
import numpy as np
import mlx.core as mx
root=Path('records/int8_schedule_20261104');d=root/'abba'
vals={k:json.loads((d/(k+'.json')).read_text()) for k in ('a1','b1','b2','a2')}
a=np.load(d/'a1.npz');checks={k:all(np.array_equal(a[n],np.load(d/(k+'.npz'))[n]) for n in a.files) for k in vals}
r=dict(runs={k:{n:v for n,v in x.items() if n not in ('samples_ms','source_hashes','module_profiles')} for k,x in vals.items()},baseline_mean_ms=(vals['a1']['mean_ms']+vals['a2']['mean_ms'])/2,candidate_mean_ms=(vals['b1']['mean_ms']+vals['b2']['mean_ms'])/2,outputs_exact=checks)
(d/'summary.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
files=list(Path('imagewam_mlx').rglob('*.py'))+list(Path('imagewam_mlx/metal').rglob('*'))+list(Path('experiments/int8_schedule_20261104').glob('*.py'))+list(Path('examples/robotwin_ep0_frame0').glob('*'))
meta=dict(python=sys.version,mlx=mx.__version__,device=mx.device_info(),hardware=subprocess.check_output(['system_profiler','SPHardwareDataType','SPDisplaysDataType'],text=True),sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},default_changed=False)
(root/'metadata.json').write_text(json.dumps(meta,indent=2,default=str))
shutil.copytree('imagewam_mlx',root/'source_snapshot/imagewam_mlx',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree('examples/robotwin_ep0_frame0',root/'source_snapshot/example')
