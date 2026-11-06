from pathlib import Path
import json,hashlib,shutil,sys,subprocess,statistics
import numpy as np
import mlx.core as mx
root=Path('records/graph_dispatch_20261105');rows=[]
ref=np.load(root/'screen/00_native_256.npz')
for p in sorted((root/'screen').glob('*.json')):
 d=json.loads(p.read_text());a=np.load(p.with_suffix('.npz'));exact=all(np.array_equal(a[k],ref[k]) for k in ref.files)
 assert exact,p
 rows.append(dict(label=p.stem,mode=d['mode'],max_ops=d['max_ops'],mean_ms=d['mean_ms'],p95_ms=d['p95_ms'],exact=exact,modules={k:statistics.mean(r[k] for r in d['module_profiles']) for k in d['module_profiles'][0]}))
(root/'screen_summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2),flush=True)
files=list(Path('imagewam_mlx').rglob('*.py'))+list(Path('experiments/graph_dispatch_20261105').glob('*.py'))+list(Path('examples/robotwin_ep0_frame0').glob('*'))
meta=dict(python=sys.version,mlx=mx.__version__,device=mx.device_info(),hardware=subprocess.check_output(['system_profiler','SPHardwareDataType','SPDisplaysDataType'],text=True),sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()})
(root/'metadata.json').write_text(json.dumps(meta,indent=2,default=str))
if not (root/'source_snapshot').exists():
 shutil.copytree('imagewam_mlx',root/'source_snapshot/imagewam_mlx',ignore=shutil.ignore_patterns('__pycache__'))
 shutil.copytree('examples/robotwin_ep0_frame0',root/'source_snapshot/example')
