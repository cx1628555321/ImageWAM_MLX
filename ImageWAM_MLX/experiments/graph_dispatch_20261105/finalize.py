from pathlib import Path
import json,statistics,hashlib
import numpy as np
root=Path('records/graph_dispatch_20261105');ref=np.load(root/'screen/00_native_256.npz');checks={}
for folder in ('screen','qos','sample','submit_abba'):
 files=list((root/folder).glob('*.npz'))
 assert files,folder
 for p in files:
  z=np.load(p);assert all(np.array_equal(z[k],ref[k]) for k in ref.files),p
 checks[folder]=len(files)
v=json.loads((root/'submit/validation.json').read_text());assert v['all_exact'] and len(v['exact_cases'])==51
checks['submit_candidates_51_cases']=True
v=json.loads((root/'submit_dynamic/results.json').read_text());assert len(v)==8 and all(r['exact'] for r in v);checks['dynamic_cases']=len(v)
summary={}
for mode,names in [('baseline',['a1','a2']),('xonly',['b1','b2']),('every5',['c1','c2'])]:
 ds=[json.loads((root/'submit_abba'/(n+'.json')).read_text()) for n in names]
 samples=[x for d in ds for x in d['samples_ms']]
 summary[mode]=dict(mean_ms=statistics.mean(samples),p95_ms=float(np.percentile(samples,95)),cpu_ms=statistics.mean(x for d in ds for x in d['cpu_ms']),runs=[d['mean_ms'] for d in ds])
previous=Path('records/int8_wide_20261104/source_snapshot/imagewam_mlx');current=Path('imagewam_mlx')
assert previous.exists()
changed=[str(p.relative_to(previous)) for p in previous.rglob('*') if p.is_file() and '__pycache__' not in p.parts and (not (current/p.relative_to(previous)).exists() or p.read_bytes()!=(current/p.relative_to(previous)).read_bytes())]
assert not changed,changed
report=dict(checks=checks,submit_independent=summary,production_unchanged=True,adopted=False)
(root/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
