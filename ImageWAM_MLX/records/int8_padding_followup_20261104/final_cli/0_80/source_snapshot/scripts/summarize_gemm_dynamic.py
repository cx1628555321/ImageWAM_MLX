"""Summarize actual completed records; no extrapolated steady-state speedup."""
import json
from pathlib import Path
import numpy as np
root=Path('records');cold=root/'dynamic_cold_20261103';warm=root/'dynamic_warm_20261103'
c=json.loads((cold/'results.json').read_text());w=json.loads((warm/'results.json').read_text())
assert c['status']==w['status']=='completed'
assert len(c['cases'])==len(w['cases'])
checks=[]
for x,y in zip(c['cases'],w['cases']):
 assert x['tokens']==y['tokens']
 with np.load(cold/f"case_{x['case']}.npz") as a,np.load(warm/f"case_{y['case']}.npz") as b:
  for key in a.files:
   checks.append({'case':x['case'],'tokens':x['tokens'],'key':key,'bytes_equal':a[key].tobytes()==b[key].tobytes(),'max_abs':float(np.max(np.abs(a[key]-b[key])))})
def stats(xs):return {'mean_ms':float(np.mean(xs)),'median_ms':float(np.median(xs)),'p95_ms':float(np.percentile(xs,95))}
r={'checks':checks,'all_bytes_equal':all(v['bytes_equal'] for v in checks),'all_repeat_equal':all(v['repeats_exact'] for v in c['cases']+w['cases']),
 'warmup_total_ms':sum(sum(v['durations_ms']) for v in w['warmup']),
 'first_request':{'cold_ms':c['cases'][0]['times_ms'][0],'warm_ms':w['cases'][0]['times_ms'][0]},
 'cold_later_lengths':stats([v['times_ms'][0] for v in c['cases'][1:-2]]),
 'warm_later_lengths':stats([v['times_ms'][0] for v in w['cases'][1:-2]]),
 'cold_repeat_requests':stats([t for v in c['cases'][:-2] for t in v['times_ms'][1:]]),
 'warm_repeat_requests':stats([t for v in w['cases'][:-2] for t in v['times_ms'][1:]]),
 'unseen_cold':c['cases'][-2],'unseen_warm':w['cases'][-2],
 'note':'Separate sequential processes, not randomized paired latency comparison. No claimed steady-state gain.'}
first=root/'dynamic_first_20261103'
if (first/'results.json').exists():
 f=json.loads((first/'results.json').read_text());assert f['status']=='completed'
 fc=[]
 for x,y in zip(c['cases'],f['cases']):
  assert x['tokens']==y['tokens']
  with np.load(cold/f"case_{x['case']}.npz") as aa,np.load(first/f"case_{y['case']}.npz") as bb:
   for key in aa.files:fc.append(aa[key].tobytes()==bb[key].tobytes())
 assert len(f['cases'])==len(c['cases'])
 r['single_sample_warmup']={'warmup_ms':sum(sum(v['durations_ms']) for v in f['warmup']),
  'first_request_ms':f['cases'][0]['times_ms'][0],'later_lengths':stats([v['times_ms'][0] for v in f['cases'][1:-2]]),
  'repeat_requests':stats([t for v in f['cases'][:-2] for t in v['times_ms'][1:]]),
  'arrays_checked':len(fc),'all_bytes_equal':all(fc),'all_repeat_equal':all(v['repeats_exact'] for v in f['cases']),
  'unseen':f['cases'][-2]}
(root/'gemm_dynamic_summary_20261103.json').write_text(json.dumps(r,indent=2))
print(json.dumps({k:v for k,v in r.items() if k!='checks'},indent=2))
