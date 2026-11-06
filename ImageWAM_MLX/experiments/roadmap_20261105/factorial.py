from pathlib import Path
import subprocess,sys,json
import numpy as np
r=Path('records/roadmap_20261105');modes=['base_bound','base_lambda','shape_bound','shape_lambda'];rng=np.random.default_rng(6115);order=list(rng.permutation(modes));seq=order+list(reversed(order));(r/'factorial_order.json').write_text(json.dumps(seq))
for i,mode in enumerate(seq):
 label=f'{i}_{mode}'
 with (r/('factor_'+label+'.log')).open('x') as f:subprocess.run([sys.executable,'-u','experiments/roadmap_20261105/factor_worker.py',mode,label],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',label,flush=True)
result={};ref=np.load(r/'independent/0_baseline.npz')['actions'][0]
for mode in modes:
 paths=[r/'factorial'/f'{i}_{m}.json' for i,m in enumerate(seq) if m==mode];ds=[json.loads(p.read_text()) for p in paths];times=[x for d in ds for x in d['samples_ms']]
 result[mode]=dict(mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95)),blocks=[d['mean_ms'] for d in ds],cpu=[d['process_cpu_ms'] for d in ds],all_exact=all(np.all(np.load(p.with_suffix('.npz'))['actions']==ref) for p in paths))
(r/'factorial_summary.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
