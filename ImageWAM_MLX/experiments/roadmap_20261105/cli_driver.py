import subprocess,sys,json
from pathlib import Path
import numpy as np
r=Path('records/roadmap_20261105');sequence=[False,True,True,False]
for i,enabled in enumerate(sequence):
 label=f'cli_{i}_'+('shape' if enabled else 'baseline');out=r/label
 cmd=[sys.executable,'-u','scripts/benchmark.py','--profile','speed','--text-cache-size','0','--steps','10','--reference','examples/robotwin_ep0_frame0','--output',str(out),'--warmup','30','--iterations','100','--profile-iterations','5']
 if enabled:cmd+=['--visual-shape-dispatch']
 with (r/(label+'.log')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',label,flush=True)
result={};ref=np.load(r/'cli_0_baseline/actions_steady_000.npy')
for enabled in (False,True):
 paths=[r/(f'cli_{i}_'+('shape' if enabled else 'baseline')) for i,v in enumerate(sequence) if v==enabled];ds=[json.loads((p/'benchmark.json').read_text()) for p in paths];rows=[v['wall_ms'] for d in ds for v in d['iterations'] if v['phase']=='steady'];exact=all(np.array_equal(ref,np.load(f)) for p in paths for f in p.glob('actions_steady_*.npy'))
 result['shape' if enabled else 'baseline']=dict(mean_ms=float(np.mean(rows)),p95_ms=float(np.percentile(rows,95)),exact=exact,blocks=[float(np.mean([v['wall_ms'] for v in d['iterations'] if v['phase']=='steady'])) for d in ds],load_s=[d['load_s'] for d in ds],first=[d['first'] for d in ds],profiles=[v['modules'] for d in ds for v in d['iterations'] if v['phase']=='profile'])
(r/'cli_summary.json').write_text(json.dumps(result,indent=2));print({k:{a:b for a,b in v.items() if a!='profiles'} for k,v in result.items()},flush=True)
