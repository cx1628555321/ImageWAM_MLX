import subprocess,sys,json
from pathlib import Path
import numpy as np
r=Path('records/int8_pipeline_20261105');order=[('baseline','confirm_a1'),('linear2','confirm_b1'),('linear2','confirm_b2'),('baseline','confirm_a2')]
for mode,label in order:
 with (r/(label+'.log')).open('x') as log:subprocess.run([sys.executable,'-u','experiments/int8_pipeline_20261105/worker.py',mode,label],stdout=log,stderr=subprocess.STDOUT,check=True)
 print(label,flush=True)
ref=np.load(r/'abba/confirm_a1.npz');result={}
for mode in ['baseline','linear2']:
 rows=[];blocks=[];cpu=[]
 for m,l in order:
  if m!=mode:continue
  d=json.loads((r/'abba'/(l+'.json')).read_text());rows+=d['samples_ms'];blocks.append(d['mean_ms']);cpu.append(d['process_cpu_ms_per_request'])
  a=np.load(r/'abba'/(l+'.npz'));assert all(np.array_equal(a[k],ref[k]) for k in ['action','normalized'])
 result[mode]=dict(mean_ms=float(np.mean(rows)),p95_ms=float(np.percentile(rows,95)),blocks=blocks,process_cpu_ms=cpu,exact=True)
(r/'confirm_summary.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
