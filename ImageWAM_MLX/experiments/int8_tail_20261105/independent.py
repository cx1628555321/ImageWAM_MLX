import subprocess,sys,json
from pathlib import Path
import numpy as np
root=Path('records/int8_tail_20261105')
sequence=[('baseline','a1'),('all','b1'),('linear2','c1'),('linear2','c2'),('all','b2'),('baseline','a2')]
for mode,label in sequence:
 with (root/(label+'.log')).open('x') as log:
  subprocess.run([sys.executable,'-u','experiments/int8_tail_20261105/worker.py',mode,label],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('DONE',mode,label,flush=True)
ref=np.load(root/'abba/a1.npz');res={}
for mode in ('baseline','all','linear2'):
 rows=[];summaries=[]
 for m,label in sequence:
  if m!=mode:continue
  data=json.loads((root/'abba'/(label+'.json')).read_text());rows+=data['samples_ms'];summaries.append(data)
  out=np.load(root/'abba'/(label+'.npz'))
  assert all(np.array_equal(out[k],ref[k]) for k in ('action','normalized')),label
 res[mode]=dict(mean_ms=float(np.mean(rows)),p95_ms=float(np.percentile(rows,95)),blocks=[s['mean_ms'] for s in summaries],all_exact=True)
(root/'independent_summary.json').write_text(json.dumps(res,indent=2));print(res,flush=True)
