import sys,subprocess,time,json,datetime
from pathlib import Path
import numpy as np
from common import MODES
r=Path('records/remaining_20261105');order=MODES+['cache_steps4']
seq=[(mode,'forward_'+mode) for mode in order]+[(mode,'reverse_'+mode) for mode in reversed(order)]
with (r/'process_snapshots.jsonl').open('x') as snapshots:
 for mode,label in seq:
  with (r/(label+'.log')).open('x') as log:
   p=subprocess.Popen([sys.executable,'-u','experiments/remaining_20261105/worker.py',mode,label],stdout=log,stderr=subprocess.STDOUT)
   while p.poll() is None:
    snapshots.write(json.dumps(dict(time=datetime.datetime.now().isoformat(),label=label,child_pid=p.pid,ps=subprocess.run(['ps','-Ao','pid,ppid,pcpu,rss,comm'],capture_output=True,text=True).stdout))+'\n');snapshots.flush();time.sleep(5)
   if p.returncode:raise RuntimeError((label,p.returncode))
  print('DONE',label,flush=True)
result={};ref=np.load(r/'independent/forward_baseline.npz')
for mode in order:
 ds=[json.loads((r/'independent'/(label+'.json')).read_text()) for m,label in seq if m==mode];rows=[x for d in ds for x in d['samples_ms']];errs=[]
 for m,label in seq:
  if m!=mode:continue
  a=np.load(r/'independent'/(label+'.npz'));d=a['action']-ref['action'];err=dict(exact=bool(np.array_equal(a['action'],ref['action']) and np.array_equal(a['normalized'],ref['normalized'])),max_abs=float(np.abs(d).max()),rmse=float(np.sqrt(np.mean(d*d))))
  if 'steps' not in mode:assert err['exact'],label
  errs.append(err)
 result[mode]=dict(mean_ms=float(np.mean(rows)),p95_ms=float(np.percentile(rows,95)),blocks=[d['mean_ms'] for d in ds],cpu=[d['process_cpu_ms'] for d in ds],load_ms=[d['load_ms'] for d in ds],first_ms=[d['first_ms'] for d in ds],errors=errs)
(r/'independent_summary.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
