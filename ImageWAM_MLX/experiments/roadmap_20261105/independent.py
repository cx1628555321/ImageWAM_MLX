import sys,time,json,subprocess
from pathlib import Path
import numpy as np
r=Path('records/roadmap_20261105');screen=json.loads((r/'screen/performance.json').read_text());valid=json.loads((r/'screen/validation.json').read_text());dyn=json.loads((r/'screen/dynamic.json').read_text())
accepted=[k for k in screen if k!='baseline' and screen[k]['exact'] and all(v['errors'][k]['exact'] for v in valid) and all(v['checks'][k]['action_exact'] and all(v['checks'][k]['trace_exact'].values()) for v in dyn)]
# Confirm one best valid candidate independently, even if tiny local gain is inconclusive.
candidate=min(accepted,key=lambda k:screen[k]['mean_ms']) if accepted else None
(r/'selection.json').write_text(json.dumps(dict(valid_candidates=accepted,selected=candidate),indent=2));assert candidate
rng=np.random.default_rng(5116);seq=[]
for i in range(6):
 order=['baseline',candidate]
 if rng.integers(2):order.reverse()
 seq.extend((k,f'{i}_{k}') for k in order)
(r/'process_order.json').write_text(json.dumps(seq,indent=2))
with (r/'process_snapshots.jsonl').open('x') as snapshots:
 for mode,label in seq:
  with (r/(label+'.log')).open('x') as log:
   p=subprocess.Popen([sys.executable,'-u','experiments/roadmap_20261105/worker.py',mode,label],stdout=log,stderr=subprocess.STDOUT)
   while p.poll() is None:
    snapshots.write(json.dumps(dict(time=time.time(),label=label,child_pid=p.pid,ps=subprocess.run(['ps','-Ao','pid,ppid,pcpu,rss,comm'],capture_output=True,text=True).stdout))+'\n');snapshots.flush();time.sleep(5)
   if p.returncode:raise RuntimeError((label,p.returncode))
  print('DONE',label,flush=True)
result={};ref=np.load(r/'independent'/('0_baseline.npz'))['actions'][0]
for mode in ['baseline',candidate]:
 ds=[json.loads((r/'independent'/(label+'.json')).read_text()) for m,label in seq if m==mode];rows=[x for d in ds for x in d['samples_ms']];errors=[]
 for m,label in seq:
  if m==mode:
   a=np.load(r/'independent'/(label+'.npz'))['actions'];errors.append(dict(label=label,all_exact=bool(np.all(a==ref)),max_abs=float(np.abs(a-ref).max())))
 result[mode]=dict(mean_ms=float(np.mean(rows)),p50_ms=float(np.median(rows)),p95_ms=float(np.percentile(rows,95)),blocks=[d['mean_ms'] for d in ds],cpu=[d['process_cpu_ms'] for d in ds],load_ms=[d['load_ms'] for d in ds],first_ms=[d['first_ms'] for d in ds],errors=errors,profiles=[d['profiles'] for d in ds])
d=np.array(result['baseline']['blocks'])-result[candidate]['blocks'];ci=np.percentile(np.mean(d[rng.integers(0,len(d),(10000,len(d)))],axis=1),[2.5,97.5]);result['comparison']=dict(candidate=candidate,paired_process_saving_ms=d.tolist(),saving_ms=float(d.mean()),process_bootstrap_ci95=ci.tolist(),promote=bool(d.mean()>=max(2,.01*result['baseline']['mean_ms']) and ci[0]>0 and result[candidate]['p95_ms']<=result['baseline']['p95_ms'] and all(v['all_exact'] for v in result[candidate]['errors'])))
(r/'independent_summary.json').write_text(json.dumps(result,indent=2));print(result['comparison'],flush=True)
