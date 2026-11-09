import json,hashlib
from pathlib import Path
import numpy as np
root=Path('records/speed_final_20261103')
reports=[json.loads(p.read_text()) for p in sorted(root.glob('benchmark_*/benchmark.json'))]
summary={'profiles':{},'validation':{}}
for profile in ('accuracy','speed'):
 rs=[r for r in reports if r['inference_profile']==profile];assert len(rs)==2 and all(r['status']=='completed' for r in rs)
 times=[i['wall_ms'] for r in rs for i in r['iterations'] if i['phase']=='steady'];modules=[i['modules'] for r in rs for i in r['iterations'] if i['phase']=='profile']
 summary['profiles'][profile]={'mean_ms':float(np.mean(times)),'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'steady_runs':len(times),'load_s':[r['load_s'] for r in rs],'first_ms':[r['first']['mean_ms'] for r in rs],'modules_ms':{k:float(np.mean([m[k] for m in modules])) for k in modules[0]},'quantization':rs[0]['quantization'],'speed_options':rs[0]['speed_options'],'memory':[r['memory'] for r in rs]}
a=summary['profiles']['accuracy']['mean_ms'];b=summary['profiles']['speed']['mean_ms'];summary.update(latency_reduction_pct=(a-b)/a*100,speedup=a/b)
old=Path('records/final_validation_20261103');new=root/'validation_accuracy';files=sorted(new.glob('*.npy'));assert len(files)==162
check=[]
for p in files:
 q=old/p.name;assert q.exists(),q
 x,y=np.load(p),np.load(q);assert x.shape==y.shape
 check.append(dict(name=p.name,exact=x.tobytes()==y.tobytes(),max_abs=float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64))))))
summary['validation']['accuracy']={'files':len(check),'all_exact':all(x['exact'] for x in check),'max_abs':max(x['max_abs'] for x in check)}
assert summary['validation']['accuracy']['all_exact']
rows=[];deltas=[]
for p in sorted((root/'validation_speed').glob('*_action.npy')):
 x=np.load(p);y=np.load(new/p.name);delta=x.astype(np.float64)-y.astype(np.float64);assert np.isfinite(x).all();deltas.append(delta)
 rows.append(dict(case=p.stem,max_abs=float(np.max(np.abs(delta))),mae=float(np.mean(np.abs(delta))),rmse=float(np.sqrt(np.mean(delta**2)))))
delta=np.stack(deltas);tokens=list((root/'validation_speed').glob('*_token_ids.npy'));assert all(np.array_equal(np.load(p),np.load(new/p.name)) for p in tokens)
summary['validation']['speed']={'case_count':len(rows),'finite':True,'tokens_exact':True,'max_abs':float(np.max(np.abs(delta))),'mae':float(np.mean(np.abs(delta))),'rmse':float(np.sqrt(np.mean(delta**2))),'joint_max_abs':np.max(np.abs(delta),axis=(0,1)).tolist(),'joint_rmse':np.sqrt(np.mean(delta**2,axis=(0,1))).tolist(),'cases':rows}
summary['validation']['dynamic_shapes']=json.loads((root/'dynamic_shapes/results.json').read_text())
summary['source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('imagewam_mlx').glob('*.py')}
(root/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ('validation','source_hashes')},indent=2))
