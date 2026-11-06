from pathlib import Path
import json,hashlib
import numpy as np
r=Path('records/action_condition_20261105');out={}
for folder in ('screen','prepare','attention'):
 d=json.loads((r/folder/'results.json').read_text());out[folder]={k:{a:b for a,b in x.items() if a!='samples_ms'} for k,x in d.items()}
 v=json.loads((r/folder/'validation.json').read_text());out[folder+'_validation']={'count':len(v),'exact_counts':{k:sum(c['errors'][k]['exact'] for c in v) for k in v[0]['errors']}}
out['independent']=json.loads((r/'independent_summary.json').read_text())
for mode in ('baseline','unrolled'):
 ds=[json.loads(p.read_text()) for p in (r/'independent').glob('*.json') if json.loads(p.read_text())['mode']==mode]
 profiles=[x for d in ds for x in d['profiles']]
 out['independent'][mode]['modules']={k:float(np.mean([p[k] for p in profiles])) for k in profiles[0]}
out['dynamic']=json.loads((r/'dynamic/results.json').read_text());out['trace_comparison']=json.loads((r/'dynamic/trace_comparison.json').read_text())
assert all(e['action_exact'] and e['latent_exact'] and e['preprocess_exact'] for x in out['dynamic'] for e in x['errors'].values())
assert all(all(v.values()) for v in out['trace_comparison'].values())
out['default_changed']=False
out['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ('imagewam_mlx','experiments/action_condition_20261105') for p in Path(directory).glob('*.py')}
(r/'summary.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out['independent'],indent=2))
