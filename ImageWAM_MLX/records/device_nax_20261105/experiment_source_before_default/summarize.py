import json,hashlib
from pathlib import Path
import numpy as np
r=Path('records/device_nax_20261105');result={}
for folder in ('screen','screen2'):
 d=json.loads((r/folder/'results.json').read_text());v=json.loads((r/folder/'validation.json').read_text())
 assert all(e['exact'] for c in v for e in c['errors'].values())
 result[folder]={k:{a:b for a,b in x.items() if a!='samples_ms'} for k,x in d.items()}
 result[folder+'_exact_cases']=len(v)
result['independent']=json.loads((r/'independent_summary.json').read_text())
for mode in result['independent']:
 assert all(x['exact'] for x in result['independent'][mode]['errors'])
dyn=json.loads((r/'dynamic/results.json').read_text())
assert all(e['action_exact'] and e['latent_exact'] and e['preprocess_exact'] for d in dyn for e in d['errors'].values())
result['dynamic']=dyn
cli=json.loads((r/'cli_summary.json').read_text());result['cli_blocks']=cli['blocks'];result['cli']={}
for mode in ('old','new'):
 ds=[json.loads((r/label/'benchmark.json').read_text()) for label,b in cli['blocks'].items() if b['mode']==mode]
 rows=[i['wall_ms'] for d in ds for i in d['iterations'] if i['phase']=='steady']
 profiles=[i['modules'] for d in ds for i in d['iterations'] if i['phase']=='profile']
 result['cli'][mode]=dict(mean_ms=float(np.mean(rows)),p95_ms=float(np.percentile(rows,95)),count=len(rows),modules={k:float(np.mean([p[k] for p in profiles])) for k in profiles[0]})
result['saving_ms']=result['cli']['old']['mean_ms']-result['cli']['new']['mean_ms']
result['reduction_percent']=100*result['saving_ms']/result['cli']['old']['mean_ms']
result['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('imagewam_mlx','experiments/device_nax_20261105') for p in Path(folder).glob('*.py')}
(r/'summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:result[k] for k in ('cli','saving_ms','reduction_percent')},indent=2))
