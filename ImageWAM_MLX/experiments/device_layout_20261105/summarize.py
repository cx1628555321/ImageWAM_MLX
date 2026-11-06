from pathlib import Path
import json,hashlib
import numpy as np
r=Path('records/device_layout_20261105');out={}
for folder in ('screen','rounding','fixed','group2'):
 d=json.loads((r/folder/'results.json').read_text());out[folder]={k:{a:b for a,b in v.items() if a!='samples_ms'} for k,v in d.items()}
 p=r/folder/'validation.json'
 if p.exists():
  c=json.loads(p.read_text());out[folder+'_validation']={'count':len(c),'exact_counts':{m:sum(x['errors'][m]['exact'] for x in c) for m in c[0]['errors']}}
 else:out[folder+'_validation']={'complete':False,'reason':'Stopped after integer-accumulator mismatch established; obsolete rounding hypothesis'}
for name in ('probe','probe_fixed'):
 d=json.loads((r/(name+'.json')).read_text());out[name]={'count':len(d),'integer_exact_count':sum(x['integer_acc_equal'] for x in d),'all_float_exact':all(all(e['equal'] for e in x['errors'].values()) for x in d)}
out['independent']=json.loads((r/'independent_summary.json').read_text());out['dynamic']=json.loads((r/'dynamic/results.json').read_text())
assert all(e['action_exact'] for c in out['dynamic'] for e in c['errors'].values())
c=json.loads((r/'cli_summary.json').read_text());out['cli_blocks']=c['blocks'];out['cli']={}
for mode in ('0','32','64'):
 ds=[json.loads((r/l/'benchmark.json').read_text()) for l,b in c['blocks'].items() if b['mode']==mode]
 samples=[x['wall_ms'] for d in ds for x in d['iterations'] if x['phase']=='steady'];profiles=[x['modules'] for d in ds for x in d['iterations'] if x['phase']=='profile']
 out['cli'][mode]={'mean_ms':float(np.mean(samples)),'p95_ms':float(np.percentile(samples,95)),'count':len(samples),'modules':{k:float(np.mean([p[k] for p in profiles])) for k in profiles[0]}}
for mode in ('32','64'):
 x=out['cli'][mode];x['saving_ms']=out['cli']['0']['mean_ms']-x['mean_ms'];x['reduction_percent']=100*x['saving_ms']/out['cli']['0']['mean_ms']
out['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for d in ('imagewam_mlx','experiments/device_layout_20261105') for p in Path(d).glob('*.py')}
(r/'summary.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out['cli'],indent=2))
