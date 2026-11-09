"""Rebuild today's comparison from saved measurements and exact output checks."""
import json,hashlib
from pathlib import Path
import numpy as np
root=Path('records');out=root/'final_day_20261103';out.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def stats(xs):return dict(mean=float(np.mean(xs)),median=float(np.median(xs)),p95=float(np.percentile(xs,95)),count=len(xs))
r={'pytorch':{},'input_checks':{},'accuracy_checks':[]}
latest=root/'final_default_20261103';b=read(latest/'benchmark.json');assert b['status']=='completed'
r['mlx']={k:v for k,v in b.items() if k!='iterations'}
r['mlx']['modules']={k:stats([x['modules'][k] for x in b['iterations'] if x['phase']=='profile']) for k in next(x['modules'] for x in b['iterations'] if x['phase']=='profile')}
for name,folder in [('base','20261103_102606_verified'),('dynamic','20261103_dynamic_profile')]:
 p=Path('../ImageWAM/evaluate_results/m5pro_robotwin')/folder;j=read(p/'benchmark.json')
 r['pytorch'][name]={'path':str(p),'benchmark_sha256':hashlib.sha256((p/'benchmark.json').read_bytes()).hexdigest(),
 'steady_ms':j['steady_ms'],'first_inference_ms':j['first_inference_ms'],
 'load_ms':next(x['wall_ms'] for x in j['spans'] if x['name']=='model_load_total'),
 'modules':j['modules'],'configuration':j['configuration'],'torch':j['environment']['torch']}
 with np.load(p/'inputs.npz') as a,np.load(latest/'inputs.npz') as z:
  checks={k:np.array_equal(a[k],z[k]) for k in z.files}
 checks['noise']=np.array_equal(np.load(p/'initial_noise_fp32.npy'),np.load(latest/'initial_noise_fp32.npy'))
 checks['prompt']=j['input']['prompt']==read(latest/'input_manifest.json')['prompt']
 r['input_checks'][name]=checks;assert all(checks.values())
 r['pytorch'][name]['relative_speedup']=j['steady_ms']['mean']/b['steady']['mean_ms']
old=read(root/'bf16_final/benchmark.json');r['mlx_early']={k:old[k] for k in ('load_s','first','steady','profile')}
reference=root/'integrated_validation_20261103';current=root/'final_validation_20261103'
v=read(current/'validation.json');assert v['status']=='completed';r['cross_framework_validation']=v['summary']
for f in sorted(current.glob('*.npy')):
 g=reference/f.name;assert g.exists(),g
 a,z=np.load(f),np.load(g)
 r['accuracy_checks'].append({'file':f.name,'bytes_equal':a.dtype==z.dtype and a.shape==z.shape and a.tobytes()==z.tobytes()})
assert len(r['accuracy_checks'])==162
r['all_162_equal']=all(x['bytes_equal'] for x in r['accuracy_checks']);assert r['all_162_equal']
cli=root/'final_cli_20261103';r['cli']=read(cli/'run.json')
r['cli_equal']=np.load(cli/'action.npy').tobytes()==np.load(latest/'reference_action.npy').tobytes();assert r['cli_equal']
steady=[np.load(f) for f in sorted(latest.glob('actions_steady_*.npy'))]
r['steady_outputs_equal']=all(x.tobytes()==steady[0].tobytes() for x in steady)
(out/'summary.json').write_text(json.dumps(r,indent=2))
print(json.dumps({k:v for k,v in r.items() if k not in ('accuracy_checks','pytorch','mlx')},indent=2))
print('FINAL',json.dumps({k:r['mlx'][k] for k in ('load_s','first','steady','modules','memory')},indent=2))
