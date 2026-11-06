import subprocess,sys,json
from pathlib import Path
import numpy as np
r=Path('records/remaining_20261105');summary={}
configs=[('base_first',[],'baseline'),('cache',['--text-cache-size','8'],'cache_hit'),('steps4',['--steps','4'],'steps4'),('cache_steps4',['--steps','4','--text-cache-size','8'],'cache_steps4'),('base_last',[],'baseline')]
for label,flags,mode in configs:
 out=r/('cli_'+label);cmd=[sys.executable,'-u','scripts/benchmark.py','--profile','speed','--reference','examples/robotwin_ep0_frame0','--output',str(out),'--warmup','8','--iterations','60','--profile-iterations','3',*flags]
 with (r/('cli_'+label+'.log')).open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 d=json.loads((out/'benchmark.json').read_text());assert d['status']=='completed';ref=np.load(r/'independent'/('forward_'+mode+'.npz'))
 for i in range(60):
  assert np.array_equal(np.load(out/f'actions_steady_{i:03d}.npy'),ref['action']),label
  assert np.array_equal(np.load(out/f'normalized_steady_{i:03d}.npy'),ref['normalized']),label
 summary[label]=dict(command=cmd,steady=d['steady'],load_s=d['load_s'],first=d['first'],text_cache=d['text_cache'],comparison=d['comparison'],modules={k:float(np.mean([x['modules'][k] for x in d['iterations'] if x['phase']=='profile'])) for k in next(x['modules'] for x in d['iterations'] if x['phase']=='profile')},matches_experiment=True)
 print(label,summary[label]['steady'],flush=True)
(r/'cli_summary.json').write_text(json.dumps(summary,indent=2))
