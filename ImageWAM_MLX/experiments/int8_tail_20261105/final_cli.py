import subprocess,sys,json
from pathlib import Path
import numpy as np
root=Path('records/int8_tail_20261105');stats={};reference=None;commands=[]
for mode,label in [('old','cli_a1'),('new','cli_b1'),('new','cli_b2'),('old','cli_a2')]:
 out=root/label
 cmd=[sys.executable,'-u','scripts/benchmark.py','--profile','speed','--reference','examples/robotwin_ep0_frame0','--output',str(out),'--warmup','10','--iterations','100','--profile-iterations','5']
 if mode=='old':cmd+=['--no-tail-visual']
 commands.append(cmd)
 with (root/(label+'.log')).open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 report=json.loads((out/'benchmark.json').read_text());assert report['status']=='completed'
 if reference is None:reference=(np.load(out/'actions_steady_000.npy'),np.load(out/'normalized_steady_000.npy'))
 for i in range(100):
  assert np.array_equal(np.load(out/f'actions_steady_{i:03d}.npy'),reference[0])
  assert np.array_equal(np.load(out/f'normalized_steady_{i:03d}.npy'),reference[1])
 stats[label]=dict(mode=mode,load_s=report['load_s'],first=report['first'],steady=report['steady'],modules={k:float(np.mean([x['modules'][k] for x in report['iterations'] if x['phase']=='profile'])) for k in next(x['modules'] for x in report['iterations'] if x['phase']=='profile')},all_exact=True)
 print(label,stats[label],flush=True)
(root/'cli_summary.json').write_text(json.dumps(dict(blocks=stats,commands=commands),indent=2))
