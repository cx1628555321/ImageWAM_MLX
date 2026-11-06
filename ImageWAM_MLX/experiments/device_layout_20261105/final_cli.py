import subprocess,sys,json
from pathlib import Path
import numpy as np
root=Path('records/device_layout_20261105');stats={};reference=None;commands=[]
for mode,label in [('0','cli_a1'),('32','cli_b1'),('64','cli_c1'),('64','cli_c2'),('32','cli_b2'),('0','cli_a2')]:
 out=root/label
 cmd=[sys.executable,'-u','scripts/benchmark.py','--profile','speed','--text-cache-size','0','--steps','10','--reference','examples/robotwin_ep0_frame0','--output',str(out),'--warmup','30','--iterations','100','--profile-iterations','5']
 cmd+=['--visual-group-rows',mode]
 commands.append(cmd)
 with (root/(label+'.log')).open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 report=json.loads((out/'benchmark.json').read_text());assert report['status']=='completed'
 assert report['steps']==10 and not report['text_cache']['enabled']
 assert report['device_visual']
 assert report['visual_group_rows']==int(mode)
 assert not report['vae_asymmetric']
 if reference is None:reference=(np.load(out/'actions_steady_000.npy'),np.load(out/'normalized_steady_000.npy'))
 for i in range(100):
  assert np.array_equal(np.load(out/f'actions_steady_{i:03d}.npy'),reference[0])
  assert np.array_equal(np.load(out/f'normalized_steady_{i:03d}.npy'),reference[1])
 stats[label]=dict(mode=mode,load_s=report['load_s'],first=report['first'],steady=report['steady'],modules={k:float(np.mean([x['modules'][k] for x in report['iterations'] if x['phase']=='profile'])) for k in next(x['modules'] for x in report['iterations'] if x['phase']=='profile')},all_exact=True)
 print(label,stats[label],flush=True)
(root/'cli_summary.json').write_text(json.dumps(dict(blocks=stats,commands=commands),indent=2))
