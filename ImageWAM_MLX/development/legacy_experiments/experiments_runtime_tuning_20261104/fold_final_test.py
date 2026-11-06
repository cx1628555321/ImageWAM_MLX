import os,sys,json,subprocess
from pathlib import Path
root=Path('records/staged_fold_20261104/final_cli');root.mkdir(exist_ok=False)
keys={'MLX_METAL_FAST_SYNCH','MLX_MAX_OPS_PER_BUFFER','MLX_MAX_MB_PER_BUFFER','MLX_BFS_MAX_WIDTH','MLX_CONV_WINOGRAD'}
rows=[]
for i,name in enumerate(['old','new','new','old']):
 label=f'{i}_{name}';env={k:v for k,v in os.environ.items() if k not in keys};cmd=[sys.executable,'scripts/benchmark.py','--profile','speed','--reference','examples/robotwin_ep0_frame0','--output',str(root/label),'--warmup','4','--iterations','40','--profile-iterations','3']
 if name=='old':cmd+=['--no-prepacked-visual']
 with (root/(label+'.log')).open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0,label
 data=json.loads((root/label/'benchmark.json').read_text());rows.append(dict(label=label,load_s=data['load_s'],first=data['first'],steady=data['steady'],staged_visual=data['staged_visual'],max_ops_per_buffer=data['max_ops_per_buffer']));(root/'summary.json').write_text(json.dumps(rows,indent=2));print(rows[-1],flush=True)
