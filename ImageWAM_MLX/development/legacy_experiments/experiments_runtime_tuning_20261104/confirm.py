import os,sys,subprocess,json
from pathlib import Path
root=Path('records/runtime_tuning_20261104/confirm');root.mkdir(exist_ok=False)
configs={'baseline':{},'ops256':{'MLX_MAX_OPS_PER_BUFFER':'256'},'mb64':{'MLX_MAX_MB_PER_BUFFER':'64'}}
keys={'MLX_METAL_FAST_SYNCH','MLX_MAX_OPS_PER_BUFFER','MLX_MAX_MB_PER_BUFFER','MLX_BFS_MAX_WIDTH','MLX_CONV_WINOGRAD'}
orders=[['baseline','ops256','mb64'],['mb64','ops256','baseline'],['ops256','baseline','mb64'],['mb64','baseline','ops256']];rows=[]
for i,order in enumerate(orders):
 for name in order:
  label=f'{i}_{name}';env={k:v for k,v in os.environ.items() if k not in keys};env.update(configs[name])
  with (root/(label+'.log')).open('w') as log:ret=subprocess.run([sys.executable,'development/legacy_experiments/experiments_runtime_tuning_20261104/worker.py','--output',str(root/label),'--iterations','30'],env=env,stdout=log,stderr=subprocess.STDOUT)
  row=dict(block=i,name=name,returncode=ret.returncode)
  if ret.returncode==0:row.update(json.loads((root/label/'results.json').read_text()))
  rows.append(row);(root/'summary.json').write_text(json.dumps(rows,indent=2));print(label,ret.returncode,row.get('mean_ms'),flush=True)
