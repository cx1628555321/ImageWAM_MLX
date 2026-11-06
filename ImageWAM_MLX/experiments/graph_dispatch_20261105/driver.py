import os,sys,subprocess,json
from pathlib import Path
root=Path('records/graph_dispatch_20261105');root.mkdir(exist_ok=False)
configs=[('native',256),('rewrap',256),('partial_bound',256),('partial',256),('partial_bound',128),('partial_bound',512),('native',256)]
for i,(mode,ops) in enumerate(configs):
 label=f'{i:02d}_{mode}_{ops}';env=dict(os.environ,MLX_MAX_OPS_PER_BUFFER=str(ops))
 with (root/(label+'.log')).open('w') as f:
  subprocess.run([sys.executable,'experiments/graph_dispatch_20261105/run.py',mode,label],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(label,flush=True)
