import os,sys,subprocess
from pathlib import Path
root=Path('records/graph_dispatch_20261105')
configs=[('native','inherit'),('native','initiated'),('partial_bound','initiated'),('native','interactive'),('partial_bound','interactive'),('native','inherit')]
for i,(mode,qos) in enumerate(configs):
 label=f'{i:02d}_{mode}_{qos}';env=dict(os.environ,MLX_MAX_OPS_PER_BUFFER='256',IMAGEWAM_TEST_QOS=qos)
 with (root/('qos_'+label+'.log')).open('w') as f:
  subprocess.run([sys.executable,'experiments/graph_dispatch_20261105/qos_run.py',mode,label],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(label,flush=True)
