import os,sys,subprocess,json,time
from pathlib import Path
root=Path('records/runtime_tuning_20261104/reverse');root.mkdir(parents=True,exist_ok=False)
configs=[('baseline_start',{}),('fast_sync',{'MLX_METAL_FAST_SYNCH':'1'}),('ops16',{'MLX_MAX_OPS_PER_BUFFER':'16'}),('ops64',{'MLX_MAX_OPS_PER_BUFFER':'64'}),('ops256',{'MLX_MAX_OPS_PER_BUFFER':'256'}),('baseline_mid',{}),('mb64',{'MLX_MAX_MB_PER_BUFFER':'64'}),('mb256',{'MLX_MAX_MB_PER_BUFFER':'256'}),('bfs8',{'MLX_BFS_MAX_WIDTH':'8'}),('bfs40',{'MLX_BFS_MAX_WIDTH':'40'}),('conv_gemm',{'MLX_CONV_WINOGRAD':'0'}),('baseline_end',{})]
keys={'MLX_METAL_FAST_SYNCH','MLX_MAX_OPS_PER_BUFFER','MLX_MAX_MB_PER_BUFFER','MLX_BFS_MAX_WIDTH','MLX_CONV_WINOGRAD'}
rows=[]
for name,config in reversed(configs):
 env={k:v for k,v in os.environ.items() if k not in keys};env.update(config)
 with (root/(name+'.log')).open('w') as log:ret=subprocess.run([sys.executable,'development/legacy_experiments/experiments_runtime_tuning_20261104/worker.py','--output',str(root/name)],env=env,stdout=log,stderr=subprocess.STDOUT)
 row=dict(name=name,config=config,returncode=ret.returncode)
 if ret.returncode==0:row.update(json.loads((root/name/'results.json').read_text()))
 rows.append(row);(root/'summary.json').write_text(json.dumps(rows,indent=2));print(name,ret.returncode,row.get('mean_ms'),flush=True)
