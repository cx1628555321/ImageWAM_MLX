import subprocess,json,sys
from pathlib import Path
root=Path('records/speed_chunk_20261103');root.mkdir(exist_ok=False)
for i,mode in enumerate(('w1m8','w4m8','w1','w4','w4m8','w1m8')):
 with (root/f'{i}_{mode}.log').open('w') as f:
  p=subprocess.run([sys.executable,'scripts/sweep_quantized_speed.py','--policy',json.dumps({'action':mode}),'--options',json.dumps({'vae':'fast_bf16'}),'--output',str(root/f'{i}_{mode}'),'--iterations','40'],stdout=f,stderr=subprocess.STDOUT)
 print(i,mode,p.returncode,flush=True)
