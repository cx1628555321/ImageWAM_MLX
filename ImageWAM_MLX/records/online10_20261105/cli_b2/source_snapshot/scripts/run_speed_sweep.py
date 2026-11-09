import subprocess,json,sys
from pathlib import Path
root=Path('records/speed_sweep_20261103');root.mkdir(exist_ok=False)
modes=['accuracy','w1','w2','w3','w4','w6','w8','mxfp4','mxfp8','nvfp4','mxfp8a','nvfp4a']
for mode in modes:
 policy='accuracy' if mode=='accuracy' else json.dumps(dict.fromkeys(('visual','action','qwen'),mode))
 with (root/(mode+'.log')).open('w') as f:
  result=subprocess.run([sys.executable,'scripts/sweep_quantized_speed.py','--policy',policy,'--output',str(root/mode)],stdout=f,stderr=subprocess.STDOUT)
 print(mode,result.returncode,flush=True)
