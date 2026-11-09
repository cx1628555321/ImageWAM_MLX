import subprocess,sys
from pathlib import Path
root=Path('records/next_20261104/screen');root.mkdir(exist_ok=False)
for i,name in enumerate(['baseline','workspace','qwen_compile','visual_addmm','vae_nhwc','baseline']):
 label=f'{i:02d}_{name}'
 with (root/(label+'.log')).open('w') as f:
  r=subprocess.run([sys.executable,'experiments_next_20261104/bench.py','--case',name,'--iterations','20','--output',str(root/label)],stdout=f,stderr=subprocess.STDOUT)
 print(label,r.returncode,flush=True)
