import subprocess,sys
from pathlib import Path
root=Path('records/tiles_20261104/e2e');root.mkdir(exist_ok=False)
for i,name in enumerate(['current_speed','tile32k128','tile32k512','tile64n64','current_speed']):
 label=f'{i:02d}_{name}'
 with (root/(label+'.log')).open('w') as f:
  r=subprocess.run([sys.executable,'development/legacy_experiments/experiments_tiles_20261104/e2e.py','--case',name,'--iterations','50','--output',str(root/label)],stdout=f,stderr=subprocess.STDOUT)
 print(label,r.returncode,flush=True)
