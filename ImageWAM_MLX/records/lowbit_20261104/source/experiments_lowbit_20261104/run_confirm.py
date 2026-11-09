import subprocess,sys
from pathlib import Path
root=Path('records/lowbit_20261104/confirm');root.mkdir(exist_ok=False)
order=['accuracy','current_speed','fused_visual','speed_native_visual','speed_native_visual','fused_visual','current_speed','accuracy']
for i,name in enumerate(order):
 label=f'{i:02d}_{name}'
 with (root/(label+'.log')).open('w') as f:
  p=subprocess.run([sys.executable,'experiments_lowbit_20261104/e2e_case.py','--case',name,'--iterations','50','--output',str(root/label)],stdout=f,stderr=subprocess.STDOUT)
 print(label,p.returncode,flush=True)
