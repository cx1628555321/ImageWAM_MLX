import subprocess,sys
from pathlib import Path
root=Path('records/lowbit_20261104/e2e');root.mkdir(exist_ok=False)
for name in ('accuracy','current_speed','fast_vae','native_action','fused_action','native_visual','fused_visual','native_all','fused_all','int8_kv','int8_conv','all_lowbit'):
 with (root/(name+'.log')).open('w') as f:
  p=subprocess.run([sys.executable,'experiments_lowbit_20261104/e2e_case.py','--case',name,'--output',str(root/name)],stdout=f,stderr=subprocess.STDOUT)
 print(name,p.returncode,flush=True)
