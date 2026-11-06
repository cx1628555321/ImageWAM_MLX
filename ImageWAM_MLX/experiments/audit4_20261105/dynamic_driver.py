from pathlib import Path
import sys,subprocess
r=Path('records/audit4_20261105')
for mode in ('baseline','prewarmed','shapeless_step','shapeless_prefill'):
 with (r/(mode+'.log')).open('x') as f:subprocess.run([sys.executable,'-u','experiments/audit4_20261105/dynamic.py',mode],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',mode,flush=True)
with (r/'export.log').open('x') as f:subprocess.run([sys.executable,'-u','experiments/audit4_20261105/export_test.py'],stdout=f,stderr=subprocess.STDOUT,check=True)
