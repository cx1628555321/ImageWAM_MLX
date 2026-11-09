"""Serial ABBA performance, full offline validation, and CLI smoke runs."""
import subprocess,sys
from pathlib import Path
root=Path('records/speed_final_20261103');root.mkdir(exist_ok=False)
def run(name,args):
 with (root/(name+'.log')).open('w') as f:
  subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name,'completed',flush=True)
for i,profile in enumerate(('accuracy','speed','speed','accuracy')):
 name=f'benchmark_{i}_{profile}'
 run(name,['scripts/benchmark.py','--profile',profile,'--reference','records/final_default_20261103','--output',str(root/name),'--iterations','100','--warmup','8','--profile-iterations','10'])
for profile in ('accuracy','speed'):
 run('validation_'+profile,['scripts/validate_cases.py','--profile',profile,'--output',str(root/('validation_'+profile))])
 run('cli_'+profile,['scripts/infer.py',*(['--profile','speed'] if profile=='speed' else []),'--inputs','examples/robotwin_ep0_frame0/inputs.npz','--prompt-file','examples/robotwin_ep0_frame0/prompt.txt','--noise','examples/robotwin_ep0_frame0/initial_noise_fp32.npy','--output',str(root/('cli_'+profile))])
run('dynamic_shapes',['scripts/check_speed_dynamic.py'])
