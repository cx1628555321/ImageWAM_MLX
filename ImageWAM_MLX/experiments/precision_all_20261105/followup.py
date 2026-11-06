import os,subprocess,time,json,sys
from pathlib import Path
r=Path('records/precision_followup_20261105');r.mkdir(exist_ok=False);env=dict(os.environ,BASE_RUN='records/precision_all_20261105/baseline_1');jobs=[('fixed_scale',[sys.executable,'experiments/precision_all_20261105/worker.py','fixed_scale',str(r/'fixed_scale')]),('kv_unit',[sys.executable,'experiments/precision_all_20261105/kv_unit.py',str(r/'kv_unit.json')])]
for i,mode in enumerate(['default','all_int8','all_int8','default']):
 name=f'confirm_{i}_{mode}';jobs.append((name,[sys.executable,'scripts/benchmark.py','--profile','speed','--prefix-precision',mode,'--text-cache-size','1','--reference','examples/robotwin_ep0_frame0','--warmup','10','--iterations','50','--profile-iterations','50','--steps','10','--output',str(r/name)]))
summary=[]
for name,cmd in jobs:
 # Execute after main suite has exited. Do not benchmark alongside another Python model.
 def others(child=None):
  return [l for l in subprocess.check_output(['ps','-Ao','pid=,args='],text=True).splitlines() if 'python' in l and int(l.split()[0]) not in (os.getpid(),child) and not any(s in l for s in ('vscode','pylance','zsh -c','sshd','python-env-tools'))]
 while others():print('WAIT',others(),flush=True);time.sleep(15)
 with (r/(name+'.log')).open('w') as f:
  p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);print('START',name,flush=True)
  while p.poll() is None:
   time.sleep(15)
   if others(p.pid):p.terminate();p.wait();raise RuntimeError('Contended; rerun this job in new output directory')
 summary.append(dict(name=name,returncode=p.returncode));(r/'summary.json').write_text(json.dumps(summary,indent=2));print('RESULT',summary[-1],flush=True)
print('ALL DONE',flush=True)
