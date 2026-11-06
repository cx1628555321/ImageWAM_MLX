import sys,os,subprocess,time,json
from pathlib import Path
r=Path('records/precision_all_20261105');r.mkdir(exist_ok=False)
modes=['baseline','visual_mlp','visual_all','fixed_scale','action_w2','action_w4','action_int8','partial_half','partial_bfloat','kv_int8'];summary=[]
def busy(child=None):
 out=[]
 for line in subprocess.check_output(['ps','-Ao','pid=,args='],text=True).splitlines():
  pid,cmd=line.strip().split(None,1)
  if int(pid) in (os.getpid(),child):continue
  if any(x in cmd for x in ('vscode','pylance','python-env-tools','zsh -c','sshd')):continue
  if 'python' in cmd:out.append(cmd)
 return out
base=None
for mode in modes:
 attempt=0
 while True:
  idle=0
  while idle<3:
   b=busy();idle=0 if b else idle+1
   if b:print('WAIT',b,flush=True)
   if idle<3:time.sleep(15)
  attempt+=1;out=r/f'{mode}_{attempt}';env=dict(os.environ)
  if base:env['BASE_RUN']=str(base)
  with (r/f'{mode}_{attempt}.log').open('w') as f:
   p=subprocess.Popen([sys.executable,'experiments/precision_all_20261105/worker.py',mode,str(out)],env=env,stdout=f,stderr=subprocess.STDOUT);start=time.monotonic();interrupted=False
   print('START',mode,p.pid,flush=True)
   while p.poll() is None:
    time.sleep(15);b=busy(p.pid)
    if b:
     p.terminate();p.wait();interrupted=True;print('CONTENTION',mode,b,flush=True);break
    if time.monotonic()-start>1800:
     p.terminate();p.wait();print('TIMEOUT',mode,flush=True);break
  if interrupted:continue
  result=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else {'status':'failed'}
  summary.append(dict(mode=mode,path=str(out),status=result['status'],steady=result.get('steady',{}).get('mean_ms'),error=result.get('error')))
  (r/'summary.json').write_text(json.dumps(summary,indent=2));print('RESULT',summary[-1],flush=True)
  if mode=='baseline':
   if result['status']!='completed':sys.exit('baseline failed')
   base=out
  break
print('ALL DONE',flush=True)
