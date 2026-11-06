import os,sys,time,json,subprocess,signal,fcntl,hashlib
from pathlib import Path
ROOT=Path.cwd();R=ROOT/'records/prefix_followup_20261106_device_fix';R.mkdir(parents=True,exist_ok=True);lock=(R/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(R/'runner.pid').write_text(str(os.getpid()))
PYTHON=str(ROOT/'.venv/bin/python');WORKER='experiments/background_20261105/worker.py';state=dict(status='running',pid=os.getpid(),started=time.strftime('%Y-%m-%d %H:%M:%S'),jobs=[],active=None);records=[]
def save():
 state['updated']=time.strftime('%Y-%m-%d %H:%M:%S');p=R/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(R/'status.json')
 lines=['# 后台优化验证进度与汇总','','状态：'+state['status'],'更新时间：'+state['updated'],'','| 任务 | 状态 | 端到端 ms |','|---|---|---:|']
 for j in state['jobs']:lines.append('| '+j['name']+' | '+j['status']+' | '+str(j.get('mean_ms','—'))+' |')
 lines+=['','## 数值与模块统计','','| 任务 | 前缀 ms | 动作 ms | 标准参考 RMSE | 51 案例最大差（相对 stock） |','|---|---:|---:|---:|---:|']
 for j in state['jobs']:
  rp=Path(j.get('path',''))/'results.json'
  if not rp.exists():continue
  d=json.loads(rp.read_text());m=d.get('modules',{});v=d.get('validation',[])
  lines.append('| '+j['name']+' | '+str(round(m.get('visual_prefill',0),3))+' | '+str(round(m.get('action_denoise',0),3))+' | '+str(d.get('supplied_reference_error',{}).get('rmse','—'))+' | '+str(max((x['max_abs'] for x in v),default=0))+' |')
 lines+=['','默认推理代码未自动切换。完整结果、误差与环境见各任务 results.json / 日志。','profile/toolchain.json 说明逐内核时间线是否具备条件；命令缓冲时间不等于逐内核时间。']
 (R/'REPORT.md').write_text('\n'.join(lines))
def busy(child=None):
 rows=[]
 for line in subprocess.check_output(['ps','-Ao','pid=,ppid=,args='],text=True).splitlines():
  parts=line.strip().split(None,2)
  if len(parts)==3:rows.append((int(parts[0]),int(parts[1]),parts[2]))
 own={os.getpid()}
 if child:own.add(child)
 for _ in range(20):
  new=own|{pid for pid,ppid,c in rows if ppid in own}
  if new==own:break
  own=new
 return [dict(pid=pid,command=c[:300]) for pid,ppid,c in rows if pid not in own and 'python' in c.lower() and not any(s in c for s in ('vscode','pylance','python-env-tools','sshd','zsh -c'))]
def stop():
 if (R/'STOP').exists():state['status']='stopped';save();sys.exit(0)
def job(name,cmd,env=None,timeout=1800,result=None):
 for attempt in range(1,4):
  idle=0
  while idle<3:
   stop();b=busy();idle=0 if b else idle+1;state.update(active=name,status='waiting_for_idle' if b else 'running',competing=b);save()
   if idle<3:time.sleep(10)
  output=R/(name+f'_attempt{attempt}');run=list(cmd)
  if result=='worker':run.append(str(output))
  j=dict(name=name,attempt=attempt,command=run,status='running',path=str(output));state['jobs'].append(j);save()
  with (R/(name+f'_attempt{attempt}.log')).open('w') as f:
   p=subprocess.Popen(run,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,**(env or {})),start_new_session=True);t=time.monotonic();interrupted=False
   while p.poll() is None:
    time.sleep(5)
    b=busy(p.pid)
    if b or time.monotonic()-t>timeout or (R/'STOP').exists():
     os.killpg(p.pid,signal.SIGTERM)
     try:p.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
     interrupted=bool(b);j['status']='interrupted_contention' if b else 'timed_out_or_stopped';j['competing']=b;break
   if j['status']=='running':j['status']='completed' if p.returncode==0 else 'failed'
   j['returncode']=p.returncode;j['elapsed_s']=time.monotonic()-t
  if (output/'results.json').exists():
   d=json.loads((output/'results.json').read_text());j['mean_ms']=d.get('steady',{}).get('mean_ms');j['result_status']=d['status']
   if d['status']!='completed' and j['status']=='completed':j['status']='failed'
  save();stop()
  if interrupted:continue
  return j
 return j
save()
try:
 py=str(ROOT/'records/background_20261106_xcode/builds/route4796/venv/bin/python');WORKER='experiments/prefix_followup_20261106/worker.py'
 j=job('quantizer_check',[py,'experiments/prefix_followup_20261106/quantizer_check.py'])
 if j['status']!='completed':raise RuntimeError('Quantizer exactness check failed')
 b=dict(status='completed',mean_ms=149.934789177496,path=str(ROOT/'records/prefix_followup_20261106_repaired/baseline_attempt1'));env={'BASE_RUN':b['path']}
 if b['status']!='completed':raise RuntimeError('Baseline failed')
 c=job('slim_quant',[py,WORKER,'slim_quant'],env=env,result='worker')
 if c['status']=='completed' and c['mean_ms']<b['mean_ms']*0.995:
  for i,mode in enumerate(['baseline','slim_quant','slim_quant','baseline']):job('confirm_'+str(i)+'_'+mode,[py,WORKER,mode],env=env,result='worker')
 state['status']='completed' if all(j['status']=='completed' for j in state['jobs']) else 'finished_with_failures';state['active']=None;save()
except BaseException as e:
 if not isinstance(e,SystemExit):state.update(status='failed',error=repr(e));save();raise
