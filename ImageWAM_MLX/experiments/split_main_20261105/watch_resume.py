import os,time,json,subprocess,sys
from pathlib import Path
from datetime import datetime
root=Path('records/split_main_watch_'+datetime.now().strftime('%Y%m%d_%H%M%S'));root.mkdir()
log=(root/'watch.jsonl').open('a',buffering=1)
def record(**kw):
 kw['time']=datetime.now().isoformat();log.write(json.dumps(kw)+'\n');print(kw,flush=True)
def busy(exclude=()):
 rows=subprocess.check_output(['ps','-Ao','pid=,pcpu=,args='],text=True).splitlines();out=[]
 for row in rows:
  parts=row.strip().split(None,2)
  if len(parts)<3:continue
  pid,cpu,cmd=parts
  if int(pid) in (os.getpid(),*exclude):continue
  if any(x in cmd for x in ('vscode','pylance','python-env-tools','sshd','zsh -c','watch_resume.py')):continue
  if 'python' in cmd.lower() or float(cpu)>100:out.append(row.strip())
 return out
attempt=0;idle=0
while True:
 b=busy();record(event='idle_check',busy=b)
 if b:idle=0
 else:idle+=1
 if idle<3:time.sleep(15);continue
 attempt+=1;r=root/f'attempt_{attempt}';env=dict(os.environ,WAM_RECORD_DIR=str(r));interrupted=False
 for script in ('test.py','validate.py'):
  with (root/f'attempt_{attempt}_{script}.log').open('w') as f:
   child=subprocess.Popen([sys.executable,'experiments/split_main_20261105/'+script],env=env,stdout=f,stderr=subprocess.STDOUT)
   record(event='started',script=script,pid=child.pid,output=str(r))
   while child.poll() is None:
    time.sleep(15);b=busy((child.pid,));record(event='running_check',script=script,busy=b)
    if b:
     child.terminate()
     try:child.wait(timeout=15)
     except subprocess.TimeoutExpired:child.kill();child.wait()
     interrupted=True;record(event='interrupted_for_contention',script=script);break
   if interrupted:break
   if child.returncode!=0:
    record(event='test_failed',script=script,returncode=child.returncode);sys.exit(1)
 if interrupted:idle=0;continue
 perf=json.loads((r/'performance.json').read_text());validation=json.loads((r/'validation/results.json').read_text());dynamic=json.loads((r/'dynamic.json').read_text())
 summary=dict(output=str(r),performance={k:{a:b for a,b in v.items() if a!='samples_ms'} for k,v in perf.items()},validation_pairs=len(validation),validation_exact=all(v['exact'] for v in validation),dynamic_exact=all(v['exact'] for v in dynamic),monitor_interval_s=15)
 (root/'summary.json').write_text(json.dumps(summary,indent=2))
 lines=['# 空闲窗口续测结果','',f'记录目录：{r}。每15秒检查并发负载；这是进程级监控，不保证没有短于采样间隔的争用。','', '|模式|均值 ms|P95 ms|计时输出一致|','|---|---:|---:|---|']
 for k,v in perf.items():lines.append(f"|{k}|{v['mean_ms']:.3f}|{v['p95_ms']:.3f}|{v['exact']}|")
 lines+=['',f"回归比较 {len(validation)} 组，全部逐值一致：{summary['validation_exact']}。动态测试通过：{summary['dynamic_exact']}。",'', '默认代码未修改。即使本轮出现收益，仍需独立进程正式入口复核后才能切换。']
 (root/'README_RESULTS.md').write_text('\n'.join(lines)+'\n')
 (root/'completed.json').write_text(json.dumps(dict(output=str(r),status='completed'),indent=2));record(event='completed',output=str(r));break
