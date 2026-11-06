import os,sys,time,subprocess,json
from pathlib import Path
root=Path('records/graph_dispatch_20261105')
for mode in ['native','partial_bound']:
 label='sample_'+mode;ready=root/(label+'.ready.json')
 assert not ready.exists()
 with (root/(label+'.log')).open('w') as f:
  proc=subprocess.Popen([sys.executable,'experiments/graph_dispatch_20261105/sample_run.py',mode,label],env=dict(os.environ,MLX_MAX_OPS_PER_BUFFER='256',IMAGEWAM_TEST_QOS='inherit'),stdout=f,stderr=subprocess.STDOUT)
  while not ready.exists():
   if proc.poll() is not None:raise RuntimeError('test exited before ready')
   time.sleep(0.25)
  pid=json.loads(ready.read_text())['pid'];assert pid==proc.pid
  with (root/(label+'_sampler.log')).open('w') as log:
   subprocess.run(['/usr/bin/sample',str(pid),'3','1','-file',str(root/(label+'.sample.txt'))],stdout=log,stderr=subprocess.STDOUT,check=True)
  assert proc.wait()==0
 print(label,flush=True)
