import subprocess,sys,time,json
from pathlib import Path
r=Path('records/remaining_20261105');label='diagnostic_baseline';ready=r/(label+'.ready.json')
assert not ready.exists()
with (r/(label+'.log')).open('x') as log:
 p=subprocess.Popen([sys.executable,'-u','experiments/remaining_20261105/sample_worker.py','baseline',label],stdout=log,stderr=subprocess.STDOUT)
 while not ready.exists():
  if p.poll() is not None:raise RuntimeError('diagnostic worker exited early')
  time.sleep(.25)
 assert json.loads(ready.read_text())['pid']==p.pid
 with (r/'sampler.log').open('x') as sample:
  subprocess.run(['/usr/bin/sample',str(p.pid),'3','1','-file',str(r/'diagnostic.sample.txt')],stdout=sample,stderr=subprocess.STDOUT,check=True)
 assert p.wait()==0
