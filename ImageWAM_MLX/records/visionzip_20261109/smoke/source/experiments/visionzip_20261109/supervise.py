"""Detach this supervisor to retain exit codes, including signal termination."""
import json
from pathlib import Path
import subprocess
import sys
import time
runner=sys.argv[1];args=sys.argv[2:]
assert runner in ('validate.py','benchmark.py')
out=Path(args[args.index('--output')+1]);job=out.with_suffix('.job.json')
started=time.time();child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name(runner)),*args])
state={'pid':child.pid,'runner':runner,'arguments':args,'status':'running','started_unix':started}
job.write_text(json.dumps(state,indent=2));code=child.wait()
state.update(status='complete' if code==0 else 'failed',returncode=code,elapsed_s=time.time()-started)
job.write_text(json.dumps(state,indent=2));print('EXIT',code,flush=True)
raise SystemExit(0 if code==0 else 1)
