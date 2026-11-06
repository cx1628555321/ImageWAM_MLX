import os,sys,json,subprocess
from pathlib import Path
os.environ['MLX_MAX_OPS_PER_BUFFER']='256';os.environ['MTL_CAPTURE_ENABLED']='1'
sys.path.insert(0,str(Path.cwd()))
r=Path('records/background_20261105/profile');r.mkdir(parents=True,exist_ok=True);out={}
for name,cmd in [('developer_dir',['xcode-select','-p']),('xctrace',['xcrun','xctrace','list','templates']),('metal_compiler',['xcrun','-f','metal'])]:
 p=subprocess.run(cmd,capture_output=True,text=True);out[name]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
(r/'toolchain.json').write_text(json.dumps(out,indent=2))
# Command-buffer timeline is explicitly not an individual-kernel timeline.
source=Path('experiments/audit4_20261105/probe_run.py').read_text().replace("r=Path('records/audit4_20261105')","r=Path('records/background_20261105/profile')").replace("profile='speed',text_cache_size=0","profile='speed',text_cache_size=1,prefix_precision='all_int8'")
exec(compile(source,'background_probe','exec'))
try:
 mx.synchronize();mx.metal.start_capture(str((r/'request.gputrace').resolve()))
 try:m.predict(**args);mx.synchronize()
 finally:mx.metal.stop_capture()
 out['capture']='saved; capture existence alone does not establish parsed kernel times'
except Exception as e:out['capture']=repr(e)
out['kernel_timeline_status']='pending_analysis' if out['xctrace']['returncode']==0 else 'blocked_missing_full_xcode'
(r/'toolchain.json').write_text(json.dumps(out,indent=2))
