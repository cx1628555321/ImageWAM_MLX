import os
os.environ['MTL_CAPTURE_ENABLED']='1';os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import mlx.core as mx
from imagewam_mlx.action_kernel import fused
r=Path('records/roadmap_20261105/capture');d=mx.load(str(r/'projection_inputs.npz'));kw=json.loads((r/'projection_parameters.json').read_text());parts=[d[f'part_{i}'] for i in range(len([k for k in d if k.startswith('part_')]))]
def run():return fused(d['x'],parts,d['residual'],d['gate'],d['bias'],**kw)
for _ in range(3):mx.eval(run())
mx.synchronize();mx.clear_cache();path=r/'isolated_projection.gputrace';mx.metal.start_capture(str(path));mx.eval(run());mx.synchronize();mx.metal.stop_capture()
size=sum(p.stat().st_size for p in path.rglob('*') if p.is_file());(r/'isolated_capture.json').write_text(json.dumps(dict(bytes=size,status='captured only; replay and GPU timing analysis unavailable without Xcode'),indent=2));print('isolated bytes',size,flush=True)
