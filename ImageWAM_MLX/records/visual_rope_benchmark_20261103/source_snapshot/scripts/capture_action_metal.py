"""Capture one warmed action step; capture duration is not a benchmark."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
from compile_experiments import Candidates
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
r={'status':'running','scope':'one action denoise step with steps=1; not the ten-step trajectory; warmed; no end-to-end timing'}
try:
 m=ImageWAM('weights/bf16','assets');c=Candidates(m);c.set('both_compile');ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};x,st=m.preprocess(images,raw['state']);hidden=m.qwen(mx.array(m.tokenize((ex/'prompt.txt').read_text())));latent=m.vae(x);cache=m.prefill(hidden,latent,st);mx.eval(cache);noise=np.load(ex/'initial_noise_fp32.npy')
 for _ in range(3):mx.eval(m.denoise(cache,noise,steps=1))
 mx.synchronize();mx.metal.start_capture(str((a.output/'action_step.gputrace').resolve()))
 try:out=m.denoise(cache,noise,steps=1);mx.eval(out);mx.synchronize()
 finally:mx.metal.stop_capture()
 np.save(a.output/'normalized_action.npy',np.array(out.astype(mx.float32)));r['status']='completed';r['trace_bytes']=sum(f.stat().st_size for f in (a.output/'action_step.gputrace').rglob('*') if f.is_file())
except BaseException:
 import traceback
 r['status']='failed';r['traceback']=traceback.format_exc()
finally:(a.output/'capture.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
