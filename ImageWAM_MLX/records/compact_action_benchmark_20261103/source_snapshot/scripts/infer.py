"""Run a new decoded RoboTwin observation without PyTorch."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,default=Path('assets'));p.add_argument('--weights',type=Path,default=Path('weights/bf16'))
p.add_argument('--inputs',type=Path,required=True,help='npz: cam_high, cam_left_wrist, cam_right_wrist uint8 HWC RGB; state float32 [14]')
g=p.add_mutually_exclusive_group(required=True);g.add_argument('--instruction');g.add_argument('--prompt-file',type=Path,help='UTF-8 complete model prompt, without an added prefix')
p.add_argument('--noise',type=Path,help='Optional FP32 [1,T,14] npy; use for cross-framework comparison')
p.add_argument('--seed',type=int,default=42);p.add_argument('--horizon',type=int,default=16);p.add_argument('--steps',type=int,default=10);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
raw=np.load(a.inputs);images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')}
prompt=a.prompt_file.read_text() if a.prompt_file else "A video recorded from a robot's point of view executing the following instruction: "+a.instruction
noise=np.load(a.noise) if a.noise else np.array(mx.random.normal((1,a.horizon,14),key=mx.random.key(a.seed)))
np.save(a.output/'initial_noise.npy',noise)
start=time.perf_counter();model=ImageWAM(a.weights,a.source);mx.synchronize();load=time.perf_counter()-start
start=time.perf_counter();out,normalized=model.predict(images,raw['state'],prompt,noise,steps=a.steps);mx.synchronize();elapsed=time.perf_counter()-start
np.save(a.output/'action.npy',out);np.save(a.output/'normalized.npy',normalized);np.savetxt(a.output/'action.csv',out,delimiter=',')
(a.output/'run.json').write_text(json.dumps({'optimization':model.optimization,'load_s':load,'first_inference_ms':elapsed*1000,'prompt':prompt,'tokens':model.tokenize(prompt).shape[1],'steps':a.steps,'horizon':noise.shape[1],'rng':'explicit noise file' if a.noise else 'MLX seed; not identical to PyTorch RNG','inputs':str(a.inputs.resolve())},indent=2))
print('action',out.shape,'load_s',round(load,3),'first_ms',round(elapsed*1000,3))
