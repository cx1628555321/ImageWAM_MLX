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
p.add_argument('--warmup',type=int,default=1,help='Prewarm actual input before measured request; 0 disables (default: 1)')
p.add_argument('--profile',choices=['accuracy','speed'],default='speed')
p.add_argument('--epilogue',choices=['baseline','nax','split'],default='split')
a=p.parse_args()
if a.epilogue!='baseline' and a.profile!='speed':p.error('epilogue candidates require speed profile')
if a.warmup<0:p.error('warmup must be nonnegative')
a.output.mkdir(parents=True,exist_ok=False)
raw=np.load(a.inputs);images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')}
prompt=a.prompt_file.read_text() if a.prompt_file else "A video recorded from a robot's point of view executing the following instruction: "+a.instruction
noise=np.load(a.noise) if a.noise else np.array(mx.random.normal((1,a.horizon,14),key=mx.random.key(a.seed)))
np.save(a.output/'initial_noise.npy',noise)
start=time.perf_counter();model=ImageWAM(a.weights,a.source,profile=a.profile)
if a.epilogue!='baseline':
 import candidates
 if a.epilogue=='split':
  from split import fused
  candidates.fused=fused
 model=candidates.install(model)
mx.synchronize();load=time.perf_counter()-start
sample=dict(images=images,state=raw['state'],prompt=prompt,noise=noise,steps=a.steps)
warmup=model.warmup([sample],repeats=a.warmup) if a.warmup else []
start=time.perf_counter();out,normalized=model.predict(**sample);mx.synchronize();elapsed=time.perf_counter()-start
np.save(a.output/'action.npy',out);np.save(a.output/'normalized.npy',normalized);np.savetxt(a.output/'action.csv',out,delimiter=',')
warmup_ms=sum(sum(row['durations_ms']) for row in warmup)
(a.output/'run.json').write_text(json.dumps({'experimental_epilogue':a.epilogue,'inference_profile':model.inference_profile,'quantization':model.quantization_info,'speed_options':model.speed_options,'kv_workspace':model.kv_workspace,'optimization':model.optimization,'compact':model.compact,'fused_vae':model.fused_vae,'overlap':model.overlap,'load_s':load,'warmup':warmup,'warmup_ms':warmup_ms,'first_inference_ms':warmup[0]['durations_ms'][0] if warmup else elapsed*1000,'request_ms':elapsed*1000,'load_warmup_request_ms':load*1000+warmup_ms+elapsed*1000,'measurement':'first_inference_ms includes first forward even when it occurs in warmup; request_ms excludes separately reported warmup','prompt':prompt,'tokens':model.tokenize(prompt).shape[1],'steps':a.steps,'horizon':noise.shape[1],'rng':'explicit noise file' if a.noise else 'MLX seed; not identical to PyTorch RNG','inputs':str(a.inputs.resolve())},indent=2))
print('action',out.shape,'load_s',round(load,3),'warmup_ms',round(warmup_ms,3),'request_ms',round(elapsed*1000,3))
