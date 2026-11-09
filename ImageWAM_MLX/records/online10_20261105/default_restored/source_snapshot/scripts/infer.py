"""Run a new decoded RoboTwin observation without PyTorch."""
import argparse,json,sys,time
import os
# Set process-level tuning before importing MLX or creating a GPU command buffer.
_startup=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
_startup.add_argument('--profile',default='accuracy')
_startup.add_argument('--no-runtime-tuning',action='store_true')
_startup_args,_=_startup.parse_known_args()
if _startup_args.profile=='speed' and not _startup_args.no_runtime_tuning:
    os.environ.setdefault('MLX_MAX_OPS_PER_BUFFER','256')
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
p.add_argument('--profile',choices=['accuracy','speed'],default='accuracy')
p.add_argument('--vae-asymmetric',action='store_true',help='Experimental asymmetric VAE padding; no confirmed end-to-end gain')
p.add_argument('--no-vae-asymmetric',action='store_true',help='Use previous explicit VAE downsample padding')
p.add_argument('--text-cache-size',type=int,default=0,help='Opt-in per-model text feature LRU entries; 0 disables; clear after changing weights')
p.add_argument('--no-tail-visual',action='store_true',help='Disable validated visual INT8 32-row tail specialization')
p.add_argument('--no-fused-gate',action='store_true',help='Use previous separate BF16 gated activation before quantization')
p.add_argument('--no-fused-concat',action='store_true',help='Use previous separate concatenation before INT8 quantization')
p.add_argument('--no-prepacked-visual',action='store_true',help='Keep W4 packed visual weights and fold in each matrix multiply')
p.add_argument('--no-staged-visual',action='store_true',help='Use previous BF16 visual speed path')
p.add_argument('--no-runtime-tuning',action='store_true',help='Do not set the speed command-buffer preset; preserves explicit environment')
p.add_argument('--no-action-epilogue',action='store_true',help='Use the previous speed action path for comparison')
a=p.parse_args()
if a.warmup<0 or a.steps<1 or a.text_cache_size<0:p.error('warmup/cache-size must be nonnegative and steps positive')
a.output.mkdir(parents=True,exist_ok=False)
raw=np.load(a.inputs);images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')}
prompt=a.prompt_file.read_text() if a.prompt_file else "A video recorded from a robot's point of view executing the following instruction: "+a.instruction
noise=np.load(a.noise) if a.noise else np.array(mx.random.normal((1,a.horizon,14),key=mx.random.key(a.seed)))
np.save(a.output/'initial_noise.npy',noise)
start=time.perf_counter();model=ImageWAM(a.weights,a.source,profile=a.profile,staged_visual=False if a.no_staged_visual else None,prepacked_visual=False if a.no_prepacked_visual else None,fused_concat=False if a.no_fused_concat else None,fused_gate=False if a.no_fused_gate else None,tail_visual=False if a.no_tail_visual else None,text_cache_size=a.text_cache_size,vae_asymmetric=a.vae_asymmetric and not a.no_vae_asymmetric,action_epilogue=False if a.no_action_epilogue else None);mx.synchronize();load=time.perf_counter()-start
sample=dict(images=images,state=raw['state'],prompt=prompt,noise=noise,steps=a.steps)
warmup=model.warmup([sample],repeats=a.warmup) if a.warmup else []
start=time.perf_counter();out,normalized=model.predict(**sample);mx.synchronize();elapsed=time.perf_counter()-start
np.save(a.output/'action.npy',out);np.save(a.output/'normalized.npy',normalized);np.savetxt(a.output/'action.csv',out,delimiter=',')
warmup_ms=sum(sum(row['durations_ms']) for row in warmup)
(a.output/'run.json').write_text(json.dumps({'vae_asymmetric':model.vae_asymmetric,'text_cache':model.text_cache_stats(),'staged_visual':model.staged_visual,'staged_visual_reason':model.staged_visual_reason,'max_ops_per_buffer':os.environ.get('MLX_MAX_OPS_PER_BUFFER'),'action_epilogue':model.action_epilogue,'action_epilogue_reason':model.action_epilogue_reason,'inference_profile':model.inference_profile,'quantization':model.quantization_info,'speed_options':model.speed_options,'kv_workspace':model.kv_workspace,'optimization':model.optimization,'compact':model.compact,'fused_vae':model.fused_vae,'overlap':model.overlap,'load_s':load,'warmup':warmup,'warmup_ms':warmup_ms,'first_inference_ms':warmup[0]['durations_ms'][0] if warmup else elapsed*1000,'request_ms':elapsed*1000,'load_warmup_request_ms':load*1000+warmup_ms+elapsed*1000,'measurement':'first_inference_ms includes first forward even when it occurs in warmup; request_ms excludes separately reported warmup','prompt':prompt,'tokens':model.tokenize(prompt).shape[1],'steps':a.steps,'horizon':noise.shape[1],'rng':'explicit noise file' if a.noise else 'MLX seed; not identical to PyTorch RNG','inputs':str(a.inputs.resolve())},indent=2))
print('action',out.shape,'load_s',round(load,3),'warmup_ms',round(warmup_ms,3),'request_ms',round(elapsed*1000,3))
