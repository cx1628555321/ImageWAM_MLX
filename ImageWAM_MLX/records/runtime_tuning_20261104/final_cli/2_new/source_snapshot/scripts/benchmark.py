"""Real three-camera benchmark. PyTorch is not imported or required."""
import argparse,hashlib,json,platform,sys,time,shutil
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
p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,default=Path('assets')); p.add_argument('--weights',type=Path,default=Path('weights/bf16')); p.add_argument('--reference',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--iterations',type=int,default=20); p.add_argument('--warmup',type=int,default=6)
p.add_argument('--no-overlap',action='store_true'); p.add_argument('--qwen-rope',action='store_true'); p.add_argument('--vae-group-inv',action='store_true');p.add_argument('--profile-iterations',type=int,default=5);p.add_argument('--vae-tail-compile',action='store_true');p.add_argument('--vae-pow-mode',choices=['original','fused','lookup'],default='original')
p.add_argument('--profile',choices=['accuracy','speed'],default='accuracy')
p.add_argument('--no-staged-visual',action='store_true',help='Use previous BF16 visual speed path')
p.add_argument('--no-runtime-tuning',action='store_true',help='Do not set the speed command-buffer preset; preserves explicit environment')
p.add_argument('--no-action-epilogue',action='store_true',help='Use the previous speed action path for comparison')
a=p.parse_args();
if a.iterations<1 or a.warmup<0 or a.profile_iterations<1:p.error('iterations/profile-iterations must be positive and warmup nonnegative')
a.output.mkdir(parents=True,exist_ok=False)
def save(name,data): (a.output/name).write_text(json.dumps(data,indent=2,ensure_ascii=False))
def stats(x):return {'mean_ms':float(np.mean(x)),'median_ms':float(np.median(x)),'p95_ms':float(np.percentile(x,95)),'min_ms':float(np.min(x)),'max_ms':float(np.max(x))}
for f in ('inputs.npz','input_manifest.json','initial_noise_fp32.npy'):shutil.copyfile(a.reference/f,a.output/f)
raw=np.load(a.reference/'inputs.npz'); images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')}; state=raw['state']; prompt=json.loads((a.reference/'input_manifest.json').read_text())['prompt']; noise=np.load(a.reference/'initial_noise_fp32.npy')
report={'status':'running','reference':str(a.reference.resolve()),'environment':{'python':sys.version,'platform':platform.platform(),'mlx':__import__('importlib.metadata',fromlist=['version']).version('mlx'),'device':str(mx.default_device())},'measurement':'decoded RGB arrays, state and text to CPU denormalized actions; online encoders; no feature caching; excludes video decode/load/disk writes','iterations':[]}; save('benchmark.json',report)
project=Path(__file__).resolve().parents[1]
shutil.copytree(project/'imagewam_mlx',a.output/'source_snapshot/imagewam_mlx',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(project/'scripts',a.output/'source_snapshot/scripts',ignore=shutil.ignore_patterns('__pycache__'))
try:
    start=time.perf_counter(); model=ImageWAM(a.weights,a.source,profile=a.profile,staged_visual=False if a.no_staged_visual else None,action_epilogue=False if a.no_action_epilogue else None,overlap=not a.no_overlap,qwen_rope=a.qwen_rope,vae_group_inv=a.vae_group_inv,vae_tail_compile=a.vae_tail_compile,vae_pow_mode=a.vae_pow_mode); mx.synchronize(); report['load_s']=time.perf_counter()-start; report['optimization']=model.optimization; report['compact']=model.compact; report['fused_vae']=model.fused_vae; report['overlap']=model.overlap; report['qwen_rope']=model.qwen_rope; report['vae_group_inv']=model.vae_group_inv; report['vae_tail_compile']=model.vae_tail_compile; report['vae_pow_mode']=model.vae_pow_mode; report['weights_path']=model.weights_path; print('LOADED',report['load_s'],flush=True)
    report['staged_visual']=model.staged_visual;report['staged_visual_reason']=model.staged_visual_reason;report['max_ops_per_buffer']=os.environ.get('MLX_MAX_OPS_PER_BUFFER')
    report['action_epilogue']=model.action_epilogue;report['action_epilogue_reason']=model.action_epilogue_reason
    report['device_info']=mx.device_info();report['inference_profile']=model.inference_profile;report['quantization']=model.quantization_info;report['speed_options']=model.speed_options;report['kv_workspace']=model.kv_workspace
    for phase,count in [('first',1),('warmup',a.warmup),('steady',a.iterations),('profile',a.profile_iterations)]:
        times=[]
        for i in range(count):
            mx.synchronize(); start=time.perf_counter(); out,normalized=model.predict(images,state,prompt,noise,profile=phase=='profile'); mx.synchronize(); ms=(time.perf_counter()-start)*1000
            if out.shape!=(16,14) or not np.isfinite(out).all():raise ValueError('Invalid model output')
            times.append(ms); np.save(a.output/f'actions_{phase}_{i:03d}.npy',out); np.save(a.output/f'normalized_{phase}_{i:03d}.npy',normalized)
            row={'phase':phase,'index':i,'wall_ms':ms,'modules':dict(model.profile)}; report['iterations'].append(row); save('benchmark.json',report); print(phase,i,round(ms,3),flush=True)
        report[phase]=stats(times)
    out,norm=model.predict(images,state,prompt,noise,trace=True); np.savez(a.output/'trace.npz',**model.trace)
    np.save(a.output/'reference_action.npy',out); np.savetxt(a.output/'reference_action.csv',out,delimiter=',')
    ref=np.load(a.reference/'reference_action.npy'); diff=np.abs(out-ref)
    report['comparison']={'max_abs':float(diff.max()),'mae':float(diff.mean()),'rmse':float(np.sqrt(np.mean((out-ref)**2))),'cosine':float(np.dot(out.ravel(),ref.ravel())/(np.linalg.norm(out)*np.linalg.norm(ref))),'note':'numerical comparison against supplied reference, not task success validation'}
    report['memory']={'active_bytes':mx.get_active_memory(),'peak_bytes':mx.get_peak_memory(),'cache_bytes':mx.get_cache_memory()}; report['torch_imported']='torch' in sys.modules
    report['status']='completed'; report['source_hashes']={str(f.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).resolve().parents[1].glob('imagewam_mlx/*.py')}; save('benchmark.json',report); print(json.dumps(report['comparison']),flush=True)
except BaseException:
    import traceback
    report['status']='failed'; report['traceback']=traceback.format_exc(); save('benchmark.json',report); raise
