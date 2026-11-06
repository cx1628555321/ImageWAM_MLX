"""Offline real RoboTwin three-camera benchmark; no simulator or feature cache."""
import argparse, csv, hashlib, json, os, platform, statistics, subprocess, sys, time, traceback
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT/'src', ROOT/'third_party/flux2/src', ROOT/'scripts/flux2'):
    sys.path.insert(0, str(p))

def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def stats(values):
    import numpy as np
    v=list(values)
    return dict(count=len(v), mean=float(np.mean(v)), median=float(np.median(v)),
                p95=float(np.percentile(v,95)), std=float(np.std(v)), min=min(v), max=max(v))

class Recorder:
    def __init__(self):
        self.records=[]; self.phase='load'; self.iteration=-1; self.stack=[]
    @contextmanager
    def span(self,name,category='module',**kwargs):
        import torch
        torch.mps.synchronize()
        parent=self.stack[-1] if self.stack else None
        self.stack.append(name); start=time.perf_counter(); status='ok'
        try: yield
        except BaseException:
            status='failed'; raise
        finally:
            torch.mps.synchronize()
            self.records.append(dict(name=name,category=category,parent=parent,phase=self.phase,
                                     iteration=self.iteration,wall_ms=(time.perf_counter()-start)*1000,status=status))
            self.stack.pop()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--episode',type=int,default=0)
    p.add_argument('--frame',type=int,default=0)
    p.add_argument('--iterations',type=int,default=20)
    p.add_argument('--profile-iterations',type=int,default=5)
    p.add_argument('--warmup',type=int,default=6)
    a=p.parse_args(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=False)
    import torch, numpy as np, av, pandas as pd
    from PIL import Image
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from imagewam.optimized_action_loader import create_imagewam_flux2_action_only
    from imagewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from profile_3090_modules import module_instrumentation
    assert torch.backends.mps.is_available()
    torch.manual_seed(42); np.random.seed(42)
    rec=Recorder(); iterations=[]; report={'status':'running','configuration':vars(a)|{'output_dir':str(out)}}
    report['environment']={'python':sys.version,'torch':torch.__version__,'platform':platform.platform(),
        'fallback_env':os.getenv('PYTORCH_ENABLE_MPS_FALLBACK','unset'),
        'git':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'packages':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True) if False else subprocess.check_output(['/Users/qklj/.local/bin/uv','pip','freeze','--python',sys.executable],text=True)}
    report['measurement']={'boundary':'Three decoded RGB arrays + raw state + task text -> CPU denormalized [16,14] action; excludes video decode, model load and file writes',
        'precision':'BF16, Qwen RoPE FP32','seed':42,'rand_device':'cpu','steps':10,'horizon':16,'context_length':512,
        'camera_layout':'compact_288x256: head 256x192, left/right 128x96 (width x height)',
        'steady':'MPS synchronize only at end-to-end boundaries; no module hooks; online text/image encoding every call',
        'profile':'Separate pass, synchronized inclusive wall time per module; parent/child timings overlap; synchronization perturbs performance',
        'load':'Once per fresh process, includes device transfers; OS file cache is uncontrolled, not a cold-disk claim'}
    def persist():
        report['iterations']=iterations; report['spans']=rec.records
        write(out/'benchmark.json',report)
    try:
        data=ROOT/'data/robotwin2.0/robotwin2.0'
        info=json.loads((data/'meta/info.json').read_text())
        ep=a.episode; chunk=ep//info['chunks_size']
        parquet=data/info['data_path'].format(episode_chunk=chunk,episode_index=ep)
        table=pd.read_parquet(parquet); row=table.iloc[a.frame]
        task_id=int(row['task_index']); instruction=None
        with (data/'meta/tasks.jsonl').open() as f:
            for line in f:
                task=json.loads(line)
                if task['task_index']==task_id:instruction=task['task'];break
        assert instruction is not None
        images={}; sources={}
        for camera in ('cam_high','cam_left_wrist','cam_right_wrist'):
            key='observation.images.'+camera
            path=data/info['video_path'].format(episode_chunk=chunk,episode_index=ep,video_key=key)
            with av.open(str(path)) as container:
                for i,frame in enumerate(container.decode(video=0)):
                    if i==a.frame:
                        images[camera]=frame.to_ndarray(format='rgb24');break
            assert camera in images
            Image.fromarray(images[camera]).save(out/(camera+'.png'))
            sources[camera]={'path':str(path),'frame':a.frame,'sha256':hashlib.sha256(images[camera].tobytes()).hexdigest(),'shape':list(images[camera].shape)}
        state=np.asarray(row['observation.state'],dtype=np.float32)
        np.savez(out/'inputs.npz',**images,state=state)
        np.save(out/'dataset_action_chunk.npy',np.stack(table.iloc[a.frame:a.frame+16]['action'].to_numpy()))
        prompt="A video recorded from a robot's point of view executing the following instruction: "+instruction
        report['input']={'episode':ep,'frame':a.frame,'timestamp':float(row['timestamp']),
            'parquet':str(parquet),'task_index':task_id,'instruction':instruction,'prompt':prompt,'cameras':sources,'state':state.tolist()}
        write(out/'input_manifest.json',report['input'])
        cfg=OmegaConf.create({'data':OmegaConf.load(ROOT/'configs/data/robotwin_omnigen2.yaml')})
        proc=instantiate(cfg.data.train.processor).eval()
        stats_path=ROOT/'models/robotwin_flux2_4b/dataset_stats.json'
        proc.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
        (out/'dataset_stats.json').write_bytes(stats_path.read_bytes())
        kwargs=dict(checkpoint_path=str(ROOT/'models/robotwin_flux2_4b/model.pt'),flux2_src_path=str(ROOT/'third_party/flux2'),
            ae_model_path=str(ROOT/'models/flux2_klein_base_4b/ae.safetensors'),qwen3_model_spec=str(ROOT/'models/Qwen3-4B'),
            proprio_dim=14,action_dit_config=dict(action_dim=14,hidden_dim=1024,num_heads=24,attn_head_dim=128,
                num_layers_double=5,num_layers_single=20,mlp_ratio=4.,max_action_horizon=64,use_gradient_checkpointing=False),
            device='mps',qwen_context_len=512,mot_checkpoint_mixed_attn=False)
        report['model_config']=kwargs
        OmegaConf.save(OmegaConf.create(kwargs),out/'resolved_model_config.yaml')
        persist()
        with rec.span('model_load_total',category='load'):
            model=create_imagewam_flux2_action_only(**kwargs,model_dtype=torch.bfloat16,recorder=rec).eval()
        report['loading_metadata']=model.loading_metadata
        print('MODEL_LOADED',flush=True);persist()
        api=SimpleNamespace(); last={}
        def preprocess():
            head=np.asarray(Image.fromarray(images['cam_high']).resize((256,192),Image.Resampling.BILINEAR))
            left=np.asarray(Image.fromarray(images['cam_left_wrist']).resize((128,96),Image.Resampling.BILINEAR))
            right=np.asarray(Image.fromarray(images['cam_right_wrist']).resize((128,96),Image.Resampling.BILINEAR))
            image=np.concatenate([head,np.concatenate([left,right],axis=1)],axis=0)
            tensor=torch.from_numpy(image).permute(2,0,1).unsqueeze(0).to(device='mps',dtype=torch.bfloat16)
            tensor=tensor*(2./255.)-1.
            batch={'state':{'default':torch.from_numpy(state.copy()).unsqueeze(0)}}
            batch=proc.action_state_transform(batch); batch=proc.normalizer.forward(batch)
            return tensor,batch['state']['default']
        def denorm(action):
            return proc.normalizer.normalizers['action']['default'].backward(action.float().cpu().unsqueeze(0))[0].numpy()
        def predict():
            image,proprio=api._obs_to_model_input()
            pred=model.infer_action_flux2(prompt=prompt,input_image=image,proprio=proprio,
                action_horizon=16,num_inference_steps=10,seed=42,rand_device='cpu')
            last['normalized']=pred['action']
            return api._denormalize_action(pred['action'])
        api._obs_to_model_input=preprocess;api._denormalize_action=denorm;api._predict_action_chunk=predict
        reference=None
        def once(phase,index):
            nonlocal reference
            rec.phase=phase; rec.iteration=index
            torch.mps.synchronize(); start=time.perf_counter()
            with torch.no_grad():action=api._predict_action_chunk()
            torch.mps.synchronize(); ms=(time.perf_counter()-start)*1000
            assert action.shape==(16,14) and np.isfinite(action).all()
            if reference is None:reference=action.copy()
            diff=float(np.max(np.abs(action-reference)))
            iterations.append({'phase':phase,'iteration':index,'wall_ms':ms,'max_abs_diff_first':diff,
                'mps_current_bytes':torch.mps.current_allocated_memory(),'mps_driver_bytes':torch.mps.driver_allocated_memory()})
            np.save(out/f'actions_{phase}_{index:03d}.npy',action)
            np.save(out/f'normalized_{phase}_{index:03d}.npy',last['normalized'].numpy())
            persist();print(phase,index,round(ms,3),'ms', 'diff',diff,flush=True)
            return ms
        report['first_inference_ms']=once('first',0)
        warm=[]
        for i in range(max(6,a.warmup)):
            warm.append(once('warmup',i))
        report['warmup_ms']=stats(warm)
        steady=[once('steady',i) for i in range(a.iterations)]
        report['steady_ms']=stats(steady)
        half=max(1,len(steady)//2)
        report['stability']={'cv':statistics.pstdev(steady)/statistics.mean(steady),
            'half_median_relative_drift':abs(statistics.median(steady[:half])-statistics.median(steady[half:]))/statistics.median(steady[:half])}
        with module_instrumentation(rec,model,api):
            prof=[once('profile',i) for i in range(a.profile_iterations)]
        report['profile_ms']=stats(prof)
        modules={}
        for row in rec.records:
            if row['category']=='module':
                name=row['name']; agg=modules.setdefault(name,{'values':[0.]*a.profile_iterations,'calls':0})
                agg['values'][row['iteration']]+=row['wall_ms'];agg['calls']+=1
        report['modules']={name:{'per_inference_ms':stats(v['values']),'calls_per_inference':v['calls']/a.profile_iterations} for name,v in modules.items()}
        image,proprio=preprocess()
        np.save(out/'model_input_bf16_as_fp32.npy',image.float().cpu().numpy())
        np.save(out/'proprio_normalized.npy',proprio.numpy())
        np.save(out/'initial_noise_fp32.npy',torch.randn((1,16,14),generator=torch.Generator(device='cpu').manual_seed(42)).numpy())
        np.save(out/'reference_action.npy',reference)
        np.savetxt(out/'reference_action.csv',reference,delimiter=',',fmt='%.10g')
        report['status']='completed'
        report['accuracy_note']='Saved MPS BF16 baseline only; no CUDA comparison or simulator success evaluation performed.'
        persist()
        for filename,rows in [('iterations.csv',iterations),('spans.csv',rec.records)]:
            with (out/filename).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        lines=['# M5 Pro RoboTwin 三图 BF16 基线','', '端到端：三张已解码 RGB 图、状态、指令到 CPU 反归一化动作；不含读视频、加载和写文件。',
            '首次推理发生在任何模型前向之前；模型仅加载一次。稳态无模块钩子，无文本/图像特征缓存。',
            '模块为独立测量阶段的 MPS 同步 wall 时间，父子项不可相加；非纯 GPU kernel 时间。','',
            f"首次推理：{report['first_inference_ms']:.3f} ms",f"稳态：{report['steady_ms']}",f"稳定性：{report['stability']}",'','|加载阶段|ms|','|---|---:|']
        lines += [f"|{r['name']}|{r['wall_ms']:.3f}|" for r in rec.records if r['category']=='load']
        lines += ['','|模块|调用数/推理|平均 ms/推理|','|---|---:|---:|']
        lines += [f"|{name}|{v['calls_per_inference']:g}|{v['per_inference_ms']['mean']:.3f}|" for name,v in report['modules'].items()]
        lines += ['',report['accuracy_note'],'','原始输入 inputs.npz + input_manifest.json；reference_action.npy/CSV 为首次输出；每轮归一化及反归一化输出均独立保存。']
        (out/'summary.md').write_text('\n'.join(lines)+'\n')
        print('COMPLETED',out,flush=True)
    except BaseException:
        report['status']='failed';report['traceback']=traceback.format_exc();persist();raise

if __name__=='__main__':main()
