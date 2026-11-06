"""Offline real RoboTwin three-camera benchmark; no simulator or feature cache."""
import argparse, csv, hashlib, json, os, platform, statistics, subprocess, sys, time, traceback
from pathlib import Path
from contextlib import contextmanager, ExitStack
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT / 'third_party/flux2/src', ROOT / 'scripts/m5pro'):
    sys.path.insert(0, str(p))
from timing import Recorder, stats, write
from robotwin_io import prepare_observation, denormalize_action
from audit import run_audit, save_provenance

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--episode', type=int, default=0)
    p.add_argument('--frame', type=int, default=0)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--profile-iterations', type=int, default=5)
    p.add_argument('--warmup', type=int, default=6)
    p.add_argument('--reference-dir', type=Path)
    p.add_argument('--audit', action='store_true')
    p.add_argument('--optimization', choices=['baseline','tail','trim','dynamic'], default='baseline')
    a = p.parse_args()
    if a.optimization in {'trim','dynamic'} and a.reference_dir:
        p.error('Historical audit requires bitwise equality. Use test_dynamic_optimization.py --full for numeric comparisons.')
    if a.iterations < 2 or a.profile_iterations < 1 or a.warmup < 6 or (min(a.frame, a.episode) < 0):
        p.error('Require iterations >= 2, profile iterations >= 1, warmup >= 6, nonnegative episode/frame')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        p.error('CPU fallback must be disabled for the MPS benchmark')
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import torch, numpy as np, av, pandas as pd
    from PIL import Image
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from imagewam.optimized_action_loader import create_imagewam_flux2_action_only
    from imagewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from profiling import module_instrumentation
    assert torch.backends.mps.is_available()
    torch.manual_seed(42)
    np.random.seed(42)
    optimization_stack = ExitStack()
    rec = Recorder()
    iterations = []
    report = {'status': 'running', 'configuration': {k: str(v) if isinstance(v, Path) else v for (k, v) in vars(a).items()}}
    report['environment'] = {'python': sys.version, 'torch': torch.__version__, 'platform': platform.platform(), 'fallback_env': os.getenv('PYTORCH_ENABLE_MPS_FALLBACK', 'unset'), 'git': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'packages': '\n'.join(sorted((f"{p.metadata['Name']}=={p.version}" for p in __import__('importlib.metadata', fromlist=['distributions']).distributions())))}
    report['measurement'] = {'boundary': 'Three decoded RGB arrays + raw state + task text -> CPU denormalized [16,14] action; excludes video decode, model load and file writes', 'precision': 'BF16, Qwen RoPE FP32', 'seed': 42, 'rand_device': 'cpu', 'steps': 10, 'horizon': 16, 'context_length': 512, 'camera_layout': 'compact_288x256: head 256x192, left/right 128x96 (width x height)', 'steady': 'MPS synchronize only at end-to-end boundaries; no module hooks; online text/image encoding every call', 'profile': 'Separate pass, synchronized inclusive wall time per module; parent/child timings overlap; synchronization perturbs performance', 'load': 'Once per fresh process, includes device transfers; OS file cache is uncontrolled, not a cold-disk claim'}

    def persist():
        report['iterations'] = iterations
        report['spans'] = rec.records
        write(out / 'benchmark.json', report)
    try:
        data = ROOT / 'data/robotwin2.0/robotwin2.0'
        info = json.loads((data / 'meta/info.json').read_text())
        ep = a.episode
        chunk = ep // info['chunks_size']
        parquet = data / info['data_path'].format(episode_chunk=chunk, episode_index=ep)
        table = pd.read_parquet(parquet)
        row = table.iloc[a.frame]
        task_id = int(row['task_index'])
        instruction = None
        with (data / 'meta/tasks.jsonl').open() as f:
            for line in f:
                task = json.loads(line)
                if task['task_index'] == task_id:
                    instruction = task['task']
                    break
        assert instruction is not None
        images = {}
        sources = {}
        for camera in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
            key = 'observation.images.' + camera
            path = data / info['video_path'].format(episode_chunk=chunk, episode_index=ep, video_key=key)
            with av.open(str(path)) as container:
                for (i, frame) in enumerate(container.decode(video=0)):
                    if i == a.frame:
                        images[camera] = frame.to_ndarray(format='rgb24')
                        break
            assert camera in images
            Image.fromarray(images[camera]).save(out / (camera + '.png'))
            sources[camera] = {'path': str(path), 'frame': a.frame, 'sha256': hashlib.sha256(images[camera].tobytes()).hexdigest(), 'shape': list(images[camera].shape)}
        state = np.asarray(row['observation.state'], dtype=np.float32)
        np.savez(out / 'inputs.npz', **images, state=state)
        np.save(out / 'dataset_action_chunk.npy', np.stack(table.iloc[a.frame:a.frame + 16]['action'].to_numpy()))
        prompt = "A video recorded from a robot's point of view executing the following instruction: " + instruction
        report['input'] = {'episode': ep, 'frame': a.frame, 'timestamp': float(row['timestamp']), 'parquet': str(parquet), 'task_index': task_id, 'instruction': instruction, 'prompt': prompt, 'cameras': sources, 'state': state.tolist()}
        write(out / 'input_manifest.json', report['input'])
        cfg = OmegaConf.create({'data': OmegaConf.load(ROOT / 'configs/data/robotwin_omnigen2.yaml')})
        proc = instantiate(cfg.data.train.processor).eval()
        stats_path = ROOT / 'models/robotwin_flux2_4b/dataset_stats.json'
        proc.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
        (out / 'dataset_stats.json').write_bytes(stats_path.read_bytes())
        kwargs = dict(checkpoint_path=str(ROOT / 'models/robotwin_flux2_4b/model.pt'), flux2_src_path=str(ROOT / 'third_party/flux2'), ae_model_path=str(ROOT / 'models/flux2_klein_base_4b/ae.safetensors'), qwen3_model_spec=str(ROOT / 'models/Qwen3-4B'), proprio_dim=14, action_dit_config=dict(action_dim=14, hidden_dim=1024, num_heads=24, attn_head_dim=128, num_layers_double=5, num_layers_single=20, mlp_ratio=4.0, max_action_horizon=64, use_gradient_checkpointing=False), device='mps', qwen_context_len=512, mot_checkpoint_mixed_attn=False)
        report['model_config'] = kwargs
        OmegaConf.save(OmegaConf.create(kwargs), out / 'resolved_model_config.yaml')
        persist()
        with rec.span('model_load_total', category='load'):
            model = create_imagewam_flux2_action_only(**kwargs, model_dtype=torch.bfloat16, recorder=rec).eval()
        from optimizations import optimized
        optimization_stack.enter_context(optimized(model, a.optimization))
        report['measurement']['optimization'] = a.optimization
        report['measurement']['text_length_policy'] = 'actual batch max; no truncation' if a.optimization == 'dynamic' else 'original padded Qwen'
        report['loading_metadata'] = model.loading_metadata
        print('MODEL_LOADED', flush=True)
        persist()
        api = SimpleNamespace()
        last = {}

        def preprocess():
            return prepare_observation(images, state, proc)

        def denorm(action):
            return denormalize_action(action, proc)

        def predict():
            (image, proprio) = api._obs_to_model_input()
            pred = model.infer_action_flux2(prompt=prompt, input_image=image, proprio=proprio, action_horizon=16, num_inference_steps=10, seed=42, rand_device='cpu')
            last['normalized'] = pred['action']
            return api._denormalize_action(pred['action'])
        api._obs_to_model_input = preprocess
        api._denormalize_action = denorm
        api._predict_action_chunk = predict
        reference = None

        def once(phase, index):
            nonlocal reference
            rec.phase = phase
            rec.iteration = index
            torch.mps.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                action = api._predict_action_chunk()
            torch.mps.synchronize()
            ms = (time.perf_counter() - start) * 1000
            assert action.shape == (16, 14) and np.isfinite(action).all()
            if reference is None:
                reference = action.copy()
            diff = float(np.max(np.abs(action - reference)))
            iterations.append({'phase': phase, 'iteration': index, 'wall_ms': ms, 'max_abs_diff_first': diff, 'mps_current_bytes': torch.mps.current_allocated_memory(), 'mps_driver_bytes': torch.mps.driver_allocated_memory()})
            np.save(out / f'actions_{phase}_{index:03d}.npy', action)
            np.save(out / f'normalized_{phase}_{index:03d}.npy', last['normalized'].numpy())
            persist()
            print(phase, index, round(ms, 3), 'ms', 'diff', diff, flush=True)
            return ms
        report['first_inference_ms'] = once('first', 0)
        warm = []
        for i in range(max(6, a.warmup)):
            warm.append(once('warmup', i))
        report['warmup_ms'] = stats(warm)
        steady = [once('steady', i) for i in range(a.iterations)]
        report['steady_ms'] = stats(steady)
        half = max(1, len(steady) // 2)
        report['stability'] = {'cv': statistics.pstdev(steady) / statistics.mean(steady), 'half_median_relative_drift': abs(statistics.median(steady[:half]) - statistics.median(steady[half:])) / statistics.median(steady[:half])}
        with module_instrumentation(rec, model, api):
            prof = [once('profile', i) for i in range(a.profile_iterations)]
        report['profile_ms'] = stats(prof)
        modules = {}
        for row in rec.records:
            if row['category'] == 'module':
                name = row['name']
                agg = modules.setdefault(name, {'values': [0.0] * a.profile_iterations, 'calls': 0})
                agg['values'][row['iteration']] += row['wall_ms']
                agg['calls'] += 1
        report['modules'] = {name: {'per_inference_ms': stats(v['values']), 'calls_per_inference': v['calls'] / a.profile_iterations} for (name, v) in modules.items()}
        (image, proprio) = preprocess()
        np.save(out / 'model_input_bf16_as_fp32.npy', image.float().cpu().numpy())
        np.save(out / 'proprio_normalized.npy', proprio.numpy())
        np.save(out / 'initial_noise_fp32.npy', torch.randn((1, 16, 14), generator=torch.Generator(device='cpu').manual_seed(42)).numpy())
        np.save(out / 'reference_action.npy', reference)
        np.savetxt(out / 'reference_action.csv', reference, delimiter=',', fmt='%.10g')
        if a.audit:
            report['audit'] = run_audit(model, images, state, proc, prompt, reference, out, a.reference_dir)
        elif a.reference_dir:
            raise ValueError('--reference-dir requires --audit')
        save_provenance(ROOT, out, report)
        report['status'] = 'completed'
        report['accuracy_note'] = 'Saved MPS BF16 baseline only; no CUDA comparison or simulator success evaluation performed.'
        persist()
        for (filename, rows) in [('iterations.csv', iterations), ('spans.csv', rec.records)]:
            with (out / filename).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        lines = ['# M5 Pro RoboTwin 三图 BF16 基线', '', '端到端：三张已解码 RGB 图、状态、指令到 CPU 反归一化动作；不含读视频、加载和写文件。', '首次推理发生在任何模型前向之前；模型仅加载一次。稳态无模块钩子，无文本/图像特征缓存。', '模块为独立测量阶段的 MPS 同步 wall 时间，父子项不可相加；非纯 GPU kernel 时间。', '', f"首次推理：{report['first_inference_ms']:.3f} ms", f"稳态：{report['steady_ms']}", f"稳定性：{report['stability']}", '', '|加载阶段|ms|', '|---|---:|']
        lines += [f"|{r['name']}|{r['wall_ms']:.3f}|" for r in rec.records if r['category'] == 'load']
        lines += ['', '|模块|调用数/推理|平均 ms/推理|', '|---|---:|---:|']
        lines += [f"|{name}|{v['calls_per_inference']:g}|{v['per_inference_ms']['mean']:.3f}|" for (name, v) in report['modules'].items()]
        lines += ['', report['accuracy_note'], '', '原始输入 inputs.npz + input_manifest.json；reference_action.npy/CSV 为首次输出；每轮归一化及反归一化输出均独立保存。']
        (out / 'summary.md').write_text('\n'.join(lines) + '\n')
        print('COMPLETED', out, flush=True)
    except BaseException:
        report['status'] = 'failed'
        report['traceback'] = traceback.format_exc()
        persist()
        raise
    finally:
        optimization_stack.close()
if __name__ == '__main__':
    main()
