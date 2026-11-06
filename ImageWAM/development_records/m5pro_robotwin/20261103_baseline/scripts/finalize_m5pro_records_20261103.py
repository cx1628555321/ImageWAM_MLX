import hashlib,json,shutil,subprocess,tarfile
from pathlib import Path
r=Path('/Users/qklj/work/cx/ImageWAM')
o=r/'evaluate_results/m5pro_robotwin/20261103_100525'
s=json.loads((o/'benchmark.json').read_text())
assert s['status']=='completed'
validation=json.loads((o/'original_policy_io_validation.json').read_text())
assert all(v['bitwise_equal'] for v in validation.values())
freeze=s['environment']['packages']
(o/'requirements-lock.txt').write_text(freeze)
lock=r/'requirements/m5pro_inference_20261103.txt'
with lock.open('x') as f:f.write(freeze)
sources=list((r/'src').rglob('*.py'))+list((r/'third_party/flux2/src').rglob('*.py'))
sources += [r/'scripts/flux2'/n for n in ['benchmark_m5pro_robotwin.py','validate_m5pro_robotwin.py','profile_3090_modules.py','convert_public_flux2_ae.py']]
sources += [r/'experiments/robotwin/imagewam_policy/deploy_policy.py',r/'configs/data/robotwin_omnigen2.yaml',r/'models/robotwin_flux2_4b/config.yaml']
hashes={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
with tarfile.open(o/'source_snapshot.tar.gz','x:gz') as tar:
 for p in sources:tar.add(p,arcname=str(p.relative_to(r)))
download=json.loads((r/'.setup/robotwin_20261102/download_completed.json').read_text())
weights=[x for x in download['verified'] if x['path'].startswith('models/')]
ae=r/'models/flux2_klein_base_4b/ae.safetensors'
weights.append({'path':str(ae.relative_to(r)),'bytes':ae.stat().st_size,'algorithm':'sha256','digest':hashlib.sha256(ae.read_bytes()).hexdigest(),'note':'Native AE converted from downloaded official Diffusers VAE'})
manifest={'source_sha256':hashes,'model_checksums':weights,'checksum_note':'Downloaded model hashes are taken from verified download records; converted AE freshly hashed.',
 'hardware':{'chip':'Apple M5 Pro','memory_GB':24},'macos':subprocess.check_output(['sw_vers'],text=True),
 'io_validation':validation,'max_repeat_abs_diff':max(x['max_abs_diff_first'] for x in s['iterations']),
 'memory_note':'Per-iteration memory samples are not peak memory measurements.'}
(o/'reproducibility_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
load=next(x['wall_ms'] for x in s['spans'] if x['name']=='model_load_total')
doc=f'''# M5 Pro ImageWAM RoboTwin 三图推理基线（2026-11-03）

结果目录：`evaluate_results/m5pro_robotwin/20261103_100525`。

已在 Apple M5 Pro / 24 GB 上实跑。PyTorch {s['environment']['torch']}、Python 3.12、Transformers 4.56.1，MPS BF16，CPU fallback 关闭。未修改原模型、配置或 CUDA 推理入口；新增独立基准和核对脚本，使用 `.venv-mps` 环境。原动作专用加载器在本机无需修改即可运行。

## 推理路线

真实 RoboTwin episode 0 / frame 0：head + left wrist + right wrist 三张 RGB 图，14 维状态，任务文本。上方 head resize 为 256×192、下方两个 wrist 各 128×96，最终 H×W=288×256。使用原始 z-score 状态归一化与动作反归一化。与仓库原 `WorldActionRobotWinPolicy` 对应方法逐元素核对：图像、状态、动作全部相等。

动作专用加载器直接恢复 RoboTwin `model.pt` 中的视觉专家、ActionDiT 与 proprio，加载 VAE encoder 和 Qwen3 前 27 层（取 hidden states 9/18/27），不读取独立 FLUX.2 基础权重。VAE 转换生成新文件 `models/flux2_klein_base_4b/ae.safetensors`，原始 Diffusers 文件保留。

每次在线编码文本和图像，无特征缓存；context=512、horizon=16、denoise steps=10、seed=42、CPU FP32 初始噪声、输出 16×14 动作。使用原 no_grad 路径，不启用 torch.compile 或 inference_mode。

## 实测性能

|项目|结果|
|---|---:|
|模型加载（一次，含转移和加载器内初始化）|{load/1000:.3f} s|
|首次推理（任何预热之前）|{s['first_inference_ms']/1000:.3f} s|
|预热次数|6|
|无钩子稳态次数|20|
|稳态平均|{s['steady_ms']['mean']:.3f} ms|
|稳态中位数|{s['steady_ms']['median']:.3f} ms|
|稳态 p95|{s['steady_ms']['p95']:.3f} ms|
|稳态标准差|{s['steady_ms']['std']:.3f} ms|
|独立模块 profiling 次数|5|
|带模块同步的端到端平均|{s['profile_ms']['mean']:.3f} ms|

端到端从已解码三张 RGB 数组、状态、文本到 CPU 反归一化动作，包含 resize/拼接、归一化、在线文本/图像编码、视觉 prefill、10 步去噪及输出回传；不含视频读取、模型加载、文件输出或模拟器。加载受系统文件缓存影响，不是受控冷磁盘读取。首次推理是新进程的首个前向，不声称系统 Metal 编译缓存为空。

模块耗时见本次 `summary.md`；使用 MPS 边界同步的 inclusive wall 计时，不是纯 GPU kernel 时间，父子模块不可相加。模块测量与无钩子稳态分开。所有测量重复相同真实观测，不代表所有任务或不同尺寸。

## 精度比较资料

- `inputs.npz`、三张 PNG、`input_manifest.json`：原始图、状态、指令、数据位置、帧索引与输入哈希。
- `model_input_bf16_as_fp32.npy`、`proprio_normalized.npy`：模型实际输入（BF16 图像转换成 FP32 保存数值）。
- `initial_noise_fp32.npy`：同一 seed 的 CPU 初始噪声。
- `reference_action.npy` / `.csv`：首次推理反归一化动作。
- `actions_<phase>_<index>.npy`、`normalized_<phase>_<index>.npy`：每轮最终输出及归一化输出。
- `dataset_action_chunk.npy`：数据集对应动作片段，不能视为模型数值精度参考。
- `benchmark.json`、`iterations.csv`、`spans.csv`：全部性能原始数据与加载记录。
- `original_policy_io_validation.json`：原入口预处理/后处理核对。
- `requirements-lock.txt`、`source_snapshot.tar.gz`、`reproducibility_manifest.json`：软件版本、源码快照与模型校验记录。

本轮所有输出有限，形状均为 16×14；32 次推理（首次1+预热6+稳态20+profile5）相对首次的最大绝对差为 0。该结果是 MPS BF16 的可复测基线，尚未与 3090 对比，也未运行 RoboTwin 闭环仿真，不能据此宣称任务成功率。

后续精度比较请复用 inputs、prompt、归一化统计、CPU 初始噪声和推理步数；报告 normalized/denormalized 两种输出的 max abs、MAE、RMSE 以及左右夹爪维度（索引6、13）。模型权重来源一致仍可能因后端与算子精度不同产生误差。

## 复跑

```bash
cd /Users/qklj/work/cx/ImageWAM
# 新机器重建隔离环境时使用；现有 .venv-mps 可直接复用
~/.local/bin/uv venv --python 3.12 .venv-mps
~/.local/bin/uv pip install --python .venv-mps/bin/python -r requirements/m5pro_inference_20261103.txt

# 每次指定不存在的新结果目录，脚本拒绝覆盖已有结果
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=0 \\
  .venv-mps/bin/python -u scripts/flux2/benchmark_m5pro_robotwin.py \\
  --output-dir evaluate_results/m5pro_robotwin/NEW_RUN \\
  --episode 0 --frame 0 --warmup 6 --iterations 20 --profile-iterations 5

.venv-mps/bin/python scripts/flux2/validate_m5pro_robotwin.py \\
  evaluate_results/m5pro_robotwin/NEW_RUN
```

本轮结束后的 MPS allocator 约 {s['iterations'][7]['mps_current_bytes']/1024**3:.2f} GiB，driver 约 {s['iterations'][7]['mps_driver_bytes']/1024**3:.2f} GiB；这是迭代结束采样，不是峰值内存。
'''
with (r/'docs/M5PRO_ROBOTWIN_INFERENCE_20261103.md').open('x') as f:f.write(doc)
(o/'development_report.md').write_text(doc)
print(doc)
