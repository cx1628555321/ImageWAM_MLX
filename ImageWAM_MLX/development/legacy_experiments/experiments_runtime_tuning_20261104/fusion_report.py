import json,hashlib,tarfile
from pathlib import Path
import numpy as np
p=Path.cwd();r=p/'records/fusion_confirm_20261104'
paired=json.loads((r/'paired_repeat/results.json').read_text());micro=json.loads((r/'micro/results.json').read_text());b=json.loads((r/'final_entry/benchmark.json').read_text());s=b['steady']
validation=json.loads((r/'final_validation/completed.json').read_text());assert all(validation[k] for k in ['native_exact','accuracy_exact','rollback_exact'])
assert b['quantization']['visual']['concat_quant_fused']
assert np.array_equal(np.load(r/'cli_infer/action.npy'),np.load(r/'final_entry/actions_steady_000.npy'))
dyn=json.loads((r/'dynamic/results.json').read_text());assert all(x['exact'] for x in dyn)
for d in [p/'records/concat_quant_20261104',p/'records/visual_epilogue_20261104',r/'paired_repeat']:
 v=json.loads((d/'validation.json').read_text());assert v['all_exact'] and len(v['exact_cases'])==51
for f in (r/'final_cli').glob('*/actions_steady_000.npy'):
 assert np.array_equal(np.load(f),np.load(r/'final_entry/actions_steady_000.npy'))
base=paired['baseline']['mean_ms'];new=paired['split_quant']['mean_ms'];lo,hi=paired['split_quant']['ci95']
lines=['# 视觉拼接与 INT8 激活量化融合：2026-11-04','',
f'已固化到 --profile speed：将每个视觉 single block 的 attention/MLP 拼接合并进 INT8 量化读取，不再生成完整拼接张量。固定 200 组共享权重配对确认：{base:.3f} → {new:.3f} ms，节省 {base-new:.3f} ms（{100*(1-new/base):.2f}%）；95% bootstrap 区间 [{lo:.3f}, {hi:.3f}] ms。这是一项约 1 ms 的小幅改善，不能把后台负载变化带来的更大下降算入优化。','',
'默认 accuracy 不变；保留 --no-fused-concat 和 Python API fused_concat=False 回退。仅在当前已验证的 speed/prepacked INT8 分支启用。磁盘权重和原配置不修改。','',
'## 实现与局部证据','',
'旧路径先计算 gated(mlp)，与 attention 拼成 BF16 张量，再量化为 INT8。新路径保留原 gated 算术及 BF16 舍入，在同一个量化 kernel 内按位置读取两个独立输入；行最大值仍覆盖两段全部通道，输出仍采用相同块布局。后续 INT8 矩阵乘、INT32 累加、BF16 输出完全沿用原实现。不是缓存输入，也不改变模型权重精度。','',
'改动位于 imagewam_mlx/split_quant_visual.py、runtime.py 和两个 CLI。生产代码不读取实验源码、不使用 AST/exec。源内核与许可沿用 staged_int8_upstream.json / LICENSE.mlx。文本长度不固定，不截断文本；每次按实际行数分配输出，模型通道维度保持其原结构。','',
f'真实投影尺寸的合成输入专项：M=328，attention=3072，原 MLP=18432。500 组同步墙钟计时，旧拼接+gate+量化 {micro["mean_ms"]["baseline"]:.6f} ms，新路径 {micro["mean_ms"]["split"]:.6f} ms，节省 {micro["saving_ms"]:.6f} ms；量化输出/尺度/指数逐值一致。19 层累计约 0.6 ms 量级，只作机制诊断，不直接等同端到端收益；这不是 GPU counter 测量。','',
'同时测试了把投影后的缩放/残差相加并入 INT8 GEMM 输出内核，保留 BF16 中间舍入。单独残差候选通过 51 个样本，但完整性能证据不足，未默认采用；组合候选也未固化。','',
'## 测试条件与证据边界','',
'Apple M5 Pro（15 CPU / 16 GPU），24 GiB 统一内存，macOS 27.2，Python 3.12.13，MLX 0.32.4，buffer256。三张已解码 RGB + state + 在线文本 → CPU 动作，10 步、默认 horizon16。不含加载、文件读取和结果写盘，不缓存跨请求特征。','',
'最初两项 160 组配对均受明显抖动影响：拼接融合节省 2.694 ms，区间 [-3.081,8.547]；残差融合节省 1.237 ms，区间 [-4.431,6.819]。这些数据本身不足以确认收益。','',
'随后进行了六进程比较与四进程确认。前者每进程 50 次、后者每进程 100 次，均预热 10 次；基线自身存在跨进程漂移，因此未将合并均值差作为净优化量。最终采用预先固定 200 组、不同随机顺序的共享权重配对确认，并与局部算子证据相互核对。全部原始结果保留，未删除慢样本。','',
'| 四进程确认 | 平均 ms |','|---|---:|']
for label in ['0_baseline','1_split','2_split','3_baseline']:
 d=json.loads((r/f'confirm_cli/{label}/benchmark.json').read_text());lines.append(f'| {label} | {d["steady"]["mean_ms"]:.3f} |')
lines+=['','正式代码部署后再次直接运行 scripts/benchmark.py（10 次预热、100 次稳态）：','',
'| 指标 | 最新正式 speed |','|---|---:|',
f'| 稳态平均 ms | {s["mean_ms"]:.3f} |',f'| P50 / P95 ms | {s["median_ms"]:.3f} / {s["p95_ms"]:.3f} |',f'| 最小 / 最大 ms | {s["min_ms"]:.3f} / {s["max_ms"]:.3f} |',f'| 模型加载 s | {b["load_s"]:.3f} |',f'| 首次前向 ms（含首次编译等） | {b["first"]["mean_ms"]:.3f} |','',
'该部署后绝对耗时与配对实验时段不同，不能直接相减归因。模块单独同步诊断各 5 次，改变了调度，仅供定位：','', '| 模块 | 平均 ms |','|---|---:|']
for key in ['text_encoder','image_encoder','visual_prefill','action_denoise']:
 v=[it['modules'][key] for it in b['iterations'] if it['phase']=='profile'];lines.append(f'| {key} | {np.mean(v):.3f} |')
lines+=['','## 精度回归','',
'正式入口对 51 个代表样本的 action/normalized 与此前保存的 speed 输出逐值一致。accuracy 标准样本与历史精度参考逐值一致，关闭 staged 的回退路径验证通过。实际 infer CLI 与 benchmark 输出逐值一致。',
'补充关闭/开启本次融合的动态边界比较：'+', '.join(f'{x["tokens"]} tokens / horizon{x["horizon"]}' for x in dyn)+'，均逐值一致。已测无新增数值误差，不代表 speed 与 PyTorch/accuracy 等精度；原量化误差仍存在，未进行机器人任务成功率测试。','',
'## 启动与回退','',
f'工作目录 /Users/qklj/work/cx/ImageWAM_MLX。最新正式实测平均 {s["mean_ms"]:.3f} ms：','',
'```bash\n.venv/bin/python scripts/benchmark.py --profile speed --reference examples/robotwin_ep0_frame0 --output records/fused_concat_next --warmup 10 --iterations 100 --profile-iterations 5\n```','',
f'关闭本次融合；配对对照平均 {base:.3f} ms（对应开启 {new:.3f} ms）：','',
'```bash\n.venv/bin/python scripts/benchmark.py --profile speed --no-fused-concat --reference examples/robotwin_ep0_frame0 --output records/unfused_concat_next --warmup 10 --iterations 100 --profile-iterations 5\n```','',
'```bash\n.venv/bin/python scripts/infer.py --profile speed --inputs examples/robotwin_ep0_frame0/inputs.npz --prompt-file examples/robotwin_ep0_frame0/prompt.txt --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy --output records/fused_concat_infer_next --warmup 1\n```','',
'输出目录需使用新名称。--no-prepacked-visual 或 --no-staged-visual 也会退出本次融合所依赖的分支。默认不指定 profile 仍是 accuracy。实验脚本已显式设置 fused_concat=False 构造旧基线，避免新默认污染后续复现。','',
'## 文件与后续','',
'records/concat_quant_20261104、records/visual_epilogue_20261104 保存初测；records/fusion_confirm_20261104 保存 before 旧源码、paired_repeat、micro、两轮独立进程、正式 51 例/accuracy 回归、动态输入、final_entry 和 cli_infer。含原始输入/输出、计时和源码快照。trace.npz 保留远端，审阅归档省略。实验源码为 development/legacy_experiments/experiments_runtime_tuning_20261104/fusion_*.py、concat_quant_test.py、visual_epilogue_test.py。',
'下一步应先量化单个归一化/调制到量化之间的开销，再决定是否进一步融合；LayerNorm 的归约与 BF16 舍入需要单独精度验证。未验证的融合不默认启用。']
name='README_CONCAT_QUANT_FUSION_20261104.md';(p/name).write_text('\n'.join(lines)+'\n')
intro=f'最新正式 speed：[拼接与 INT8 量化融合]({name})，同期配对 {base:.3f} → {new:.3f} ms，节省约 0.94 ms；已测输出逐值一致。以下为历史记录。\n\n'
read=p/'README.md';t=read.read_text();read.write_text(intro+t if not t.startswith(intro) else t)
manifest={}
for basepath in [p/'imagewam_mlx',p/'scripts',r/'final_entry']:
 for f in basepath.rglob('*'):
  if f.is_file() and '__pycache__' not in str(f):manifest[str(f.relative_to(p))]=hashlib.sha256(f.read_bytes()).hexdigest()
(r/'manifest.json').write_text(json.dumps(manifest,indent=2))
archive=p/'records/concat_quant_fusion_20261104_review.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 paths=[p/name,read,p/'imagewam_mlx',p/'scripts',r,p/'records/concat_quant_20261104',p/'records/visual_epilogue_20261104',p/'development/legacy_experiments/experiments_runtime_tuning_20261104/concat_quant_test.py',p/'development/legacy_experiments/experiments_runtime_tuning_20261104/visual_epilogue_test.py']+list((p/'development/legacy_experiments/experiments_runtime_tuning_20261104').glob('fusion_*.py'))
 for f in paths:tar.add(f,arcname=str(f.relative_to(p)),filter=lambda t:None if t.name.endswith('trace.npz') or '__pycache__' in t.name else t)
print('SHA256',hashlib.sha256(archive.read_bytes()).hexdigest(),'bytes',archive.stat().st_size)
print('FINAL',dict(paired_old_ms=base,paired_new_ms=new,final_mean_ms=s['mean_ms'],final_p95_ms=s['p95_ms']))
