from pathlib import Path
import json,hashlib,tarfile,shutil,ast
import numpy as np
root=Path('records/epilogue_20261104')
runs={name:json.loads((root/name/'results.json').read_text()) for name in ('paired','split_fixed_paired','action_paired','split_confirm_paired')}
(root/'split_paired/INVALID.md').write_text('此轮无效：BN32/WM2/WN2 导致 TM=TN=1，NAX helper 未执行矩阵乘。所有速度数字不得作为有效收益。修正版使用 BN64，并加入编译断言及数值门槛。\n')
# Preserve the original failed probe values from the unchanged log.
old=[ast.literal_eval(line) for line in (root/'split_probe.log').read_text().splitlines() if line.startswith("{'m':")]
(root/'split_probe_invalid.json').write_text(json.dumps(old,indent=2))
lines=['# W1 Linear 门控残差融合测试（2026-11-04）','','## 结论','','已完成单内核 epilogue 与 Split-K 归约融合两个候选。生产 accuracy/speed 默认路径均未修改；Split-K 的第二轮端到端复测取得正向收益，但存在数值差异与较大运行波动，因此先提供独立可选入口。','','## 实现与范围','','保持动作 W1/group64 权重、BF16 激活，视觉/Qwen/三图预处理不变。不调用完整权重 dequantize；沿用块内解码与矩阵乘。融合 bias、gate 乘法及 residual 加法，保留 BF16 中间舍入。仅动作序列长度 >13 且 K>4096 的打包 Linear 使用候选，其他情况回退官方实现。文字长度没有固定或截断。','','单核版：NAX 矩阵乘输出直接计算门控残差。Split-K 版：NAX 输出 FP32 部分和，单独内核归约并融合 bias/gate/residual，保留一个 FP32 中间缓冲，并非零拷贝。修改累加顺序后不保证逐值一致。','','官方调度与 Split-K 参考：[MLX 0.32.4 quantized.cpp](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/quantized.cpp)。代码结构解释不是 GPU 时间线证明。生成器、上游哈希和 MIT 许可证随源码归档。','','## 环境与计时','','M5 Pro，15 CPU 核 / 16 GPU 核，24 GiB 统一内存，macOS 27.2，Python 3.12.13，MLX 0.32.4。真实 RoboTwin 三图、39-token 文本、状态、16×14 动作、10 步。每候选与当前 speed（KV 工作区开启）共用权重；双方首次运行及各 3 次预热后 60 对 AB/BA 交替，端到端含 CPU 动作输出，不含模型加载、文件读取及保存。','','本轮基线 268–275 ms，明显高于前轮约 241 ms，且单次波动较大；不可跨轮直接比较。主机存在 mediaanalysisd、VideoToolbox、虚拟机等进程，未终止它们；没有 GPU 时间线，不能归因到某个进程。首次运行先执行基线、后执行候选，共享部分编译缓存，不能用于比较冷启动收益。','','|候选|同轮基线 ms|候选 ms|节省 ms|节省 bootstrap 95% 区间 ms|最终动作 max abs|','|---|---:|---:|---:|---|---:|']
for name in ('paired','split_fixed_paired'):
 r=runs[name];lines.append(f"|{name}|{r['base_ms']:.3f}|{r['fused_ms']:.3f}|{r['saving_ms']:.3f}|[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]|{r['max_abs']:.8f}|")
lines+=['','以上两个端到端区间均跨零，不认为已证明提速。区间仅反映本轮配对样本，不覆盖不同系统状态。','','## 模块同步计时','','每配置 3 次，逐模块同步会改变调度，不能直接相加作为端到端。','','|轮次/模块|基线 ms|候选 ms|','|---|---:|---:|']
for name in ('paired','split_fixed_paired'):
 r=runs[name]
 for k in r['modules']['base'][0]:
  a=np.mean([v[k] for v in r['modules']['base']]);b=np.mean([v[k] for v in r['modules']['fused']]);lines.append(f'|{name}/{k}|{a:.3f}|{b:.3f}|')
r=runs['action_paired'];lines+=['','## 隔离动作模块复核','',f"120 对交替、双方各 5 次预热；同一前缀 KV，仅用于隔离计时，不能当成完整请求耗时。每次仍重新执行 10 步 denoise。基线 {r['base_ms']:.3f} ms，Split-K 候选 {r['fused_ms']:.3f} ms，节省 {r['saving_ms']:.3f} ms，95% 区间 [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] ms。归一化动作 max abs={r['max_abs']:.8f}、RMSE={r['rmse']:.8f}。",'','## 动态输入与精度','','短动作长度 1、8 回退路径输出逐值一致。下表为修正版与当前 speed 的差异；不是相对 PyTorch 的精度，也未验证闭环任务成功率。','','|文本 token|动作长度|步数|max abs|RMSE|','|---:|---:|---:|---:|---:|']
for d in runs['split_fixed_paired']['dynamic']:lines.append(f"|{d['tokens']}|{d['horizon']}|{d['steps']}|{d['max_abs']:.8f}|{d['rmse']:.8f}|")
lines+=['','## 排除的失败实现','','初版 Split-K 使用了不受 helper 支持的 TM=TN=1 fragment 组合，矩阵乘漏算，出现 max abs>2 的动作差异。该轮数据在 split_paired/INVALID.md 明确标为无效，不计收益。已改为 BN64，并新增静态断言、随机矩阵相对 RMSE 检查；修正后的分块部分和与完整矩阵乘吻合至浮点舍入差异。保留失败日志是为了可追溯。','','## 启动命令','','```bash','cd /Users/qklj/work/cx/ImageWAM_MLX','.venv/bin/python development/legacy_experiments/experiments_epilogue_20261104/build.py','.venv/bin/python development/legacy_experiments/experiments_epilogue_20261104/build_split.py','.venv/bin/python development/legacy_experiments/experiments_epilogue_20261104/split_probe.py','.venv/bin/python development/legacy_experiments/experiments_epilogue_20261104/paired.py','.venv/bin/python development/legacy_experiments/experiments_epilogue_20261104/split_paired.py','.venv/bin/python development/legacy_experiments/experiments_epilogue_20261104/action_paired.py','```','','paired.py / split_paired.py 对应上表两种端到端耗时；action_paired.py 仅为动作模块计时。脚本拒绝覆盖已存在的测试目录；重复执行应先在脚本副本中将 root 指向新的 records 子目录。所有输入输出 npz、原始采样、首次耗时和加载耗时已归档。','','## 后续方向','','已完成 Split-K 第二轮复测及真实样本覆盖，后续应在负载更稳定时复测并评估任务成功率，再决定默认配置。若继续做内核优化，应先扫描 BN/分块数、测量实际调度与中间缓冲成本，而不是假定算子数越少越快。本轮取得可复核的局部及第二轮端到端收益，尚未验证闭环任务精度，不直接替换默认路径。']
r=runs['split_confirm_paired']
lines+=['','## 第二轮端到端复测','',f"120 对交替：基线 {r['base_ms']:.3f} ms，Split-K 候选 {r['fused_ms']:.3f} ms，节省 {r['saving_ms']:.3f} ms，95% 区间 [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] ms；最终动作 max abs={r['max_abs']:.8f}。复现脚本为 development/legacy_experiments/experiments_epilogue_20261104/confirm.py。",'']
validation=json.loads((root/'validation/completed.json').read_text())
lines+=['## 真实样本覆盖','',f"{validation['cases']} 组真实样本/长度案例，最终动作均有限。最大绝对差 {validation['max_abs']:.8f}，逐值一致 {validation['exact_count']} 组。全部输入对应的动作与归一化动作已保存。此检查不能证明任务成功率不变。",'','## 实验推理入口','','```bash','cd /Users/qklj/work/cx/ImageWAM_MLX','RUN_TAG=$(date +%Y%m%d_%H%M%S)','.venv/bin/python development/legacy_experiments/experiments_epilogue_20261104/infer.py --epilogue split \\', '  --inputs examples/robotwin_ep0_frame0/inputs.npz \\', '  --prompt-file examples/robotwin_ep0_frame0/prompt.txt \\', '  --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy \\', '  --warmup 3 --output records/$RUN_TAG-epilogue-infer','```','','通过 --epilogue baseline 可在同一入口回到当前 speed，--epilogue nax 对应非分块候选。生产 scripts/infer.py 默认仍保持 accuracy，原 speed 未变。']
Path('README_EPILOGUE_20261104.md').write_text('\n'.join(lines)+'\n')
readme=Path('README.md');backup=root/'README.before_epilogue.md'
if not backup.exists():shutil.copy2(readme,backup)
link='[W1 Linear 门控残差融合测试](README_EPILOGUE_20261104.md)'
if link not in readme.read_text():readme.write_text(readme.read_text()+'\n- '+link+'：两种融合候选、端到端与动作模块配对，默认路径保持不变。\n')
sources=list(Path('development/legacy_experiments/experiments_epilogue_20261104').glob('*'))
(root/'source_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.is_file()},indent=2))
with tarfile.open(root/'review.tar.gz','w:gz') as t:
 for p in root.rglob('*'):
  if p.is_file() and p.suffix in ('.json','.npz','.log','.md'):t.add(p,arcname=str(p))
 for p in sources:
  if p.is_file():t.add(p,arcname=str(root/'source'/p.name))
print({k:v for k,v in runs['action_paired'].items() if k!='pairs'})
