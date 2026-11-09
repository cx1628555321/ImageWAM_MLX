"""Build the detailed Chinese report from retained measurements."""
import collections,json,re
from pathlib import Path
import numpy as np
root=Path.cwd();folder=root/'records/detailed_20261103';d=json.loads((folder/'analysis_with_ci.json').read_text());acc=json.loads((root/'records/accuracy_detailed_20261103/accuracy.json').read_text());av=json.loads((root/'records/async_validation_20261103/async_vs_original_mlx.json').read_text());loads=[json.loads(f.read_text()) for f in sorted(folder.glob('load_probe_*.json'))]
assert d['status']=='completed' and len(loads)==3 and av['all_bitwise_equal'] and len(av['cases'])==51
s=d['steady_all'];mods=d['modules'];total=sum(v['mean'] for v in mods.values());ac=acc['dynamic']['aggregate'];groups=collections.defaultdict(float)
for k,v in d['operators'].items():groups['.'.join(k.split('.')[:2])]+=v['mean_ms_per_inference']
lines=['# M5 Pro ImageWAM MLX 详细性能、精度与优化分析（2026-11-03）','',
'全部记录位于 `/Users/qklj/work/cx/ImageWAM_MLX`。本次没有修改模型权重、输入处理或默认推理路径；四组优化使用独立脚本内的可逆替换，结果独立保存。','',
'## 环境与测量口径','',
'Apple M5 Pro，15 核 CPU、16 核 GPU、24 GiB 统一内存，macOS 27.2，Python 3.12.13，MLX 0.32.4。没有独立显存；设备建议工作集约 17.76 GiB。详见 [环境信息](records/environment.json)。',
'BF16、三图拼接 H×W=288×256、原始状态 14 维、输出动作 16×14、10 步去噪；同一份 PyTorch CPU seed=42 噪声。文本由实际内容决定长度，每次在线运行文本/图像编码，不缓存跨输入特征。',
'端到端从三张已解码 RGB、状态、完整 prompt 到 CPU 反归一化动作；不含视频解码、加载和文件写入。计时结束前同步 MLX。预热 8 次，稳态分 3 段各 50 次；另做 10 次模块同步分析、3 次细粒度算子分析。各阶段顺序执行，没有并发运行其他模型测试。','',
'## 模型加载与首次推理','',
f'详细测试进程加载总计 **{d["load_total_ms"]/1000:.3f} s**，首次推理 **{d["first_ms"]:.3f} ms**。加载不包括 Python 导入时间；文件读取计时包括 `mx.eval` 与同步，VAE 权重布局变换可能延迟到首次推理。',
'','|权重文件|体积 GiB|读取/求值 ms|','|---|---:|---:|']
for r in d['load']:lines.append(f'|{r["file"]}|{r["bytes"]/2**30:.4f}|{r["ms"]:.3f}|')
lines += [f'|构造、校验、tokenizer/配置等剩余开销|—|{d["load_non_file_ms"]:.3f}|','',
'三个独立进程的加载复测：'+ '、'.join(f'{x["total_ms"]/1000:.3f} s' for x in loads)+f'；均值 {np.mean([x["total_ms"] for x in loads])/1000:.3f} s。',
'OS 文件缓存和驱动缓存未清空，因此这些数字不是严格冷盘加载时间。此前首次推理出现过 1.148–5.827 s，本次 8.169 s，首次路径明显受运行状态影响；不挑选最低值作为启动保证。','',
'## 稳态端到端','',
'|指标|ms|','|---|---:|']
for k,label in [('mean','平均'),('median','P50'),('p90','P90'),('p95','P95'),('p99','P99'),('min','最小'),('max','最大'),('std','标准差')]:lines.append(f'|{label}|{s[k]:.3f}|')
lines += ['',f'共 {s["count"]} 次，变异系数 {s["cv"]*100:.2f}%；150 份动作输出相对首次输出最大差为 {d["repeatability_max_abs"]}。每轮 NPY 已保存，未删慢样本。','',
'|连续测试段|次数|平均 ms|P95 ms|','|---|---:|---:|---:|']
for i,b in enumerate(d['steady_blocks']):lines.append(f'|{i+1}|{b["count"]}|{b["mean"]:.3f}|{b["p95"]:.3f}|')
lines += ['','测试期间采样发现系统 swap 占用约 5.6 GB。两次全系统 vm_stat 样本之间 Swapins 增加 1824 页（页面 16 KiB，约 28.5 MiB），Swapouts 未增加。采样跨越部分测试阶段，且是全系统计数，无法将这些换入归因到本模型或全部稳态抖动。温控查询没有记录到警告；没有逐帧 GPU 频率遥测，不能排除频率/后台任务影响。保留了 `records/detailed_vmstat_sample*.txt` 和 swap 样本。',
f'本测试进程结束时 MLX active={d["memory"]["active"]/2**30:.3f} GiB、peak={d["memory"]["peak"]/2**30:.3f} GiB、cache={d["memory"]["cache"]/2**30:.3f} GiB。peak 覆盖全部实验，不能归属到某单个候选方案，也不是系统总内存。','',
'## 模块与文本长度','',
'|模块，10 次独立同步测量|平均 ms|P95 ms|同步模块总和占比|','|---|---:|---:|---:|']
for k,v in mods.items():lines.append(f'|{k}|{v["mean"]:.3f}|{v["p95"]:.3f}|{v["mean"]/total*100:.1f}%|')
lines += ['', '模块占比的分母是独立同步模块时间总和；插入同步改变调度，不能把该总和当成无钩子的稳态端到端值。视觉前缀和动作去噪合计约占模块时间的 70%，为首要瓶颈。','',
'|真实 prompt token 长度|独立预热后次数|平均 ms|P95 ms|','|---|---:|---:|---:|']
for k,v in d['length_sweep'].items():lines.append(f'|{k}|{v["count"]}|{v["mean"]:.3f}|{v["p95"]:.3f}|')
lines += ['', '各长度来自不同真实任务文本，输入图像和状态相同，每个长度先预热 6 次；按 31、48、64、79 顺序测量。结果反映本次运行的长度趋势，但混有执行顺序/系统状态影响，不是严格隔离的 token 数因果实验。代码不固定文本长度。','',
'## 更细粒度的热点','',
'每个算子诊断在计时前求值输入、同步，再求值输出并同步，记录含 CPU 调度和同步开销的 wall time。该模式严重扰动流水线：例如文本 Linear 的诊断和会大于整个文本模块正常时间。以下仅用于定位候选热点，不能用作生产耗时占比或 GPU kernel 时间。','',
'|算子组|诊断平均 ms/推理|','|---|---:|']
for k,v in groups.items():lines.append(f'|{k}|{v:.3f}|')
lines += ['', '视觉 single block 的 linear1（QKV 与 MLP 投影）是显著热点；动作专家每次执行 25 层×10 步，诊断记录到 250 次注意力调用。VAE 第一分辨率层的 GroupNorm 也较重。正常路径的最主要时间仍由视觉前缀与动作去噪决定；单独优化 VAE 只能影响部分端到端延迟。','',
'## 已实测的优化方案','',
'每个候选先单独预热，再与原 MLX 交替顺序各测 30 次；不把不同候选配对中的 baseline 混用。','',
'|方案|配对 baseline ms|候选 ms|平均延迟降低|相对 MLX 原版动作最大差|','|---|---:|---:|---:|---:|']
for k,v in d['experiments'].items():lines.append(f'|{k}|{v["timings"]["baseline"]["mean"]:.3f}|{v["timings"][k]["mean"]:.3f}|{v["reduction_percent"]:.2f}%|{v["error_vs_mlx"]["max_abs"]:.8f}|')
lines += ['',
'- `gn_fast`：保持 PyTorch 分组布局，使用 FP32 `mx.fast.layer_norm` 再做 affine。归约顺序变化传播到 BF16 输出；收益小，不建议默认采用。',
'- `gn_compile`：编译原 GroupNorm 运算，融合产生的数值差异仍能传播到动作。收益小，不建议默认采用。',
'- `async`：仅将每一步去噪的 `mx.eval(x)` 改成 `mx.async_eval(x)`，最终 CPU 输出和计时边界仍等待完成。减少 CPU 等 GPU 后才构造下一步的串行间隙。**优先推荐此方向**：本次约 4.05% 收益，性能样本及另外 51 组代表输入全部动作逐值一致；长短切换和输入扰动亦完成。',
'- `combined`：异步加 GroupNorm 编译，约 4.95% 收益，但继承了 GroupNorm 数值变化，相比异步单独收益不足以直接推荐。','',
'描述性成对差值 bootstrap 95% 区间（baseline 减候选，ms）：','']
for k,v in d['experiments'].items():lines.append(f'- {k}: [{v["paired_delta_ms_bootstrap95"][0]:.3f}, {v["paired_delta_ms_bootstrap95"][1]:.3f}]')
lines += ['', '前两组区间跨 0，无法从本实验确认稳定收益。bootstrap 使用 10000 次重采样，没有校正时间序列自相关，不能当成跨设备/跨工作负载的收益保证。所有候选都按实际文本运行，测试了 31→79→48→31 切换；编译首次成本单独记在 JSON，未混入稳态。','',
'## 精度','',
'51 组代表输入：48 种 RoboTwin 已观察文本长度各一例，加 3 组额外真实观察，共 11424 个动作标量。比较对象为既有 MPS BF16 原版和动态版；它们均不是 FP32/CUDA 真值，尚未测试闭环任务成功率。','',
'|相对 MPS dynamic|值|','|---|---:|']
for k in ('mae','rmse','max_abs','p95_abs','p99_abs','relative_l2'):lines.append(f'|{k}|{ac[k]:.9f}|')
w=acc['dynamic']['worst'];lines += ['',f'最大差出现在 `{w["case"]}` 的 horizon step={w["horizon_step"]}、action dimension={w["action_dimension"]}（均从 0 起）；MLX={w["mlx_value"]:.9f}，MPS={w["mps_value"]:.9f}。',
'逐动作维度、逐 horizon 时间步、逐案例误差均在 [accuracy.json](records/accuracy_detailed_20261103/accuracy.json)。原 MLX 的 150 次重复推理逐值一致；异步优化对 51 组输出也逐值一致，因此在已测输入上没有新增相对 MLX 的动作误差。原有 MLX↔MPS 数值差异仍然存在。',
'此前中间结果对比已验证图像/状态输入逐值一致，VAE latent 最大差 0.2890625、末层视觉 K 最大差 0.50390625。应优先对 VAE 卷积/归一化和 BF16 舍入路径进一步定位，不能仅依据最终动作小误差推断所有中间计算逐值等价。','',
'## 后续优先级','',
'1. 优先将已验证的异步去噪调度做成显式可选模式，继续覆盖不同 seed/horizon，并单独记录其峰值内存；默认行为暂未修改。',
'2. 对动作专家单步和视觉 block 进行局部编译，显式传入 timestep、noise、KV 和动态长度，不捕获某个固定输入。当前只测试了 GroupNorm 编译，不能把它的结果推断为整块编译收益。',
'3. 进一步优化视觉/动作投影的数据搬运与小算子调度；量化应作为独立分支验证，不与 BF16 等价优化混为一谈。先试 8-bit，再评估 4-bit 的动作误差与任务成功率；目前没有实测量化收益。',
'4. VAE 优先研究保持原归约/舍入语义的融合核、卷积与 bias/激活融合。现有两种 GroupNorm 替换没有达到值得默认采用的收益/精度折中。',
'5. 内存方面可实验 MLX cache 上限或模型驻留策略，并在可控空闲环境重测。已有系统换页证据，但尚未完成 cache-limit 对照，不能承诺减少 cache 一定更快。',
'6. 加载优化优先保持 safetensors、复用已加载模型；Qwen 与视觉权重占读取主体。无需在每个动作请求重建模型。首次推理应单独安排预热，并与服务就绪时间分别记录。','',
'编译的融合和形状变化行为参考 [MLX 官方 compile 文档](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/compile.rst)。形状变化可触发重编译，不能为了提速而固定或截断任务文本。','',
'## 重现命令','',
'```bash','cd /Users/qklj/work/cx/ImageWAM_MLX','RUN_TAG=$(date +%Y%m%d_%H%M%S)',
'.venv/bin/python scripts/detailed_analysis.py --output records/$RUN_TAG-detailed --rounds 50',
'.venv/bin/python scripts/analyze_accuracy.py --output records/$RUN_TAG-accuracy',
'.venv/bin/python scripts/validate_async.py --output records/$RUN_TAG-async-validation',
'.venv/bin/python scripts/load_probe.py records/$RUN_TAG-load.json','```','',
'第一条命令对应本报告 353.910 ms 基线及四个配对实验；后三条分别用于已有输出的精度汇总、异步全案例验证和单进程加载诊断，不是独立稳态性能结论。目录/文件需不存在，避免覆盖历史结果。',
'','原始数据：[详细 JSON](records/detailed_20261103/analysis.json)、[含描述性区间的 JSON](records/detailed_20261103/analysis_with_ci.json)、[异步一致性](records/async_validation_20261103/async_vs_original_mlx.json)。源码快照、每轮输出、算子调用计数、各长度测试、首次候选成本均已留档。']
(root/'README_PERFORMANCE_20261103.md').write_text('\n'.join(lines)+'\n')
print('REPORT_WRITTEN',len(lines))
