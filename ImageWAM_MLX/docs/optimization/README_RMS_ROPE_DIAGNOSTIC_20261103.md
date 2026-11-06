# 真实 Q/K 数值诊断与 rsqrt 修正（2026-11-03）

本轮捕获真实 31-token RoboTwin 输入的全部十步动作去噪 Q/K，并定位一个可复现的 BF16 舍入分歧。默认运行时暂不改变；各修正版均独立保留。

## 诊断可信度

[diagnose_rms_rope.py](../../scripts/diagnose_rms_rope.py) 为每个编译动作步骤返回原始 Q/K，共 10×25×2=500 个张量。捕获后的每步动作状态均与原默认 trace 逐值一致；带中间输出的诊断 kernel，其旋转输出也与原实验 kernel 在 500 个张量上逐值一致。因此此样本中的分歧不是捕获改变了动作轨迹所致。

逐阶段对比：450 个张量的均值有差异，最大 3.8147e-6；500 个张量的倒平方根有差异，最大 2.3842e-7；8 个张量的 BF16 归一化结果有差异，7 个张量的权重乘法/旋转结果有差异。一个“张量有差异”不代表其中全部元素不同。

首个旋转分歧出现在 step=1（第 2 步）、layer=19（零起始总层索引，对应 single_blocks.14）的 K。归一化中只有一个元素分歧，其位置为 [0,7,15,2]：

|数值|原 MLX|原联合 kernel|
|---|---:|---:|
|原始输入|1.0703125|1.0703125|
|均值|0.653282642364502|0.653282642364502|
|rsqrt|1.237226128578186|1.2372262477874756|
|BF16 归一化|1.3203125|1.328125|

这个元素的均值相同，倒平方根约 1.19e-7 的差异跨过 BF16 舍入边界，导致 0.0078125 的归一化差异。原联合 kernel 使用 metal::rsqrt；已安装 MLX 的 mlx/backend/metal/kernels/unary_ops.h 中 Rsqrt 使用 **metal::precise::rsqrt**。这是首个分歧的具体证据，不推断所有后续差异都只有这一个原因。

完整证据：[diagnostic.json](../../records/rms_rope_diagnostic_20261103/diagnostic.json)、[首个差异具体数值](../../records/rms_rope_diagnostic_20261103/first_divergence_values.json)、[可重放真实输入和各阶段输出](../../records/rms_rope_diagnostic_20261103/first_divergence.safetensors)。诊断没有用于性能计时。

## 两个修正方案

1. [原生归约方案](../../scripts/rms_rope_native_reduce_experiments.py)：保留 MLX 的 mean/rsqrt，仅将后续 BF16 归一化乘法、权重乘法与 RoPE 融合。动态文本和 horizon/steps 初筛全部逐值一致。40 组配对基线 307.958 ms、候选 310.530 ms，未获得平均收益；未扩展为完整 51 样本验证。
2. [precise rsqrt 联合方案](../../scripts/rms_rope_precise_experiments.py)：保留联合归约 kernel，仅将倒平方根改为 metal::precise::rsqrt，避免退回多个中间算子。完整验证和性能见文末。

## 环境与测量

Apple M5 Pro，15 CPU / 16 GPU 核，24 GiB 统一内存；macOS 27.2、Python 3.12.13、MLX 0.32.4。性能比较基线为当前正式默认路径，各 40 组 AB/BA 交替，计时前后同步 GPU。

端到端包含已解码三图/状态预处理、完整动态文本在线编码、VAE、视觉前缀、十步动作去噪和 CPU 动作返回，不包含模型加载、文件读取和保存。所有候选均只替换动作分支，不改模型权重、文本长度或图像内容。

## 复现

输出目录必须不存在。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
# 真实 Q/K 逐阶段诊断，不用于性能统计
.venv/bin/python scripts/diagnose_rms_rope.py --output records/rms_rope_diagnostic_repeat
# 保留 MLX 原生归约：本轮候选 310.530 ms
.venv/bin/python scripts/rms_rope_native_reduce_experiments.py --output records/rms_rope_native_repeat --iterations 40
# precise rsqrt 联合方案：同期端到端数值见文末
.venv/bin/python scripts/rms_rope_precise_experiments.py --output records/rms_rope_precise_repeat --iterations 40
.venv/bin/python scripts/run_rms_rope_precise_experimental.py --entry validate_cases --output records/rms_rope_precise_validation_repeat
```

原始结果：[原生归约](../../records/rms_rope_native_reduce_20261103/results.json)、[precise 版本](../../records/rms_rope_precise_20261103/results.json)、[precise 完整验证](../../records/rms_rope_precise_validation_20261103/validation.json)。

## precise 版本的结果与限制

仅修正 rsqrt：40 组同期基线 311.255 ms，候选 307.805 ms。完整 51 样本有 41 个动作输出与原 MLX 逐值一致（原版仅 7 个），10 个仍不同，最大动作差异 0.0125736217；horizon=32 初筛仍存在差异。因此修正明确改善一致性，但不能声称已经无损。

结果见 [逐文件精度比较](../../records/rms_rope_precise_validation_20261103/precise_vs_original.json)。随后测试了“四元素归约分组＋precise rsqrt”的组合，结果追加于下方。正式默认路径未改变。

## 四元素分组＋precise rsqrt：组合修复

[组合实现](../../scripts/rms_rope_aligned_precise_experiments.py) 同时采用四元素顺序累计和 metal::precise::rsqrt。51 个动作输出全部逐值一致；加上归一化动作、token IDs、输入扰动和长度切换，162 个保存文件全部一致。动态初筛的 horizon/steps=8/3、16/7、32/10 亦全部一致。结果见 [完整逐文件比较](../../records/rms_rope_aligned_precise_validation_20261103/aligned_precise_vs_original.json)。这些结论限于已测样本，不声称所有浮点输入均等价或闭环任务成功率无损。

40 组同期配对：默认基线 287.598 ms，候选 286.648 ms，平均降低 0.950 ms（0.330%）。逐组节省时间 bootstrap 95% 区间 [0.436,1.449] ms（10000 次、seed=42；未校正时序相关性）。这轮系统波动小于此前几轮，不能用不同轮次基线差异宣称额外提速。

提供可选入口，正式默认仍使用独立 RMSNorm＋RoPE Metal。修复主要价值是定位并消除了当前测试集的数值分歧；性能收益较小，需要更多场景验证后再决定是否提升为默认。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
# 组合修复配对性能：本轮候选 286.648 ms
.venv/bin/python scripts/rms_rope_aligned_precise_experiments.py --output records/rms_rope_aligned_precise_repeat --iterations 40
# 独立加载/首次/稳态：本轮数值见下方
.venv/bin/python scripts/run_rms_rope_aligned_precise_experimental.py --entry benchmark --reference ../ImageWAM/evaluate_results/m5pro_robotwin/20261103_dynamic_profile --output records/rms_rope_aligned_precise_benchmark_repeat --warmup 8 --iterations 50
# 完整精度验证
.venv/bin/python scripts/run_rms_rope_aligned_precise_experimental.py --entry validate_cases --output records/rms_rope_aligned_precise_validation_repeat
```

[组合修复配对原始数据](../../records/rms_rope_aligned_precise_20261103/results.json)、[统计区间](../../records/rms_rope_aligned_precise_summary_20261103.json)、[独立基准及源码快照](../../records/rms_rope_aligned_precise_benchmark_20261103/benchmark.json)。

## 组合修复独立性能

加载 2.536 s，首次推理 1359.712 ms；8 次预热、50 次稳态平均 286.029 ms，P95 288.022 ms。未清空 OS/驱动缓存，不是严格冷启动。

|模块（另 5 次同步分析）|平均 ms|
|---|---:|
|preprocess|3.377|
|tokenization|0.287|
|text_encoder|38.812|
|image_encoder|52.866|
|visual_prefill|116.973|
|action_denoise|77.527|
|postprocess|0.184|

同步模块分析有额外求值开销，不等同于无同步端到端分摊。已保存 [当前安装 MLX 的源码证据与哈希](../../records/rms_rope_diagnostic_20261103/mlx_source_reference)。
