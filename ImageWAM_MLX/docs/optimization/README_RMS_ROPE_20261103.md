# RMSNorm＋RoPE 联合 Metal 实验（2026-11-03）

已实现两个联合 kernel 并实际测试。两者均未纳入默认运行时：完整模型存在额外数值差异，端到端收益区间跨零。当前正式默认路径保持上一轮已验证的独立 RMSNorm＋RoPE Metal 融合。

## 实现

[rms_rope_experiments.py](../../scripts/rms_rope_experiments.py) 将动作分支 Q/K 的 FP32 平方求和、rsqrt、BF16 归一化舍入、权重乘法 BF16 舍入、FP32 旋转及 BF16 输出放入一个 Metal kernel。head_dim=128，64 线程负责一行，显式 strides 索引，不预先复制为连续张量。

[rms_rope_aligned_experiments.py](../../scripts/rms_rope_aligned_experiments.py) 尝试每 lane 顺序累计四个相邻元素，匹配已安装 MLX 头文件中的 REDUCE_N_READS=4 分组。参考文件为 mlx/include/mlx/backend/metal/kernels/defines.h 与 reduction/reduce_row.h。尚未用 Metal trace 确认原模型实际选择哪种归约 kernel，不能声称已完全复刻原归约行为。

两种候选只替换动作分支；视觉前缀、Qwen、VAE、步数、噪声与输入均保持现有默认路径。支持动态文本与动作长度。没有量化或特征缓存。源码独立保存，不覆盖默认 runtime.py。

## 数值检查

初始随机算子检查：长度 1、16、64、328，各自连续/非连续输入，共 8 组，全部与原 rms＋rotate 逐值一致。结果在 [kernel 检查](../../records/rms_rope_kernel_20261103/results.json)。

但完整推理动态初筛中，31/79-token 文本以及部分 horizon/steps 场景出现额外动作差异，最大约 0.008。两个 kernel 变体在这些初筛场景产生相同的误差数值；四元素分组调整没有解决问题。

第一版完整 51 样本验证：仅 7 个动作输出与原 MLX 逐值一致，44 个不同，最大动作差异 **0.0125736217**。相对 MPS dynamic 的最大动作差异 **0.0160012823**，平均逐样本 MAE **0.0008898569**。没有任务可接受误差阈值，不能声称精度无损。第二版仅完成动态初筛，未重复完整 51 样本验证。

数值差异可能与归约、rsqrt 或融合舍入路径有关，未定位到唯一原因。本轮证据只证明随机算子测试不足以替代完整模型验证。

## 配对性能

环境：Apple M5 Pro（15 CPU / 16 GPU 核），24 GiB 统一内存，macOS 27.2、Python 3.12.13、MLX 0.32.4。[完整环境](../../records/environment.json)。

各 40 组 AB/BA 交替，基线是已整合的正式默认路径，计时前后同步 GPU。端到端包含三图/状态预处理、完整文本在线编码、VAE、前缀、十步动作去噪及 CPU 返回，不含模型加载与文件 I/O。

|候选|同期基线 ms|候选 ms|平均降低|配对节省 ms 的 95% 区间|
|---|---:|---:|---:|---|
|双 SIMD 分组归约|307.236|304.655|0.840%|[-5.502, 10.939]|
|四元素分组归约|306.796|304.828|0.642%|[-7.534, 11.195]|

区间为逐组时间差 bootstrap（10000 次、seed=42，未校正时序相关性），均跨零，没有确认稳定收益。未另跑加载/独立稳态性能宣传，因为两者均未满足默认采纳条件。

## 复现命令

输出目录必须不存在。默认模型和已有权重不受这些实验影响。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
# 独立算子随机/strides 检查，不能视为完整模型精度
.venv/bin/python scripts/validate_rms_rope_kernel.py --output records/rms_rope_kernel_repeat

# 第一版端到端候选本轮 304.655 ms
.venv/bin/python scripts/rms_rope_experiments.py --output records/rms_rope_pair_repeat --iterations 40
# 第一版完整精度验证，保留全部动作和中间结果
.venv/bin/python scripts/run_rms_rope_experimental.py --entry validate_cases --output records/rms_rope_validation_repeat

# 四元素分组变体，端到端候选本轮 304.828 ms
.venv/bin/python scripts/rms_rope_aligned_experiments.py --output records/rms_rope_aligned_repeat --iterations 40
```

## 原始记录

- [第一版配对结果](../../records/rms_rope_20261103/results.json)、[区间统计](../../records/rms_rope_summary_20261103.json)。
- [四元素分组配对结果](../../records/rms_rope_aligned_20261103/results.json)、[区间统计](../../records/rms_rope_aligned_summary_20261103.json)。
- [完整模型验证](../../records/rms_rope_validation_20261103/validation.json)、[相对原 MLX 的逐动作误差](../../records/rms_rope_validation_20261103/rms_rope_vs_original.json)。
- [当前推荐运行方式](README_CURRENT_RUNTIME_20261103.md)。

后续如继续此方向，应捕获真实 Q/K 中间输入，逐项比较归一化结果，定位发生 BF16 舍入分歧的具体位置，而不是继续扩大融合范围。该定位本轮尚未完成。
