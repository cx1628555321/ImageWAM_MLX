# MLX 编码器与 QKV 优化实测（2026-11-03）

结论：完成六项优化适配及实测。继续推荐已有 `both_compile`；新增 QKV 合并并编译仅作为显式实验入口。配对测量有 3.55% 平均收益，但新增数值差异和约 810 MiB 投影副本，不作为默认路径。

## 环境与方法

远端目录 `/Users/qklj/work/cx/ImageWAM_MLX`。Apple M5 Pro，15 CPU / 16 GPU 核，24 GiB 统一内存（无独立显存）；macOS 27.2，Python 3.12.13，MLX 0.32.4。详细硬件见 [environment.json](../../records/environment.json)。

全部以 `both_compile` 为基线，保留三张真实 RoboTwin 图像、状态、在线文本编码、BF16 权重及 10 步动作去噪。实际 token 长度参与计算，没有截断、固定文本长度或跨输入特征缓存。端到端从已解码 RGB 数组、状态和文本到 CPU 反归一化动作，不含文件读取、模型加载、输出落盘或机器人执行。每次计时前后同步 GPU。

编码器实验各 30 组，QKV 各 40 组，按 AB/BA 交替运行；候选及基线分别预热。表中置信区间对逐组耗时差做 10000 次 bootstrap（seed=42），是本轮描述性估计，未校正时序相关性或多重比较，不代表跨时段稳定收益。

## 六项实验

|候选|基线 ms|候选 ms|平均缩短|节省 ms 的 95% 区间|动态文本最大动作差异|
|---|---:|---:|---:|---|---:|
|vae_contiguous|311.169|318.821|-2.46%|[-18.862, 3.492]|0.00000000|
|qwen_compile|316.466|317.360|-0.28%|[-12.299, 10.525]|0.00000000|
|encoders_compile|319.213|311.495|2.42%|[-2.755, 18.145]|0.00800064|
|qwen_fast_rms|318.728|313.036|1.79%|[-5.351, 16.665]|0.00800064|
|fused|324.750|324.423|0.10%|[-8.260, 8.533]|0.00800064|
|fused_compile|330.676|318.926|3.55%|[2.711, 20.252]|0.00800064|

动态初筛依次使用 31、79、48、31 tokens，不能据此宣称覆盖全部长度。除 `fused_compile` 外，其余候选的节省区间均跨零；没有足够证据证明其收益。权重保持 BF16 不代表计算逐值相同：融合或矩阵形状变化可能改变舍入路径，具体成因尚未做 kernel 级定位。

适配文件：[编码器实验](../../scripts/encoder_experiments.py)、[QKV 合并实现及配对测试](../../scripts/qkv_experiment.py)、[可选运行入口](../../scripts/run_qkv_experimental.py)。QKV 保留原始权重并增加 849346560 字节投影副本，未改写模型文件。

## QKV 完整验证与独立基准

完整验证 51 个代表样本，包含 RoboTwin 已观测的 48 种文本长度（31–79，无 78）及额外真实观测；另外运行非单调长度切换、超出数据集长度的文本，以及逐路摄像头、状态、文本扰动。所有输入敏感性检查通过。不是遍历整个数据集，也不是闭环任务精度。

与原 MLX 相比：37/51 动作逐值一致，14/51 不同，最大动作绝对差异 **0.0125736217**。与 PyTorch MPS dynamic 相比：最大动作绝对差异 **0.0158013701**，平均逐样本 MAE **0.0008855303**。未定义任务可接受误差阈值，因此不能声称精度无损。原始动作、归一化动作和 token IDs 均保存在 [验证目录](../../records/qkv_validation_20261103)。

独立进程运行，8 次预热、50 次稳态、另 5 次同步模块分析：
- 加载（含 QKV 打包）：2.680 s。
- 首次推理：1771.158 ms。
- 稳态平均：324.276 ms；P95：360.425 ms。
- OS 文件缓存和驱动缓存未清理，不作为严格冷启动。独立均值不优于历史 both_compile 的 317.923 ms，历史非同期结果不可直接用于比较优化收益。

|模块（独立同步分析）|平均 ms|
|---|---:|
|preprocess|2.913|
|tokenization|0.262|
|text_encoder|41.234|
|image_encoder|62.648|
|visual_prefill|126.451|
|action_denoise|94.579|
|postprocess|0.173|

模块分析带来额外同步，其求和不等于无模块同步的稳态端到端。MLX allocator active 约 15.04 GiB、进程内 peak 约 16.41 GiB，非系统总内存；未导入 PyTorch。完整基准及源码快照见 [benchmark.json](../../records/qkv_benchmark_20261103/benchmark.json)。

## 启动命令

所有命令在项目根目录执行；输出目录必须不存在，以保护已有记录。实验入口不改变默认 `infer.py` 和原始权重。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
# 六项配对测试：各候选端到端均值见上表
.venv/bin/python scripts/encoder_experiments.py --output records/encoders_repeat --iterations 30
.venv/bin/python scripts/qkv_experiment.py --output records/qkv_repeat

# QKV 实验入口：本轮稳态 324.276 ms，首次 1771.158 ms
.venv/bin/python scripts/run_qkv_experimental.py --entry benchmark \
  --reference ../ImageWAM/evaluate_results/m5pro_robotwin/20261103_dynamic_profile \
  --output records/qkv_benchmark_repeat --warmup 8 --iterations 50

# 精度验证，逐样本耗时含形状首次编译，不作为稳态性能
.venv/bin/python scripts/run_qkv_experimental.py --entry validate_cases \
  --output records/qkv_validation_repeat

# 单次三图推理；沿用相同实验模式，耗时以独立基准为准
.venv/bin/python scripts/run_qkv_experimental.py --entry infer \
  --inputs examples/robotwin_ep0_frame0/inputs.npz \
  --prompt-file examples/robotwin_ep0_frame0/prompt.txt \
  --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy \
  --output records/qkv_infer_repeat
```

已验证默认推荐方案的命令和性能见 [编译优化报告](README_COMPILE_OPTIMIZATION_20261103.md)。

## 网上建议如何落地，以及下一步优先级

1. [MLX 官方编译说明](https://ml-explore.github.io/mlx/build/html/usage/compile.html)：落实为编码器编译及 QKV 合并后编译。针对实际形状编译，避免固定文本长度。实验表明编译范围扩大不必然提速。
2. [MLX 官方 fast API](https://ml-explore.github.io/mlx/build/html/python/fast.html)：落实为 Qwen 快速 RMSNorm 实验。当前收益区间跨零且有数值变化，保留实验实现。
3. [Apple WWDC25 MLX 优化](https://developer.apple.com/videos/play/wwdc2025/315/)：编译、快速算子和自定义 Metal 是候选方向；本轮先验证高层实现，未宣称已完成自定义 Metal kernel。

后续优先级（尚未实现/实测）：

- 优先分析视觉前缀的 attention、投影、RoPE 与归一化，约占本轮模块耗时 38.5%。已有编译基础上只优化真实热点；保持每层动作需要的 K/V，不改变文本或图像内容。
- 其次针对动作专家 10 步去噪（约 28.8%）的 kernel 提交和中间张量，尝试融合归一化、调制、残差，保留步数、噪声和可变 horizon。需要 Metal trace 支持，不能仅凭模块计时断言带宽或算力瓶颈。
- 编码器优化优先级较低。QKV 的小幅收益需跨时段复测、内存评估及任务精度验证后才能升级为推荐。
- 量化可作为单独精度分支研究；本轮未量化，未减少去噪步数。

## 原始记录

- [编码器 30 组配对数据](../../records/encoders_20261103/results.json)
- [QKV 40 组配对数据](../../records/qkv_20261103/results.json)
- [配对置信区间汇总](../../records/encoder_qkv_summary_20261103.json)
- [51 样本与输入扰动验证](../../records/qkv_validation_20261103/validation.json)
- [相对原 MLX 的逐文件差异](../../records/qkv_validation_20261103/qkv_vs_original_mlx.json)
- [独立 50 轮及各轮输出](../../records/qkv_benchmark_20261103)
