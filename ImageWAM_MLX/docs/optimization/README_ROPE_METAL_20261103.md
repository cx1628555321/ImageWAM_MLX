# 动作分支 RoPE Metal 融合（2026-11-03）

实现可选 `rope` 模式：在上一轮 lazy_prefix 基础上，仅替换动作分支 Q/K 的交错 RoPE。保持模型权重、BF16、真实动态文本、三图预处理和原去噪步数。当前配对结果支持小幅收益，不修改默认入口。

## 实现

[rope_experiments.py](../../scripts/rope_experiments.py) 使用自定义 Metal kernel，把 FP32 转换、读取配对元素、旋转和 BF16 写回合并。设置 ensure_row_contiguous=False，显式按输入 strides 索引，输出连续；使用独立乘积变量保留分步舍入意图。测试支持本模型的 head_dim=128、动态动作长度。没有缓存图像、文本、状态或 KV，也没有替换 RMSNorm、Attention 或视觉分支。

参考 [MLX 官方自定义 Metal 文档](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)。现有编译器也可能融合部分原始算子；没有 kernel 时间线，不能断言节省时间全部来自某一种数据拷贝。

## 性能

环境：Apple M5 Pro，15 CPU / 16 GPU 核，24 GiB 统一内存，macOS 27.2，Python 3.12.13，MLX 0.32.4。[完整环境](../../records/environment.json)。

40 组 AB/BA 交替配对，基线 lazy_prefix；每次计时前后同步 GPU。端到端从已解码三图、状态和文本到 CPU 反归一化动作，包含在线编码，不含模型加载及文件 I/O。

|模式|均值 ms|P95 ms|
|---|---:|---:|
|同期 lazy_prefix|289.746|290.732|
|RoPE Metal|288.102|289.090|

平均缩短 1.644 ms，即 0.567%。逐组节省时间 bootstrap 95% 区间 [1.465, 1.831] ms，10000 次、seed=42；未校正时序相关性。不能把前几轮更高的独立耗时全部算作本优化收益。

独立进程：加载 2.536 s，首次推理 1373.882 ms；8 次预热、50 次稳态平均 290.652 ms，P95 292.202 ms。OS/驱动缓存未清空，不是严格冷启动。

|模块（另 5 次同步分析）|平均 ms|
|---|---:|
|preprocess|3.070|
|tokenization|0.267|
|text_encoder|39.083|
|image_encoder|53.407|
|visual_prefill|121.090|
|action_denoise|81.267|
|postprocess|0.183|

模块分析额外同步，不代表无同步端到端分摊。正式基准使用的 trace 路径也执行相同 Metal kernel。

## 精度与动态输入

- 51 个代表样本，48 种已观测文本长度（31–79，无 78）及额外真实观测；动作、归一化动作、token IDs，加输入扰动与长度切换输出，共 162 个文件，全部与原 MLX 逐值一致。
- 配对初筛覆盖 31→79→48→31 tokens；horizon/steps 为 8/3、16/7、32/10，均一致。
- 独立 kernel 检查：长度 1/8/16/32/64，连续与跨步切片输入，以及非连续 cos/sin，各 10 组随机测试，与原始 rotate 逐值一致。
- 相对 MPS dynamic 的完整样本最大动作差异仍为 0.0158013701；这些是离线数值检查，不是全数据集或闭环任务成功率。

## Metal capture 分析边界

本轮解析了已有 capture 的二进制 plist 元数据，见 [metadata_readable.json](../../records/metal_action_20261103/metadata_readable.json)。它提供会话信息，没有直接可读的 kernel 时间；capture 主体为二进制数据。服务器没有 Xcode/xctrace，仍未获取 kernel 时间线。未删除或重新生成约 17.1 GiB 的已有 capture。

## 复现命令

输出目录必须不存在，以保留原记录。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
# 配对端到端：同期 289.746 -> 288.102 ms
.venv/bin/python scripts/rope_experiments.py   --output records/rope_pair_repeat --iterations 40

# 独立端到端：本轮平均 290.652 ms，加载 2.536 s
.venv/bin/python scripts/run_rope_optimized.py --mode rope --entry benchmark   --reference ../ImageWAM/evaluate_results/m5pro_robotwin/20261103_dynamic_profile   --output records/rope_benchmark_repeat --warmup 8 --iterations 50

# 完整数值/动态输入检查，不用于稳态计时
.venv/bin/python scripts/run_rope_optimized.py --mode rope --entry validate_cases   --output records/rope_validation_repeat
.venv/bin/python scripts/validate_rope_kernel.py   --output records/rope_kernel_repeat
```

同一入口支持 `--entry infer`，接受原 infer.py 的输入参数。回退时使用原 `run_graph_optimized.py --mode lazy_prefix`；权重不需要重新转换。

## 原始记录

- [配对结果及各轮输出](../../records/rope_metal_20261103/results.json)、[配对置信区间](../../records/rope_summary_20261103.json)。
- [独立基准](../../records/rope_benchmark_20261103/benchmark.json)，同目录保存源码快照、每轮动作及 trace。
- [完整验证](../../records/rope_validation_20261103/validation.json)、[逐文件差异](../../records/rope_validation_20261103/rope_vs_original_mlx.json)。
- [kernel 边界与 strides 验证](../../records/rope_kernel_validation_20261103/results.json)。

本轮完成的是动作 RoPE 融合，尚未实施分段 KV Attention 或 RMSNorm+RoPE 联合归约。后续扩展需继续比较累计动作误差及端到端收益。
