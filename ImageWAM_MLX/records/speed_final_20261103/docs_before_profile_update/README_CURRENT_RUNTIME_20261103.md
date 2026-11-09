# 当前正式默认运行时与后续优化（2026-11-03）

**最终更新：**当前默认已经包含 compact 动作权重、视觉/动作编译与 RoPE、VAE GN+SiLU 融合、CPU/GPU 重叠；普通 CLI 默认单样本预热。最新 100 次稳态平均 **281.650 ms**、P95 **285.014 ms**，完整 162 文件回归逐字节一致。默认代码、PyTorch 对照、模块指标和启动命令以 [当天最终汇总](README_DAILY_FINAL_20261103.md) 为准。

**以下为当天较早阶段的历史记录。**其中“未加入默认”的选择性 K/V 后续已由 compact 权重方案整合；历史耗时不代表最终版本。

最新适配：默认已启用 CPU/GPU 重叠；组内复用与 Qwen RoPE 提供可选开关。结果、精度与命令见 [本轮适配记录](README_REMAINING_ADAPTATION_20261103.md)。

后续更新：VAE GN+SiLU 融合已接入默认，正式稳态 287.763 ms；完整精度、各候选结果与回退方式见 [本轮适配报告](README_NEXT_OPTIMIZATIONS_RESULTS_20261103.md)。

`from imagewam_mlx import ImageWAM` 现在直接使用 [runtime.py](imagewam_mlx/runtime.py)。普通 infer.py、benchmark.py 不再要求实验包装入口。

## 已整合的有效优化

- 视觉前缀编译。
- 动作单步去噪编译，每步使用 async_eval 调度。
- 移除非 profile 主路径中前缀后的强制求值，让依赖触发计算。
- 视觉及动作分支的 strided RoPE Metal 融合。

正式运行时包含明确的单步计算代码，不依赖 scripts、inspect、exec、源码字符串替换或 monkeypatch。保留原始权重、三图预处理、动态文本长度、状态、噪声、步数和 horizon。没有量化、观测特征缓存或输入截断。预处理和文本编码逻辑继承原始实现。

未采纳无可靠收益或有额外数值差异的 QKV 合并、编码器联合编译、快速 RMSNorm、连续 VAE 权重副本、两步/五步联合编译、普通分段 Attention 和时间调制缓存。

原始实现仍在 [model.py](imagewam_mlx/model.py)，作为 `LegacyImageWAM` 导出。整合前源码快照保存在 [pre_runtime_integration_20261103](records/pre_runtime_integration_20261103)。不覆盖任何模型文件。

## 使用与回退

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX

# 默认即包含全部已采纳优化
.venv/bin/python scripts/infer.py \
  --inputs examples/robotwin_ep0_frame0/inputs.npz \
  --prompt-file examples/robotwin_ep0_frame0/prompt.txt \
  --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy \
  --output records/current_infer_repeat

# 默认路径：独立稳态本轮 307.974 ms；同期新旧比较另见报告末尾
.venv/bin/python scripts/benchmark.py \
  --reference ../ImageWAM/evaluate_results/m5pro_robotwin/20261103_dynamic_profile \
  --output records/current_benchmark_repeat --warmup 8 --iterations 50

# 原始 MLX 回退，禁止与默认路径使用同一个输出目录
IMAGEWAM_MLX_OPTIMIZATION=legacy .venv/bin/python scripts/infer.py \
  --inputs examples/robotwin_ep0_frame0/inputs.npz \
  --prompt-file examples/robotwin_ep0_frame0/prompt.txt \
  --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy \
  --output records/legacy_infer_repeat

# 全样本回归和新旧实现配对比较
.venv/bin/python scripts/validate_cases.py --output records/current_validation_repeat
.venv/bin/python scripts/validate_integrated_runtime.py --output records/current_pair_repeat
```

Python 也可以使用 `ImageWAM(weights, source, optimization='legacy')` 显式回退；显式参数优先于环境变量。默认 `optimization='default'`。普通推理/基准 JSON 记录实际模式。旧 run_* 实验包装入口已明确切回原始模型环境以复现历史算法，避免叠加新默认路径；这些环境设置只作用于其进程。

## 整合回归与独立性能

实测环境：Apple M5 Pro，15 CPU / 16 GPU 核，24 GiB 统一内存；macOS 27.2、Python 3.12.13、MLX 0.32.4。[完整环境](records/environment.json)。

51 个代表样本和额外输入扰动/长度切换，总计 162 个动作、归一化动作、token IDs 等文件，与原 MLX 保存记录逐值一致。覆盖 RoboTwin 已观测的 48 种文本长度（31–79，无 78），不是全数据集或闭环成功率。

独立默认入口：加载 2.540 s，首次推理 1500.147 ms；8 次预热、50 次稳态平均 307.974 ms、P95 345.235 ms，范围 277.556–350.403 ms。系统波动明显，不能与上一轮约 290 ms 的非同期运行直接比较。OS/驱动缓存未清空，不代表严格冷启动。

端到端边界仍是已解码三图、状态和完整文本到 CPU 动作，包含在线编码和十步去噪，不含模型加载、文件读取、结果落盘。同期配对检查使用同一组驻留权重，分别构建新正式运行时和旧 visual_rope 实验路径，避免重复加载导致额外内存压力。

## 后续实验：最后一层只计算 K/V

实现 [selective_kv_experiments.py](scripts/selective_kv_experiments.py)。视觉最后一层只为动作专家提供 K/V，因此尝试只使用 linear1 权重中 K/V 对应行（3072:9216），避免计算不会使用的 Q/MLP 投影。

40 组 AB/BA 配对：整合默认路径 313.077 ms，候选 306.813 ms，平均缩短 6.264 ms（2.001%）。但逐组节省时间的 bootstrap 95% 区间为 **[-3.707, 16.067] ms**（10000 次、seed=42，未校正时序相关性），跨零，尚不能认定稳定提速。因此未加入默认路径。

动态初筛：31→79→48→31 tokens，horizon/steps 8/3、16/7、32/10，输出一致。矩阵形状变化有改变舍入的可能，必须以完整验证为依据。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
# 后续实验：本轮配对候选 306.813 ms，不作为稳定提速承诺
.venv/bin/python scripts/selective_kv_experiments.py \
  --output records/selective_kv_repeat --iterations 40
.venv/bin/python scripts/run_selective_kv_experimental.py --entry validate_cases \
  --output records/selective_kv_validation_repeat
```

## 原始记录

- [整合默认入口基准与源码快照](records/integrated_benchmark_20261103/benchmark.json)。
- [整合完整验证](records/integrated_validation_20261103/validation.json)、[逐文件一致性](records/integrated_validation_20261103/integrated_vs_original.json)。
- [新默认与旧优化路径同期比较](records/integrated_pair_20261103/results.json)。
- [选择性 K/V 配对](records/selective_kv_20261103/results.json)、[统计区间](records/selective_kv_summary_20261103.json)。
- [选择性 K/V 完整验证](records/selective_kv_validation_20261103/validation.json)。
