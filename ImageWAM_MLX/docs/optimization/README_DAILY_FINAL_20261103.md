# 2026-11-03 ImageWAM 默认优化整合与最终性能对照

## 当前结论

所有已经采纳的计算优化均由 `from imagewam_mlx import ImageWAM` 默认启用。普通 `scripts/infer.py` 现默认预热一个真实输入，支持 `--warmup 0` 关闭。基准脚本仍记录任何预热之前的首次前向，避免把预热后耗时称作首次推理。

M5 Pro 最新默认 MLX：100 次稳态平均 **281.650 ms**，P95 **285.014 ms**；首次前向 **1200.423 ms**，加载 **2.616 s**。本轮重新运行 MLX 基准与完整精度回归；PyTorch 对照取今天已保存的同机记录，未在本轮重跑。

## 环境、输入和口径

- Apple M5 Pro，15 核 CPU、16 核 GPU，24 GiB 统一内存，无独立显存；设备推荐工作集约 17.76 GiB。
- macOS 27.2、Python 3.12.13、MLX 0.32.4；PyTorch 2.15.0 / MPS，CPU fallback 关闭。
- RoboTwin episode 0 / frame 0，三张原始 RGB 均为 480×640，预处理拼图为 288×256；14 维状态，完整指令有效长度 39 token，固定相同 FP32 初始噪声，16×14 动作，10 步去噪。
- 已逐项核验 PyTorch base、dynamic 与本次 MLX 的三图、状态、文本、噪声完全一致。base 保留原文本 padding 计算方式，dynamic 与 MLX 按实际文本长度计算。因此 base→MLX 不仅是框架切换收益。
- 端到端：已解码图像、状态和文本到 CPU 反归一化动作，包含在线编码，不含视频读取、磁盘保存、模型加载与预热。不使用跨请求文本/图像特征缓存。
- 当前 MLX 8 次预热、100 次稳态、另 20 次同步模块测量；历史 PyTorch 各 6 次预热、20 次稳态、5 次模块测量。
- 历史对照为非同期独立进程，不是随机交替 A/B；系统文件/Metal 缓存未清空。加载及首次耗时不代表受控冷启动性能。

## 端到端性能

| 实现 | 加载 s | 首次前向 ms | 稳态平均 ms | P95 ms | 稳态次数 |
|---|---:|---:|---:|---:|---:|
| PyTorch MPS base（已裁掉无用加载，但原文本计算方式） | 11.140 | 4829.371 | 739.498 | 770.935 | 20 |
| PyTorch MPS dynamic | 9.354 | 2476.715 | 351.494 | 352.850 | 20 |
| 今天早期 MLX BF16 | 2.653 | 5826.997 | 335.098 | 337.482 | 20 |
| **最终默认 MLX，本轮重测** | **2.616** | **1200.423** | **281.650** | **285.014** | **100** |

按上述已记录均值计算：最新 MLX 对 base 为 **2.626×**，耗时降低 **61.91%**；对 PyTorch dynamic 为 **1.248×**，降低 **19.87%**；相对今天早期 MLX 降低 **15.95%**。这些是历史记录间的观察差异，不应把它们全部归因某一项改动。最新中位数 281.895 ms，范围 277.559–287.487 ms，所有样本保留。

默认 CLI 独立验证：加载 2.609 s，预热首次前向 1098.473 ms，随后请求 269.047 ms。三段合计 3976.405 ms，不含导入、读写等脚本开销；**269.047 ms 不是冷启动耗时，也不是 100 次稳态均值**。预热把首次执行成本前移，并不减少总计算量。

## 模块对照

下表为独立同步分析阶段的平均 wall 时间（ms/推理），并非纯 GPU kernel 时间。PyTorch 5 次、MLX 20 次。边界按近似功能对应，不能精确相加或据此归因端到端提升：MLX 视觉前缀包含状态投影及输入准备，PyTorch 将它们单列；MLX 后处理还包含输出转 NumPy。

| 模块 | PyTorch base | PyTorch dynamic | 最终 MLX |
|---|---:|---:|---:|
| 图像/状态预处理（边界略有不同） | 2.733 | 3.292 | 4.412 |
| 文本整体：模板/tokenize/Qwen | 205.407 | 102.467 | 48.450 |
| 其中 Qwen 前向 | 203.756 | 100.714 | 48.086 |
| 图像编码/latent | 22.217 | 22.048 | 42.521 |
| 状态投影及打包 | 2.354 | 0.675 | 含于视觉前缀 |
| 视觉输入准备 | 7.786 | 2.184 | 含于视觉前缀 |
| 视觉前缀/KV | 333.805 | 138.362 | 114.650 |
| 10 步动作去噪 | 213.086 | 173.308 | 80.696 |
| 反归一化/返回（边界略有不同） | 0.017 | 0.018 | 0.232 |
| 模块分析阶段整体 wall | 790.107 | 445.208 | 291.228 |
| **正常无模块同步端到端** | **739.498** | **351.494** | **281.650** |

PyTorch dynamic 文本统计保留了一次 356.430 ms 的异常样本，其文本整体中位数为 **39.497 ms**（Qwen 中位数 37.374 ms），因此不能凭 102.467 vs 48.450 宣称 MLX 文本编码常态快一倍。MLX Qwen 中位数 48.267 ms。

MLX 的主要同步阶段耗时是视觉前缀 114.650 ms、动作去噪 80.696 ms；图像编码 42.521 ms 仍明显高于 PyTorch 的约 22 ms，是后续值得研究的方向。profile 会强制阶段同步并削弱默认 CPU/GPU 重叠，其总时间 291.228 ms 不应替代正常端到端 281.650 ms。

## 今天已整合的优化

1. 动作专用权重与冗余裁剪：Qwen 只保留所需前 27 层特征，移除未用层、最终 norm/LM head；只载入 VAE encoder；移除视觉最终输出层。进一步 compact 权重令视觉最后一层只生成 K/V，去掉无用 Q/MLP/output；VAE quant_conv 只生成动作编码使用的均值部分。原始权重保留，compact manifest 校验来源与形状。
2. 视觉 prefill 与动作单步编译；每步 `async_eval`，去掉非 profile 路径的前缀强制求值边界。保留可变文本和动作长度，不联合固定十步大图。
3. 视觉/动作的 stride-aware RoPE Metal kernel，保留 BF16 舍入语义。
4. VAE GroupNorm 后 affine/SiLU 融合。既有配对复测约节省 5.44 ms，完整精度通过。
5. 先提交 Qwen GPU 工作，再做 CPU 图像预处理，减少串行等待。既有配对复测约节省 2.72 ms，完整精度通过。
6. 显式真实样本预热接口接入默认 CLI；Python 用户仍可在开始服务前调用 `model.warmup([sample])`。不在构造函数隐藏执行输入，也不在每次 `predict` 中额外预热。

以上全部是当前默认代码，不依赖实验脚本包装。组逆方差复用与 Qwen RoPE 候选收益不稳定，保持可选关闭。QKV 合并、跨步大图、split attention、连续转置权重等不满足速度或数值要求的候选没有并入默认；二维 GEMM 改写无显著收益。没有量化、少步去噪、固定文本长度或缓存旧输入特征。

## 精度与内存

本轮再次完成 51 个真实离线案例及输入扰动/长度切换，162 个输出 NPY 与此前 MLX 默认记录逐字节一致；100 次稳态动作相同，默认预热 CLI 动作与正式基准相同。覆盖已观测 48 种文本长度（31–79，缺 78）及 193 token 扩展输入。

与 PyTorch 跨框架 BF16 比较不逐位相同：51 案例最大动作绝对差约 **0.01580137**，相对 dynamic 的平均案例 MAE **0.00088513**；本次性能样本最大动作差 **0.00800064**、MAE **0.00104442**。这些差异未因本次默认整合增大，不等同于已证明闭环任务成功率不受影响。

MLX allocator active 约 **14.05 GiB**，peak active **15.56 GiB**，cache 约 **2.53 GiB**。peak 覆盖整个运行含中间结果导出；这些不是进程 RSS 或独立显存，不能把不同时间的峰值与 cache 简单相加。

## 默认启动和复测

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
# 默认预热一次；本轮独立 CLI 请求 269.047 ms，预热另计 1098.473 ms。
.venv/bin/python scripts/infer.py \
  --inputs examples/robotwin_ep0_frame0/inputs.npz \
  --prompt-file examples/robotwin_ep0_frame0/prompt.txt \
  --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy \
  --output records/new_default_infer

# 首次前向 + 8 次预热 + 100 次稳态 + 20 次模块分析。
# 本轮正常端到端 281.650 ms；profile 整体 291.228 ms。
.venv/bin/python scripts/benchmark.py \
  --reference ../ImageWAM/evaluate_results/m5pro_robotwin/20261103_dynamic_profile \
  --output records/new_default_benchmark --warmup 8 --iterations 100 --profile-iterations 20

.venv/bin/python scripts/validate_cases.py --output records/new_default_validation
```

输出目录必须不存在。CLI 加 `--warmup 0` 可测无预热单次请求；基准脚本一直保留真实首次前向，不受 CLI 默认预热影响。

本轮记录：

- `records/final_default_20261103/benchmark.json`：逐次时间、源码快照/哈希、配置、内存和模块测量。
- 同目录 `inputs.npz`、`initial_noise_fp32.npy`、`actions_*.npy`、`normalized_*.npy`、`reference_action.npy/.csv`、`trace.npz`：输入、最终及中间输出。
- `records/final_validation_20261103`：全案例输出与跨框架差异；`records/final_cli_20261103`：默认预热 CLI。
- `records/final_day_20261103/summary.json`：输入一致性、162 文件比较、各框架原始指标来源与 SHA256。
- PyTorch base：`../ImageWAM/evaluate_results/m5pro_robotwin/20261103_102606_verified`；dynamic：同级 `20261103_dynamic_profile`；早期 MLX：`records/bf16_final`。

所有记录都位于项目目录，没有覆盖原始模型或历史性能结果。
