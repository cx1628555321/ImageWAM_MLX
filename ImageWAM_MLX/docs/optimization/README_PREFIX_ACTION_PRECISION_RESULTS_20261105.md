# 前缀与动作量化适配测试汇总（2026-11-05）

正式入口 ABBA 复测：原 speed 缓存命中 169.287 ms → 可选前缀 INT8 157.703 ms，节省 11.584 ms，耗时降低 6.84%，加速 1.073 倍。保留三图在线编码、H=16、动作 10 步；只缓存文本特征。原 accuracy/speed 默认策略不变。

## 环境与测量口径

- Apple M5 Pro，15 CPU / 16 GPU 核，24 GiB 统一内存（无独立显存）；macOS 27.2，Python 3.12.13，MLX 0.32.4，applegpu_g17s。
- 每个候选：10 次预热、50 次正式端到端、另 50 次带同步分模块统计。端到端为已解码 RGB 数组、state、文本到 CPU 反归一化动作；不含磁盘读取、视频解码、加载和预热。
- 首轮筛选使用独立进程；推荐方案通过正式 scripts/benchmark.py 做 default / all_int8 / all_int8 / default 复测，各 50 次。每次计时同步 GPU，图片每次在线计算。
- 51 个输入案例（含不同文本长度和观察帧），另测 H=1、H=64、H=16 长文本；保存动作、归一化结果与逐步轨迹。动态测试不等同于完整 RoboTwin 闭环成功率评估。
- 分模块统计会增加同步开销，其和不等于单独测得的端到端。

## 全部候选结果

以下为首轮筛选；收益相对同轮基线 169.090 ms。小于约 1 ms 的结果未通过独立复测，不认定为稳定收益。

| 方案 | 端到端 ms | 前缀 ms | 动作 10 步 ms | 节省 ms | 相对 speed：51 案例最大绝对差 | 平均案例 RMSE |
|---|---:|---:|---:|---:|---:|---:|
| 原 speed | 169.090 | 77.388 | 62.330 | 0 | 0 | 0 |
| 前缀 MLP INT8 | 161.325 | 69.843 | 62.366 | +7.765 | 0.022613 | 0.003109 |
| 前缀扩展 INT8 | 157.507 | 66.064 | 62.411 | +11.584 | 0.032003 | 0.003482 |
| 激活固定定标 | 168.955 | 77.447 | 62.332 | +0.136 | 0.032003 | 0.005412 |
| 动作 W2A16 | 169.236 | 77.354 | 62.996 | -0.146 | 2.320186 | 0.518774 |
| 动作 W4A16 | 170.420 | 77.327 | 63.868 | -1.329 | 2.284183 | 0.514319 |
| 动作部分 W8A8 | 177.807 | 77.245 | 71.281 | -8.716 | 1.315077 | 0.360953 |
| Split-K partial FP16 | 168.541 | 77.334 | 62.009 | +0.549 | 0.016001 | 0.002587 |
| Split-K partial BF16 | 168.047 | 77.304 | 61.574 | +1.043 | 0.032003 | 0.002646 |
| 融合 INT8 KV 注意力 | 232.386 | 77.365 | 124.220 | -63.295 | 0.022613 | 0.002719 |
| 前缀 INT8 + 动作 W4 | 158.857 | 66.012 | 63.781 | +10.234 | 2.280183 | 0.514261 |
| 前缀 INT8 + partial BF16 | 156.979 | 66.208 | 61.792 | +12.112 | 0.032003 | 0.003491 |

固定定标首次因 Metal 标量输入声明失败，修复后已完整重测；失败记录保留，表中使用成功重测。51 案例误差对照是当前 W1 speed，不是 BF16 真值。W2/W4 与 W1 差别大，不代表其精度更差。

## 正式入口确认

| 顺序 | 参数 | 稳态 mean ms | P95 ms | 模型加载 s | 首次推理 ms（含文本未命中） |
|---|---|---:|---:|---:|---:|
| 1 | default | 169.015 | 169.356 | 3.216 | 852.735 |
| 2 | all_int8 | 157.731 | 158.168 | 3.553 | 770.228 |
| 3 | all_int8 | 157.674 | 158.061 | 3.574 | 771.358 |
| 4 | default | 169.558 | 170.407 | 3.195 | 886.872 |

| 模块 | 原 speed ms | 前缀 INT8 ms |
|---|---:|---:|
| 分词 | 0.184 | 0.180 |
| 缓存文本特征 | 0.029 | 0.028 |
| 预处理 | 2.545 | 2.519 |
| 图像编码 | 27.902 | 27.756 |
| 多模态前缀、投影、KV 生成 | 78.320 | 66.640 |
| 动作去噪 10 步 | 62.640 | 62.518 |
| 后处理 | 0.166 | 0.168 |

## 精度解释

以下仅为标准样本相对 examples/robotwin_ep0_frame0/reference_action.npy 的反归一化动作误差。各动作维度单位不同，不能解读为毫米误差或任务成功率。

| 方案 | RMSE | 最大绝对差 |
|---|---:|---:|
| 原 speed W1 | 0.518440 | 1.808145 |
| visual_all | 0.517816 | 1.808145 |
| action_w2 | 0.024008 | 0.101200 |
| action_w4 | 0.007500 | 0.024610 |
| combined_w4 | 0.007745 | 0.024002 |
| combined_partial | 0.518214 | 1.808145 |

前缀 INT8 相对 speed 的 51 案例最大绝对差为 0.032003、平均案例 RMSE 为 0.003482，属于新增数值误差；不能宣称精度无损。当前 W1 speed 本身相对参考已有较大误差，不能因为新增误差小，就把它称为高精度版本。W4 的参考样本误差明显更小，但还需多任务闭环评估。

## 方案原理与取舍

- **前缀 INT8：采用为可选正式路径。** 新增 43 个可量化矩阵：5 个双流块的 MLP/QKV/输出投影，以及 img_in、txt_in、末层 KV。权重从原 BF16 生成逐输出通道 INT8；激活动态 INT8；NAX INT8 乘加、INT32 累加，输出 BF16。已有 38 个 W4 衍生 INT8 矩阵保留。不在 forward 还原完整浮点权重。Softmax、归一化、RoPE、残差等仍为浮点，因此不是全模型 W8A8。
- **动作 W2/W4：已适配测试，但没有速度优势。** 从原始 BF16 权重重新量化，保留 Split-K 和融合残差，避免在 W1 上二次量化。可以用于精度取舍，不能宣传降低权重位宽必然提速。
- **动作 INT8：当前实现不采用。** 小 M=16 矩阵的填充、激活量化及失去部分融合残差开销，使其慢于现有 W1；不是硬件不支持 INT8，也不是所有可能的 INT8 内核都不行。
- **Split-K partial FP16/BF16：保留实验。** 只降低部分和存储精度，矩阵累加仍为 FP32；收益不足 1 ms 且有舍入误差。
- **固定定标：不采用。** 3 个输入校准加 25% 余量，独立裁剪检查中未发现溢出，但只代表已测分布；前缀没有稳定收益，输出误差更大。动态定标继续保留，不固定文本长度。
- **INT8 KV 融合注意力：不采用当前实现。** 前缀 K/V 请求内量化一次，10 步共享；在注意力内读取 INT8 并缩放，避免完整反量化张量。但原型标量点积与归约效率低，动作耗时增至约 124 ms。不能用减少字节数替代真实计算性能。原 BF16 prefix 的 Python 引用仍可能存活，所以也不宣称整个进程显存按比例减少。
- KV 独立算子与 CPU FP64 softmax/attention（使用相同量化后 KV）对照：M=1/16/64、P=39/83/189，最大差均低于 0.001625。该测试验证实现，不消除 KV 量化误差。

## 启动命令

在 m5pro 执行；输出目录必须不存在，可改名保留多次记录。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
.venv/bin/python scripts/infer.py \
  --profile speed --prefix-precision all_int8 --text-cache-size 1 \
  --inputs examples/robotwin_ep0_frame0/inputs.npz \
  --prompt-file examples/robotwin_ep0_frame0/prompt.txt \
  --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy \
  --steps 10 --warmup 10 --output records/prefix_int8_infer_new
```

```bash
.venv/bin/python scripts/benchmark.py \
  --profile speed --prefix-precision all_int8 --text-cache-size 1 \
  --reference examples/robotwin_ep0_frame0 \
  --steps 10 --warmup 10 --iterations 50 --profile-iterations 50 \
  --output records/prefix_int8_benchmark_new
```

API：`ImageWAM("weights/bf16", "assets", profile="speed", prefix_precision="all_int8")`。先 `prepare_text(prompt)`，随后 `predict(...)`；文本改变会重新编码。`prefix_precision="default"` 保持原 speed；`mlp_int8` 为较小范围的可选量化。`accuracy` 不接受这些 INT8 参数。

## 代码与证据位置

- 正式模块：`imagewam_mlx/prefix_int8.py`；参数接入 `runtime.py`、`scripts/infer.py`、`scripts/benchmark.py`。
- 全部实验实现：`experiments/precision_all_20261105/`。每个 worker 的第二参数是新输出目录；环境变量 `BASE_RUN=records/precision_all_20261105/baseline_1` 指定 speed 对照。
- 首轮原始记录：`records/precision_all_20261105/`；修复与组合验证、ABBA：`records/precision_followup_20261105/`。完整数组和轨迹保存在 m5pro，轻量 JSON 汇总同步本地。
- 正式基准目录包含源代码快照和 SHA256、每次时间、逐步输出、环境和参考误差。首次候选 157.507 ms 是筛选值；对外推荐引用复测均值 **157.703 ms**。

## 组合方案复现

组合方案仍在实验入口，尚未切换正式默认。combined_w4 兼顾参考误差；combined_partial 的不足 1 ms 增益尚待独立确认。

```bash
BASE_RUN=records/precision_all_20261105/baseline_1 .venv/bin/python experiments/precision_all_20261105/worker.py combined_w4 records/combined_w4_new
BASE_RUN=records/precision_all_20261105/baseline_1 .venv/bin/python experiments/precision_all_20261105/worker.py combined_partial records/combined_partial_new
```
