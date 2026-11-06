# ImageWAM MLX 推理版本与启动索引

开发服务器：m5pro；项目根目录 `/Users/qklj/work/cx/ImageWAM_MLX`。
硬件：Apple M5 Pro，24 GiB 统一内存。验证环境：macOS 27.2、Xcode 27.2 (27B52)、Metal Toolchain 27B47。

## 选择哪个版本

下列速度均为 **三图输入、10 步动作去噪、文本缓存命中**，图片每次重新编码。每轮10次预热、50次正式测量。ABBA 指基线—候选—候选—基线的四个独立进程。

| 启动版本 | Python 环境 | 初筛 E2E ms | ABBA E2E ms | 主要变化与适用范围 |
|---|---|---:|---:|---|
| `stock` | `.venv/bin/python`，MLX 0.32.4 | 166.651 | 160.747（跨运行时对照） | 原有 speed + all_int8 前缀；生产环境保持不变 |
| `runtime` | 下方 `$NEWPY` | 146.969 | 149.187（跨运行时对照） | 新 MLX 源码构建、Metal JIT、NAX 可用；现有算法配置不变，但数值不逐位相同 |
| `native-w1` | `$NEWPY` | 144.844 | 146.903（同运行时对照） | 动作线性层采用原生 W1 路径；本组基线152.195ms，降低3.48% |
| `native-w4` | `$NEWPY` | 145.430 | 147.885（同运行时对照） | 动作权重改为W4，标准样本参考误差更小；本组基线149.453ms，平均降低1.05%，波动较大 |

不同对照的收益不能相加；初筛最低值不是稳定延迟承诺。`native-w1` 标准样本参考 RMSE 约0.51806，`native-w4` 约0.00755；这不是机器人任务成功率测试。新版本均为显式选择的实验入口，没有替换生产默认环境。

完整数据、模块耗时、模型加载和首次预测口径见 [Xcode 后验证报告](docs/optimization/README_XCODE_MLX_VALIDATION_20261106.md)。历史版本见 [量化与速度汇总](docs/optimization/README_INFERENCE_QUANTIZATION_SUMMARY_20261105.md)。

## 直接启动真实推理

以下命令在 **m5pro** 上执行。首次设置：

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
NEWPY="$PWD/records/background_20261106_xcode/builds/route4796/venv/bin/python"
SAMPLE="$PWD/examples/robotwin_ep0_frame0"
```

原有生产运行时（文本缓存命中 ABBA 160.747ms）：

```bash
.venv/bin/python scripts/infer_version.py --variant stock \
  --inputs "$SAMPLE/inputs.npz" --prompt-file "$SAMPLE/prompt.txt" \
  --noise "$SAMPLE/initial_noise_fp32.npy" --steps 10 --warmup 10 \
  --output "records/infer_stock_$(date +%Y%m%d_%H%M%S)"
```

仅切换已测新运行时（ABBA 149.187ms）：

```bash
"$NEWPY" scripts/infer_version.py --variant runtime \
  --inputs "$SAMPLE/inputs.npz" --prompt-file "$SAMPLE/prompt.txt" \
  --noise "$SAMPLE/initial_noise_fp32.npy" --steps 10 --warmup 10 \
  --output "records/infer_runtime_$(date +%Y%m%d_%H%M%S)"
```

速度优先 W1（ABBA 146.903ms）：

```bash
"$NEWPY" scripts/infer_version.py --variant native-w1 \
  --inputs "$SAMPLE/inputs.npz" --prompt-file "$SAMPLE/prompt.txt" \
  --noise "$SAMPLE/initial_noise_fp32.npy" --steps 10 --warmup 10 \
  --output "records/infer_w1_$(date +%Y%m%d_%H%M%S)"
```

精度/性能折中 W4（ABBA 147.885ms）：

```bash
"$NEWPY" scripts/infer_version.py --variant native-w4 \
  --inputs "$SAMPLE/inputs.npz" --prompt-file "$SAMPLE/prompt.txt" \
  --noise "$SAMPLE/initial_noise_fp32.npy" --steps 10 --warmup 10 \
  --output "records/infer_w4_$(date +%Y%m%d_%H%M%S)"
```

`infer_version.py` 默认使用 `speed + all_int8`，并通过版本检查限制到已测 MLX。实验运行时的兼容性放行只影响该进程。`--help` 显示透传的输入参数；额外必需参数是上面的 `--variant`。

输出包含 `action.npy`、`normalized.npy`、`action.csv`、`initial_noise.npy`、`run.json`。其中 `run.json.tested_version` 记录版本、实际 MLX 模块路径、动作适配耗时。`request_ms` 是本次单请求计时，不是表中的50轮平均值。

## 换成自己的文本和三张图

`inputs.npz` 包含 `cam_high`、`cam_left_wrist`、`cam_right_wrist`（uint8、HWC、RGB）和 `state`（float32、14维）。推荐保持 RoboTwin 样本图像尺寸。

- `--prompt-file` 输入完整模型提示词，不额外添加前缀。
- 也可用 `--instruction '你的任务描述'` 替换 `--prompt-file`，入口会添加模型训练所用提示前缀；不要同时传两者。
- 文本长度不固定；仍受 tokenizer/模型本身最大上下文长度限制。验证覆盖原有51案例及额外长文本。
- 同一模型对象中，默认文本缓存大小1；文本改变时重新计算。多次独立启动 CLI 不共享内存缓存。
- 图像特征没有跨请求缓存。`--steps 10` 保持本报告比较口径。
- 精度比较使用同一份 `--noise`；省略时使用 MLX 随机数，不保证与 PyTorch 同 seed 生成相同噪声。
- `--warmup 0` 可测首次预测，但文本准备仍单独计时。完整启动时间查看 `startup_total_ms`。
- 需要原有高精度配置时使用 `.venv/bin/python scripts/infer.py --profile accuracy ...`；本轮没有重测 accuracy，不把上述速度归给它。

## 复现全量性能/数值验证

下面是 **测试入口**，会运行50轮计时和多输入验证；与上面的单次实际推理入口区分使用：

```bash
BG_ALLOW_NEW_MLX=1 \
BASE_RUN="$PWD/records/background_20261105/stock_baseline_attempt1" \
"$NEWPY" experiments/background_20261105/worker.py native_w1 \
  "records/recheck_native_w1_$(date +%Y%m%d_%H%M%S)"
```

将 `native_w1` 改为 `baseline` 或 `native_w4` 可复测另两个版本；stock 的 baseline 使用 `.venv/bin/python` 并省略 `BG_ALLOW_NEW_MLX`。输出目录必须不存在。

其余已测实验路径见下表，它们没有优于最佳原生路径，不作为推荐推理入口：

| 测试 mode | 环境 | 初筛 E2E ms | 改动 |
|---|---|---:|---|
| `qq_mxfp4` | route4796 | 156.585 | 动作网络 MXFP4 权重/激活量化矩阵乘法 |
| `qq_mxfp8` | route4796 | 172.918 | 动作网络 MXFP8 权重/激活量化矩阵乘法 |
| `qq_nvfp4` | route4796 | 156.796 | 动作网络 NVFP4 权重/激活量化矩阵乘法 |
| `affine_w4` | affine4802 | 150.652 | 动作 affine 4位权重，qqmm 激活量化路径 |
| `affine_w8` | affine4802 | 164.842 | 动作 affine 8位权重，qqmm 激活量化路径 |
| `prefix_affine_w4` | affine4802 | 162.983 | 前缀选定线性层改用 affine 4位 qqmm |
| `prefix_affine_w8` | affine4802 | 166.683 | 前缀选定线性层改用 affine 8位 qqmm |

这些 mode 使用同一测试命令。affine4802 的 Python 路径为 `records/background_20261106_xcode/builds/affine4802/venv/bin/python`，不能在 route4796 环境混测后沿用此表结论。

## 后续前缀优化

已在原生 W1 动作路径上固定其余条件，完成偏置融合、归一化/量化融合、二者组合，以及删除固定指数路径无用缓冲写入的适配验证。组合融合 ABBA 为148.593ms，对应基线146.937ms；缓冲裁剪150.290ms，对应基线149.935ms。本轮没有稳定提速，推荐版本不变。

四个实际推理入口均已执行，输出与此前同版本测试的标准输出逐位相同。单次CLI速度与完整实验结果见 [前缀后续验证](docs/optimization/README_PREFIX_FOLLOWUP_20261106.md)。

代码：`experiments/prefix_followup_20261106/`；结果：`records/prefix_followup_20261106/` 及补测目录。失败、修复、负结果均保留，不作为默认优化。

## 最新同版本复测（2026-11-06）

`native-w1` 在前缀分块对照中的两轮均值为 **144.773 ms**（144.529、145.016 ms），历史版本确认均值为146.903ms；这是运行批次差异，不是新增优化收益。分块候选146.535ms，没有提速，保持原推荐入口。完整过程见 [前缀分块验证](docs/optimization/README_PREFIX_TILES_20261106.md)。

## Action 浮点对照（2026-11-06）

新运行时只替换动作专家的实测：原始BF16 179.828ms、W1展开BF16 180.000ms，对应同轮W1 150.171ms；FP16 179.216ms，对应同轮W1 147.503ms。整体改浮点未提速，速度推荐仍为native-w1。BF16/FP16标准样本参考RMSE约0.00220/0.00224。详见 [Action 浮点对照](docs/optimization/README_ACTION_FLOAT_COMPARE_20261106.md)。

## 新保留的精度入口

`--variant action-bf16` 已保留原始BF16 action＋当前all_int8前缀，使用新运行时，已有正式对照均值179.828ms。它与全模型accuracy不同。所有保留版本、性能与启动方式见 [保留版本总表](README_RETAINED_VERSIONS_20261106.md)。

## RoboTwin 多样本精度更新

六版本已完成同批12个episode、36个真实帧、两份噪声共432次推理。相对全模型accuracy的RMSE：native-w1 0.571768、native-w4 0.007701、action-bf16 0.003633。精度/速度折中优先native-w4，更低平均数值偏差用action-bf16；W1仅保留速度极限用途。详情与边界见 [RoboTwin精度报告](README_ROBOTWIN_ACCURACY_20261106.md)，不要用此前单样本误差替代本轮结论。
