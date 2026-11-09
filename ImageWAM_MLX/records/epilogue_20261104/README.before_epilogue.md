[Steel 分块矩阵双段 KV 实验](README_STEEL_KV_20261104.md)：完整模型未提速，默认 speed 保持不变。

[双段 KV 自定义 attention 测试](README_SEGMENTED_KV_20261104.md)：本版未提速，保留现有 speed 默认。

2026-11-04：speed 默认启用 KV 工作区，100组配对平均节省0.865ms；[适配与验证报告](README_STRUCTURAL_20261104.md)。accuracy不变。

后续验证：[分块交替复测](README_INT8_RETEST_20261104.md)、[K 循环屏障移除](README_INT8_BARRIER_20261104.md)。未确认稳定收益，默认配置保持不变。

INT8 分块专项测试见 [2026-11-04 记录](README_INT8_TILES_20261104.md)，默认配置不变。

2026-11-04：原生 INT8、融合、量化 KV/attention 与卷积实验及完整复测见 [低比特验证报告](README_LOWBIT_VALIDATION_20261104.md)。现有默认路径保持不变。

# ImageWAM MLX：M5 Pro 三图动作推理

**量化分支更新：**不传参数或 `--profile accuracy` 使用原 BF16 精度路径；`--profile speed` 使用允许精度退化的混合量化配置。同期各 200 次稳态平均为 **291.082 / 277.018 ms**；默认 162 个数组逐字节一致。启动命令、算子类型/反量化审计、全部候选和输出误差以 [精度与速度分支报告](README_SPEED_QUANTIZATION_20261103.md) 为准。以下保留此前各轮结果，不能直接跨轮相减归因。

进一步的 VAE 诊断定位了 GN 平方阶段的编译差异，并新增默认关闭的精确 pow/查表候选。两轮端到端测试尚未确认稳定收益，详见 [深入诊断记录](README_VAE_DEEP_20261103.md)。

图像编码专项实验新增 `vae_tail_compile=True` 可选模式，完整精度通过但端到端收益复测不稳定，默认关闭。参见 [图像编码实验](README_VAE_OPTIMIZATION_20261103.md)。

**2026-11-03 最终默认版本：**100 次稳态平均 **281.650 ms**、P95 **285.014 ms**；默认 CLI 预热一次并分别记录预热和正式请求。完整优化清单、PyTorch base/dynamic 对照、模块耗时、精度和最新命令见 [当天最终汇总](README_DAILY_FINAL_20261103.md)。下面的 335.098 ms 表格保留为早期历史基线。

项目位置：`/Users/qklj/work/cx/ImageWAM_MLX`。已实现并实际运行 BF16 MLX 推理，输出为 `[16,14]` 动作序列。推理环境不安装、不导入 PyTorch；PyTorch 仅供一次性权重转换和独立参考验证。

当前默认路径为单样本动作推理：Qwen3 前 27 层的第 9/18/27 层特征、FLUX.2 VAE 编码器、视觉前缀 KV 缓存、动作专家及 10 步 flow-matching 去噪。没有量化、文本/图像特征跨输入缓存、步数裁减或固定文本长度。视觉最后一层在取得动作需要的 K/V 后停止。此入口不提供图像生成、训练或多样本批处理。

## 实测环境与早期历史性能

2026-11-03 实测：Mac mini `Mac17,16`，Apple M5 Pro，15 核 CPU、16 核 GPU、24 GiB 统一内存；macOS 27.2、Python 3.12.13、MLX/MLX Metal 0.32.4、Transformers 4.56.1。GPU 与 CPU 共享内存，没有独立显存。设备建议工作集为 17.760 GiB，不能当作独立显存容量。完整信息见 [environment.json](records/environment.json) 和 [依赖锁定](requirements-lock.txt)。

|版本/记录|加载 s|首次推理 ms|稳态平均 ms|P95 ms|
|---|---:|---:|---:|---:|
|MLX BF16，最终复测|2.653|5826.997|335.098|337.482|
|MPS BF16 dynamic，同机重新测试|12.464|5658.243|353.595|370.776|
|MLX BF16，前一轮有效测试|2.914|1147.604|329.444|330.803|

本轮 MLX 比 MPS dynamic 平均延迟低 **5.23%**。这是先后独立进程运行的同机比较，非逐轮交替测试；保留所有稳态样本，不能声称所有场景都提速。两者都执行 6 次预热、20 次无模块钩子的稳态测试、另 5 次同步模块分析。

首次推理在不同进程间有明显波动；最终 MLX 运行的第一个预热也用了 5547.835 ms，之后进入稳定状态。OS 文件缓存、驱动缓存和系统内存状态未受控，加载/首次耗时均不是严格冷启动结论。早期试跑记录不用于宣称更低的正式首次耗时。

端到端边界：三张已解码 RGB 数组、14 维原始状态、完整文本 → CPU 上反归一化动作。包含缩放拼图、归一化、在线 tokenization、文本与图像编码、视觉前缀、10 步去噪、CPU 结果返回；不含视频读取/解码、模型加载、保存文件、仿真执行。初始噪声为同一份 PyTorch CPU seed=42 导出的 FP32 数组，两框架每轮重新使用该噪声。

|MLX 模块，独立同步分析|平均 ms|
|---|---:|
|三图/状态预处理|2.597|
|Tokenization|0.192|
|Qwen 文本编码|44.736|
|VAE 图像编码|54.035|
|视觉前缀/KV|131.747|
|10 步动作去噪|106.091|
|CPU 转换/反归一化|0.161|

模块分析增加同步，不能替代无模块同步的端到端值。最终运行 MLX allocator active 约 14.24 GiB，进程内 MLX peak 约 15.62 GiB；完整字节数在结果 JSON。peak 覆盖加载、预热、测试和中间结果导出，不是仅一次稳态推理；allocator 指标不是系统总内存，也不能与 MPS driver/allocator 数据直接相加或等同。

## 直接运行：已有样例

以下命令均在新项目目录运行。`RUN_TAG` 用于新建独立输出目录；程序拒绝覆盖已有输出。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
RUN_TAG=$(date +%Y%m%d_%H%M%S)
.venv/bin/python scripts/infer.py \
  --inputs examples/robotwin_ep0_frame0/inputs.npz \
  --prompt-file examples/robotwin_ep0_frame0/prompt.txt \
  --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy \
  --output records/$RUN_TAG-infer
```

当前该命令默认先用真实输入预热一次，再执行正式请求；`run.json` 分别记录加载、预热、真正首次前向与正式请求时间。加 `--warmup 0` 关闭预热。这不是稳态基准。旧版 CLI 无预热时的历史记录见 [cli_verified/run.json](records/cli_verified/run.json)。

新输入的 NPZ 包含 `cam_high`、`cam_left_wrist`、`cam_right_wrist`（uint8 HWC RGB）及 `state`（有限 FP32 `[14]`）。可改用 `--instruction '新的任务指令'`，程序会添加与原项目相同的 instruction 前缀；`--prompt-file` 则读取完整 prompt，不额外添加前缀。两种文本方式只能选一种。

不传 `--noise` 时用 MLX RNG 的 `--seed 42` 生成噪声；同一数字种子不意味着与 PyTorch 噪声一致。跨框架精度比较应传同一份 `--noise`。`--horizon` 默认 16，支持 1–64；`--steps` 默认 10，正式性能和精度报告仅覆盖 horizon=16、steps=10。

## 正式性能测试命令

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
RUN_TAG=$(date +%Y%m%d_%H%M%S)
.venv/bin/python scripts/benchmark.py \
  --reference ../ImageWAM/evaluate_results/m5pro_robotwin/20261103_dynamic_profile \
  --output records/$RUN_TAG-benchmark --warmup 6 --iterations 20
```

对应上述 **335.098 ms** 稳态结果：[最终 benchmark.json](records/bf16_final/benchmark.json)。`--reference` 只提供原始图像/状态/任务、同一份初始噪声和用于误差比较的动作结果，不读取或复用参考模型的文本/图像特征。MLX 运行所需的 tokenizer、配置和归一化统计已复制到本项目 `assets/`，模型在 `weights/bf16/`。

保存内容包括每轮动作 NPY、最终动作 NPY/CSV、模块数据、原始输入、初始噪声、`trace.npz` 中间结果、源码快照及源码哈希。GPU 计算通过 `mx.eval` / `mx.synchronize` 完成后再记录耗时。

MPS 同机对照命令（所有结果仍放新项目目录）：

```bash
cd /Users/qklj/work/cx/ImageWAM
RUN_TAG=$(date +%Y%m%d_%H%M%S)
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv-mps/bin/python \
  scripts/m5pro/benchmark_robotwin.py --optimization dynamic \
  --output-dir ../ImageWAM_MLX/records/$RUN_TAG-mps \
  --warmup 6 --iterations 20 --profile-iterations 5
```

对应 **353.595 ms** 稳态结果：[mps_dynamic_retest/benchmark.json](records/mps_dynamic_retest/benchmark.json)。

## 动态输入与精度验证

文本使用原 Qwen tokenizer/chat template，`truncation=False`，序列长度从实际输入得到，不固定为 39、79 或 512。超过 Qwen 配置位置上限报错，不静默截断。RoboTwin 原扫描记录共有 921032 条文本，实际 token 长度 31–79，共 48 种长度（其中没有 78）；本项目执行了每种长度一个代表样本，以及 3 组额外真实观察，共 51 组比较。另验证了 79→31→193→31 长度切换。

三图按既有 RoboTwin compact 布局：头部缩至 256×192，左右腕部各 128×96，最终 H×W=288×256。输入图像原始尺寸可变化，随后遵循这一已验证缩放规则。图像 BF16 归一化舍入顺序已与 MPS 对齐；正式样本图像、状态输入逐值一致。三路摄像头、状态和文本分别扰动均改变输出。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
RUN_TAG=$(date +%Y%m%d_%H%M%S)
.venv/bin/python scripts/validate_cases.py --output records/$RUN_TAG-validation
```

此命令是精度/输入覆盖测试，各样本耗时包含新形状首次运行成本，不能作为稳态平均值。结果见 [validation_v1/validation.json](records/validation_v1/validation.json)：

|比较范围|动作最大绝对差|平均逐样本 MAE|归一化动作最大差|
|---|---:|---:|---:|
|51 组，MLX 对 MPS dynamic|0.01580137|0.00088513|0.015625|
|51 组，MLX 对原 padded MPS|0.01580136|0.00086112|0.015625|

正式性能样本的最终动作最大差为 0.00800064、MAE 0.00104442。**跨框架输出不逐值相等，尚未验证 RoboTwin 闭环任务成功率，也没有用户指定的精度验收阈值。**不能将输出有限、余弦相似度高或单样本误差小等同于任务精度通过。全量语料只做过长度扫描，未执行全量任务推理。

中间结果包含文本特征、VAE latent、选定视觉 KV 层和每一步动作。VAE latent 最大差 0.2890625，最终视觉 KV 最大差 0.50390625；这些差异都保留在 [component_comparison.json](records/bf16_final/component_comparison.json)，后续可以继续定位跨框架 BF16 算子差异。

可重新独立导出 MPS 中间结果并比较：

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
RUN_TAG=$(date +%Y%m%d_%H%M%S)
PYTORCH_ENABLE_MPS_FALLBACK=0 ../ImageWAM/.venv-mps/bin/python scripts/export_mps_trace.py \
  --source ../ImageWAM \
  --reference ../ImageWAM/evaluate_results/m5pro_robotwin/20261103_dynamic_profile \
  --output records/$RUN_TAG-mps-trace
.venv/bin/python scripts/compare_traces.py \
  --mlx records/bf16_final/trace.npz \
  --mps records/$RUN_TAG-mps-trace/trace.npz \
  --output records/$RUN_TAG-trace-comparison.json
```

这两个命令用于数值诊断，不报告正式端到端性能。

## 文件结构与重建

- `imagewam_mlx/`：原生 MLX 文本、VAE、视觉前缀和动作推理代码。
- `scripts/infer.py`：独立输入推理；`scripts/benchmark.py`：性能和输出保存。
- `scripts/validate_cases.py`、`export_mps_trace.py`、`compare_traces.py`：覆盖测试及跨框架诊断。
- `weights/bf16/`：从现有 ImageWAM checkpoint、Qwen、VAE 转换的 BF16 safetensors，带张量形状及 SHA256 清单。不需要另下载独立 FLUX.2 基础 transformer。
- `assets/`：独立运行所需 tokenizer/config/stats；`examples/`：真实三图样例。
- `records/`：全部测试数据；[records/README.md](records/README.md) 标明正式、历史与失败记录。
- `references/pi05-mlx/`：用户提供的参考项目，commit `6a63544d75db45c63f114e9c014c9deff924a7fe`。

当前环境已安装完成，不需再次转换。若重建，先用 Python 3.12 创建新虚拟环境并按 `requirements-lock.txt` 安装。转换脚本必须用包含 PyTorch 的原环境执行，并选一个不存在的目标目录：

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
RUN_TAG=$(date +%Y%m%d_%H%M%S)
../ImageWAM/.venv-mps/bin/python scripts/convert_weights.py \
  --source ../ImageWAM --output weights/$RUN_TAG-bf16
```

转换和资产准备脚本拒绝覆盖已有目标；原 `/Users/qklj/work/cx/ImageWAM` 的模型及配置未修改。模型转换、推理和结果均位于新项目目录，没有使用 `/tmp` 存放开发产物。

实现参考了 [pi05-mlx](https://github.com/papasanimohansrinivas/pi05-mlx) 的前缀 KV 复用、分组件验证及权重加载组织方法。Pi0.5 的 SigLIP/Gemma 架构与本模型不同，本项目没有使用它的模型权重或照搬其速度结论。模型算法按现有 ImageWAM/FLUX.2/Qwen3 源码移植；原 ImageWAM 版本为 `5d4a341ed20a95cdb08f0293f3d44778b9a9e05a`（包含此前本地适配改动），许可证保留在 [LICENSE.ImageWAM](LICENSE.ImageWAM)。模型权重仍受各自原始许可约束。

- [NAX 双段 KV 对照测试](README_NAX_KV_20261104.md)：包含独立进程 ABBA、100 对同进程复核及输出一致性；候选保持实验入口。
