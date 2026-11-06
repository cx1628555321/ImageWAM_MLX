# 我们的 ImageWAM 使用手册：3090 推理测试与适配记录

本文件是我们维护的独立 README，统一记录 3090 服务器上已验证的测试代码、启动命令、参数、输出位置和环境适配。文档放在仓库根目录 `README_3090_inference.md`；原仓库 `README.md` 保持不变。日常运行从下面的速查开始，重新配置环境或追查修改时再看后续详细记录。最近实测日期：2026-10-27。

**已完成适配并跑通原生推理。** `libero_spatial/task 0` 的单回合测试成功，修复 EGL 清理逻辑后用相同参数复测仍为 **1/1 成功**，正常退出。100 项 Python 依赖位于仓库 `.venv`，原有 CUDA/PyTorch 保持不变；AE、Qwen3-4B 和视觉初始化权重已准备并校验。首轮整卡显存采样峰值为 **19,278 MiB（约 18.83 GiB）**。这是单任务启动验证，不是完整 LIBERO benchmark。

## 0. 日常使用速查

### 连接服务器并运行默认测试

在本机连接服务器：

```bash
ssh -p 9022 -o ClearAllForwardings=yes ubuntu@47.109.191.96
```

在服务器执行：

```bash
cd /home/ubuntu/data_1/chenxing/ImageWAM/ImageWAM
bash scripts/flux2/infer_3090_libero.sh
```

脚本直接使用仓库 `.venv/bin/python`，无需手动激活 conda/venv 或重复安装依赖。默认使用 GPU 0、BF16、`libero_spatial/task 0`、1 个回合，预测 16 步动作、执行 12 步后重规划、每次去噪 10 步。以下命令均在上述仓库目录执行。

### inference_mode、编译、TorchScript 与 CUDA Graph 对比

新增执行方式比较入口：

```bash
bash scripts/flux2/compare_3090_execution_modes.sh
```

默认比较 `eager inference_mode compile torchscript cudagraph`；只比较原生基线和 inference_mode 可设置 `EXECUTION_MODES="eager inference_mode"`。具体生效范围、正确性检查、加载/准备/预热、显存和五项模块对照见 [五种方式统一对比与结论](README_3090_execution_modes.md#comparison-summary)。普通推理与下方性能入口默认仍使用 eager。

当前统一结果（2026-10-27，RTX 3090，BF16；每项 20 次稳态推理）：

| 方式 | E2E 平均 ms | 相对 Eager 延迟下降 | allocated 峰值 GiB |
|---|---:|---:|---:|
| Eager / no_grad | 400.409 | 0.00% | 17.360 |
| Eager + inference_mode | 383.178 | 4.30% | 17.360 |
| torch.compile | 260.685 | 34.90% | 17.281 |
| TorchScript | 282.798 | 29.37% | 18.399 |
| CUDA Graph | 316.240 | 21.02% | 18.027 |

Eager/inference_mode 为最近相邻测试，其他三项沿用同日已验证实验；所有加速百分比统一使用 400.409 ms 的 Eager 基线。Eager 有短时抬升且保留全部样本；inference_mode 按中位数的改善为 3.57%。compile 的稳态耗时最低，但本次准备与验证耗时约 86 秒。完整模块、加载与精度对照见上面的统一章节。

结果目录：`evaluate_results/3090_modes/modes_20261027_inference_verified/`，可直接查看其中 `comparison.md`。

### 一次加载、多次推理性能统计

```bash
bash scripts/flux2/profile_3090_inference.sh
```

默认仅加载一次完整模型，先预热 6～20 次，随后统计 20 次端到端推理，再在同一模型上采集 10 次模块计时。输出保存在 `evaluate_results/3090_profile/<RUN_ID>/`，优先查看 `summary.md`，机器可读统计见 `benchmark.json` 和 CSV。详细口径、参数与实测数据见第 9 节；下载地址、固定版本和加载路径见第 3.0 节。

增加采样次数：

```bash
BENCH_ITERS=50 PROFILE_ITERS=20 WARMUP_MAX=30 \
RUN_ID="profile_n50_$(date +%Y%m%d_%H%M%S)_$$" \
  bash scripts/flux2/profile_3090_inference.sh
```

性能脚本拒绝复用已有输出目录。它使用固定的真实双相机观测测量动作生成延迟，回合成功率仍由上面的 `infer_3090_libero.sh` 测试。

### 常用运行方式

指定任务和回合数，并给本次运行命名（示例参数，已验证基线仍是 task 0 的单回合）：

```bash
TASK_SUITE=libero_spatial TASK_ID=0 NUM_TRIALS=5 \
RUN_ID="spatial_task0_n5_$(date +%Y%m%d_%H%M%S)_$$" \
  bash scripts/flux2/infer_3090_libero.sh
```

调整去噪步数做对比测试：

```bash
INFERENCE_STEPS=20 SEED=42 \
RUN_ID="steps20_$(date +%Y%m%d_%H%M%S)_$$" \
  bash scripts/flux2/infer_3090_libero.sh
```

只查看最终 Hydra 配置，不运行模型前向：

```bash
bash scripts/flux2/infer_3090_libero.sh --cfg job
```

| 常用环境变量 | 当前启动脚本默认值 | 用法 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0` | 选择物理 GPU，程序内部使用逻辑 `cuda:0` |
| `TASK_SUITE` / `TASK_ID` | `libero_spatial` / `0` | 选择任务套件和任务编号 |
| `NUM_TRIALS` | `1` | 当前任务的回合数，不是任务数量 |
| `INFERENCE_STEPS` | `10` | 模型去噪迭代数，不是仿真步数 |
| `ACTION_HORIZON` / `REPLAN_STEPS` | `16` / `12` | 预测动作数 / 每次实际执行动作数 |
| `SEED` / `QWEN_CONTEXT_LEN` | `42` / `512` | 随机种子 / 文本上下文长度 |
| `RUN_ID` | 自动生成时间戳和 PID | 为输出目录命名，日常实验优先使用此项 |
| `CKPT_PATH` / `DATASET_STATS_PATH` | `../flux2_klein_4b/model.pt` / 同目录统计文件 | 更换策略时同时指定其匹配的归一化统计 |

表中是启动脚本覆盖后的值；底层 YAML 中 `num_trials=50`、`replan_steps=10` 等值不会成为本脚本的实际默认值。重复使用同一 `RUN_ID` 会复用输出目录并覆盖同名日志、结果与配置，因此示例加入时间戳和 PID。

### 日志、结果和视频在哪里

当前 `model.pt` 的普通运行输出目录为：

```text
evaluate_results/3090/model/<RUN_ID>/
├── run.log
├── eval_libero_single.log
└── libero_spatial/
    ├── eval_config_gpu0_task0.yaml
    ├── gpu0_task0_results.json
    └── videos/*.mp4
```

以启动时打印的 `Native LIBERO inference output` 为准；其他任务会替换套件目录和 `task0` 编号。`run.log` 查看加载/执行异常，结果 JSON 查看 `successes` 与 `total_episodes`，`videos/` 查看实际 rollout。默认关闭的是额外的未来预测视频，实际 rollout MP4 仍会保存。

已验证结果保留在 `evaluate_results/3090/model/verified_3090_20261027/`，可直接查看：

```bash
cat evaluate_results/3090/model/verified_3090_20261027/libero_spatial/gpu0_task0_results.json
ls -lh evaluate_results/3090/model/verified_3090_20261027/libero_spatial/videos/
```

历史首轮目录中的 `run_metrics.json` 和 `gpu_samples.csv` 来自当时的外层监控，普通启动脚本不会自动生成这两项。自定义 `OUTPUT_DIR` 时，脚本可能插入 checkpoint 标签，且该变量优先于 `RUN_ID`；日常命名优先使用 `RUN_ID`。checkpoint 和输出路径通过环境变量设置，不追加 `ckpt=`、`EVALUATION.output_dir=` 或 `hydra.run.dir=`。

### 测试代码入口和查找索引

| 查找内容 | 文件 |
|---|---|
| 我们的统一启动入口、默认参数和路径 | [scripts/flux2/infer_3090_libero.sh](scripts/flux2/infer_3090_libero.sh) |
| 执行方式比较与实测 | [README_3090_execution_modes.md](README_3090_execution_modes.md)、[scripts/flux2/compare_3090_execution_modes.sh](scripts/flux2/compare_3090_execution_modes.sh) |
| 性能统计启动、路径与采样参数 | [scripts/flux2/profile_3090_inference.sh](scripts/flux2/profile_3090_inference.sh) |
| 一次加载、预热、重复推理与报告 | [scripts/flux2/benchmark_3090_inference.py](scripts/flux2/benchmark_3090_inference.py) |
| 各模型加载计时、CUDA event 记录与统计 | [scripts/flux2/profile_3090_timing.py](scripts/flux2/profile_3090_timing.py) |
| 原生推理模块的临时计时钩子 | [scripts/flux2/profile_3090_modules.py](scripts/flux2/profile_3090_modules.py) |
| 原生评测：加载策略、执行任务、保存结果；本次增加环境关闭逻辑 | [experiments/libero/eval_libero_single.py](experiments/libero/eval_libero_single.py) |
| 模型工厂入口 | [src/imagewam/runtime.py](src/imagewam/runtime.py) |
| ImageWAM 模型与 FLUX.2 动作推理实现 | [src/imagewam/models/backbones/imagewam.py](src/imagewam/models/backbones/imagewam.py) |
| 模型结构与任务配置 | [configs/model/imagewam_flux2_klein_4b_base.yaml](configs/model/imagewam_flux2_klein_4b_base.yaml)、[configs/task/libero_flux2_klein_4b_base_imagewam.yaml](configs/task/libero_flux2_klein_4b_base_imagewam.yaml) |
| 仿真评测基础配置 | [configs/sim_libero_omnigen2.yaml](configs/sim_libero_omnigen2.yaml) |
| uv 环境安装与保护检查 | [scripts/setup/setup_3090_inference.sh](scripts/setup/setup_3090_inference.sh)、[scripts/setup/guard_3090_environment.py](scripts/setup/guard_3090_environment.py) |
| 直接依赖与精确版本锁 | [requirements/3090-inference.in](requirements/3090-inference.in)、[requirements/3090-inference.lock](requirements/3090-inference.lock) |
| LIBERO 本地配置与初始化状态兼容性修改 | [scripts/flux2/configure_3090_libero.py](scripts/flux2/configure_3090_libero.py) |

详细记录按章节查找：第 1 节服务器与权重路径；第 2 节 uv 环境和依赖；第 3 节模型资产下载、校验与转换；第 4 节 LIBERO/EGL；第 5 节原生调用链与完整参数；第 6 节全部改动清单；第 7 节环境保护证据；第 8 节回合实测结果；第 9 节推理性能统计。

## 1. 服务器、代码和原始权重

| 项目 | 已确认值 |
|---|---|
| SSH | `ssh -p 9022 ubuntu@47.109.191.96` |
| ImageWAM 仓库 | `/home/ubuntu/data_1/chenxing/ImageWAM/ImageWAM` |
| ImageWAM 基线 commit | `5d4a341ed20a95cdb08f0293f3d44778b9a9e05a` |
| 策略权重目录 | `/home/ubuntu/data_1/chenxing/ImageWAM/flux2_klein_4b` |
| 策略 checkpoint | `model.pt`，9,035,379,577 bytes，约 8.41 GiB |
| checkpoint 内容 | step `34720`，完整 `mot` 与 proprio 参数，BF16 |
| 配套文件 | 同目录 `dataset_stats.json`、`config.yaml` |
| GPU | NVIDIA GeForce RTX 3090，24,576 MiB |
| NVIDIA 驱动 | `590.48.01` |
| 系统 CUDA 编译器 | `nvcc` 显示 CUDA `12.0` |

`model.pt` 是策略 checkpoint；归一化统计必须使用它旁边的 `dataset_stats.json`。不能只凭文件名，用其他任务或其他模型的统计替代。

下文命令在服务器仓库目录执行，统一使用这些变量：

```bash
export IMAGEWAM_ROOT=/home/ubuntu/data_1/chenxing/ImageWAM/ImageWAM
export IMAGEWAM_BASE_PYTHON=/home/ubuntu/miniconda3/envs/py312/bin/python
export IMAGEWAM_UV_BIN=/home/ubuntu/.local/bin/uv
export IMAGEWAM_PYTHON="$IMAGEWAM_ROOT/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export CKPT_PATH=/home/ubuntu/data_1/chenxing/ImageWAM/flux2_klein_4b/model.pt
export DATASET_STATS_PATH=/home/ubuntu/data_1/chenxing/ImageWAM/flux2_klein_4b/dataset_stats.json
cd "$IMAGEWAM_ROOT"
```

## 2. 现有环境与仓库内适配

| 组件 | 采用的现有版本 | 来源 |
|---|---|---|
| Python | `3.12.13` | `/home/ubuntu/miniconda3/envs/py312` |
| PyTorch | `2.8.0+cu128` | 同上 |
| torchvision | `0.23.0+cu128` | 同上 |
| torchaudio | `2.8.0+cu128` | 同上 |
| Triton | `3.4.0` | 同上 |
| uv | `0.9.27` | 现有工具 |
| 推理环境 | 仓库 `.venv`，`--system-site-packages` | 新建的隔离环境 |

使用现有 Python 创建 `.venv`，继承基础环境中的 PyTorch/CUDA 相关包，其他适配依赖安装到仓库 `.venv`。以下为创建时使用的命令；已有 `.venv` 时直接复用，不要重复创建覆盖：

```bash
"$IMAGEWAM_UV_BIN" venv --python "$IMAGEWAM_BASE_PYTHON" --system-site-packages .venv
```

系统 `nvcc 12.0` 与 PyTorch wheel 的 `cu128` 表示不同组件：前者是主机编译工具链，后者是该 PyTorch 构建使用的 CUDA 版本。此次适配保留两者，不安装驱动、不升级系统 CUDA，也不编译额外 CUDA 扩展。运行兼容性仍需以实际 CUDA 推理检查为准。

原始 `pyproject.toml` 指定 `torch==2.7.1`、`torchvision==0.22.1`，并通过 uv 配置指向 `cu118`。因此本方案不执行 `uv sync`，也不将项目做 editable 安装。ImageWAM、FLUX.2 和 LIBERO 均通过明确的 `PYTHONPATH` 引入，原始 `pyproject.toml` 与 `uv.lock` 保持原样。

FLUX.2 使用 `transformers==4.56.1`；它要求的 `huggingface-hub` 下界是 `0.34.0`，与原始项目的 `huggingface-hub==0.30.0` 不兼容。此次 profile 采用 `huggingface-hub==0.35.3`、`tokenizers==0.22.1`。保留 NumPy `1.26.4`，OpenCV 使用兼容该版本的 `opencv-python-headless==4.11.0.86`。

使用以下已提供的适配文件：

- `requirements/3090-inference.in`：推理直接依赖及版本。
- `requirements/3090-inference.lock`：已解析的 100 项依赖，默认安装来源。
- `scripts/setup/setup_3090_inference.sh`：创建/复用 `.venv`、解析、安装和导入检查。
- `scripts/setup/guard_3090_environment.py`：基础环境快照、GPU 包排除、lock 检查和前后比较。

复现安装入口：

```bash
UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
UV_CONCURRENT_DOWNLOADS=4 UV_HTTP_TIMEOUT=120 UV_HTTP_RETRIES=3 \
BASE_PYTHON="$IMAGEWAM_BASE_PYTHON" \
  bash scripts/setup/setup_3090_inference.sh
```

这里仅通过本次进程的 `UV_INDEX_URL` 切换依赖下载源，不修改系统 pip 配置。脚本默认复制现有 `requirements/3090-inference.lock` 到本次日志目录，只有设置 `UPDATE_LOCK=1` 或 lock 不存在时才重新执行 `uv pip compile --quiet`。当前重试使用相同的 100 项版本锁，没有因下载问题调整依赖版本。

脚本设置 `UV_NO_CONFIG=1`，避免原项目的 CUDA wheel index 配置干扰本次解析；uv 缓存在仓库 `.cache/uv`。它读取基础环境中已有的 `torch`、`torchvision`、`torchaudio`、`triton` 和 `nvidia-*`，在重新解析时通过 `--excludes` 将其排除。lock 的每项依赖必须固定版本，且不能出现受保护的 GPU 包；通过检查后，先 `uv pip install --no-deps --dry-run`，再以 `--no-deps` 安装同一份 lock。

`opencv-python` 也从解析中排除，由明确固定的 headless 包提供 `cv2`，避免两种 wheel 安装重叠文件。安装前后由 guard 比较基础环境及 `.venv` 继承的包版本/位置、Python 版本、Torch CUDA 构建版本和项目配置摘要；即使 setup 中途失败，退出处理仍会尝试保留快照和比较结果。

每次安装日志目录为 `artifacts/3090_setup/run_<UTC时间>_<PID>/`，包含 `installation.log`、`requirements.lock`、`inherited-gpu-packages.txt`、`base_before.json`、`base_after.json`、`venv_before.json` 和 `venv_after.json`。完整传递依赖以该次生成的 `requirements.lock` 为准，下面是 profile 的直接依赖版本，不能把“已经声明”当作“已经安装成功”：

| 用途 | 直接依赖及版本 |
|---|---|
| 模型/文本 | `accelerate 1.12.0`、`einops 0.8.1`、`huggingface-hub 0.35.3`、`safetensors 0.5.3`、`tokenizers 0.22.1`、`transformers 4.56.1` |
| 配置/数据 | `datasets 3.6.0`、`hydra-core 1.3.2`、`omegaconf 2.3.0`、`numpy 1.26.4`、`pandas 2.2.3`、`pyarrow 23.0.0`、`h5py 3.13.0`、`jsonlines 4.0.0` |
| 图像/视频 | `av 16.0.1`、`imageio 2.37.0`、`imageio-ffmpeg 0.6.0`、`pillow 12.0.0`、`opencv-python-headless 4.11.0.86` |
| 仿真/数值 | `mujoco 3.3.2`、`robosuite 1.4.0`、`bddl 1.0.1`、`gym 0.25.2`、`easydict 1.13`、`thop 0.1.1.post2209072238`、`future 1.0.0`、`cloudpickle 3.1.1`、`scipy 1.15.3`、`numba 0.61.2`、`matplotlib 3.10.1` |
| 通用工具 | `boto3 1.35.99`、`gitpython 3.1.45`、`packaging 25.0`、`regex 2025.11.3`、`rich 14.2.0`、`termcolor 2.5.0`、`tqdm 4.66.5`、`typing-extensions 4.15.0`、`python-dotenv 1.1.1` |

安装尝试与网络处理记录：

| 日志目录（相对于 `artifacts/3090_setup/`） | 当前结果 |
|---|---|
| `run_20261027T065926Z_1284472` | 直连约 3 分钟只取得 3 个小包，停止本次 uv 进程；退出码 143，基础环境和 `.venv` 的受保护项比较均通过 |
| `run_20261027T070616Z_1292054` 及相关重试日志 | 使用 `http://127.0.0.1:7897` 作为进程级 HTTP/HTTPS 代理；`llvmlite`、`pyarrow` 下载出现 TLS EOF，失败后的 guard 比较通过 |
| `install_mirror_console.log`、`run_20261027T070831Z_1294703/installation.log` | 同一份 lock 改用阿里云 PyPI 镜像安装成功，退出码 0；100 项依赖安装、核心/数据导入和基础环境保护检查均通过 |

PyPI simple 索引可访问，但 `files.pythonhosted.org` 的文件下载通过默认路径和显式代理均出现 `SSL_ERROR_SYSCALL`。这是依赖文件的网络传输失败，不能描述为版本解析冲突；镜像安装已经以退出码 0 和导入检查确认成功。

不要直接安装现有 LIBERO 目录的旧 `requirements.txt`：其中固定了 `numpy==1.22.4`、`transformers==4.21.1` 等版本。应将仿真所需依赖纳入此次仓库内的适配 requirements。

## 3. 资产来源和准备方式

以下是本次采用的仓库内路径。

| 资产 | 来源或当前状态 | 复现路径 |
|---|---|---|
| ImageWAM 策略 | 已存在，完整 checkpoint | `$CKPT_PATH` |
| 归一化统计 | 已存在，随策略发布 | `$DATASET_STATS_PATH` |
| FLUX.2 源码 | 已独立 clone 并切到固定 commit | `third_party/flux2` |
| FLUX 视觉参数 | 已导出，149 张量，严格加载通过 | `checkpoints/3090/flux2_imagewam_visual.safetensors` |
| 公开 AE 原文件 | 已下载、SHA-256 校验通过 | `checkpoints/3090/public_vae/vae/diffusion_pytorch_model.safetensors` |
| 原生格式 AE | 已转换，251 张量，严格加载通过 | `checkpoints/3090/ae.safetensors` |
| Qwen3-4B 文本模型 | 10 文件、3 权重分片下载校验完成 | `checkpoints/3090/Qwen3-4B` |
| LIBERO 源码及资产 | 已从共享目录复制，约 427 MB | `third_party/LIBERO` |

### 3.0 模型下载地址、固定版本与实际加载路径

本节同时用于普通推理和性能统计。下表中的路径是当前 3090 服务器的默认值；修改启动脚本环境变量后，以运行输出保存的最终配置与模型路径为准。

```bash
export IMAGEWAM_ROOT=/home/ubuntu/data_1/chenxing/ImageWAM/ImageWAM
export IMAGEWAM_PYTHON="$IMAGEWAM_ROOT/.venv/bin/python"
cd "$IMAGEWAM_ROOT"
```

| 实际加载组件 | 官方下载地址与固定 revision | 当前加载路径 / 参数 |
|---|---|---|
| ImageWAM 完整策略：训练后的 FLUX 视觉分支、ActionDiT、proprio encoder | [yuyangalin/ImageWAM-FLUX.2-4B-LIBERO](https://huggingface.co/yuyangalin/ImageWAM-FLUX.2-4B-LIBERO)，`5d1b7651b6a0e863fe0556ddc409213c38f6ee85` | `CKPT_PATH=/home/ubuntu/data_1/chenxing/ImageWAM/flux2_klein_4b/model.pt`；Hydra `ckpt`；`model.load_checkpoint()` 恢复完整策略 |
| 策略归一化统计 | 同一 ImageWAM 发布仓库、同一 revision 的 `dataset_stats.json` | `DATASET_STATS_PATH=/home/ubuntu/data_1/chenxing/ImageWAM/flux2_klein_4b/dataset_stats.json`；Hydra `EVALUATION.dataset_stats_path` |
| FLUX.2 视觉初始化权重 | 从上面的 `model.pt` 导出，当前方案没有单独下载另一份视觉基础权重 | `FLUX2_MODEL_PATH=$IMAGEWAM_ROOT/checkpoints/3090/flux2_imagewam_visual.safetensors`；Hydra `model.flux2_model_path` |
| FLUX.2 图像 AE | [BFL 官方公开 VAE 文件](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B/tree/a3b4f4849157f664bdbc776fd7453c2783562f4d/vae)，`a3b4f4849157f664bdbc776fd7453c2783562f4d` | 下载原文件：`$IMAGEWAM_ROOT/checkpoints/3090/public_vae/vae/diffusion_pytorch_model.safetensors`；运行加载转换结果：`FLUX2_AE_MODEL_PATH=$IMAGEWAM_ROOT/checkpoints/3090/ae.safetensors`，Hydra `model.ae_model_path` |
| Qwen3-4B 文本编码器及 tokenizer | [Qwen 官方 Hugging Face 文件](https://huggingface.co/Qwen/Qwen3-4B/tree/1cfa9a7208912126459214e8b04321603b3df60c)，`1cfa9a7208912126459214e8b04321603b3df60c`；传输备用源为 [Qwen 官方 ModelScope](https://modelscope.cn/models/Qwen/Qwen3-4B) | `FLUX2_QWEN3_MODEL_SPEC=$IMAGEWAM_ROOT/checkpoints/3090/Qwen3-4B`；Hydra `model.qwen3_model_spec`；从该目录加载 config、tokenizer 和 3 个 safetensors 分片 |

ImageWAM 策略的 revision 来自现有下载目录 `.cache/huggingface/download/` 下 `model.pt.metadata`、`dataset_stats.json.metadata`、`config.yaml.metadata` 和 `README.md.metadata`；四者记录相同 commit。策略模型卡与原仓库 README 均指向上述发布仓库。其下载元数据记录的 `model.pt` ETag 为 `53620f93f8772d2047f27f9ca9b4a1f56cdd7932de566282e0f4a80cfff21cb5`；这是下载记录中的哈希，本次文档核对没有重新计算 9 GB 文件的完整 SHA-256。

#### 完整策略与配套统计：固定版本下载命令

这是重新部署时的复现命令；当前服务器已有这些文件，无需重复下载。使用仓库 `.venv` 中已经安装的 CLI，不需要修改系统依赖。

```bash
HF_HOME="$IMAGEWAM_ROOT/.cache/huggingface" \
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_XET=1 \
  "$IMAGEWAM_ROOT/.venv/bin/huggingface-cli" download \
    yuyangalin/ImageWAM-FLUX.2-4B-LIBERO \
    model.pt dataset_stats.json config.yaml README.md \
    --repo-type model \
    --revision 5d1b7651b6a0e863fe0556ddc409213c38f6ee85 \
    --local-dir "$IMAGEWAM_ROOT/../flux2_klein_4b"
```

CLI 参数已在当前 `.venv` 中核对；这里记录的是固定版本的复现命令，并不声称这是用户最初下载时使用的命令。`config.yaml` 是发布时的训练配置记录；日常测试与性能统计使用当前仓库的 `configs/task/libero_flux2_klein_4b_base_imagewam.yaml`，不直接实例化该发布文件内的历史绝对路径。

#### 视觉初始化权重：从策略导出

```bash
"$IMAGEWAM_PYTHON" scripts/flux2/export_3090_flux2_weights.py \
  --checkpoint "$IMAGEWAM_ROOT/../flux2_klein_4b/model.pt" \
  --output "$IMAGEWAM_ROOT/checkpoints/3090/flux2_imagewam_visual.safetensors" \
  --flux2-src-path "$IMAGEWAM_ROOT/third_party/flux2" \
  --variant klein-base-4b
```

已生成该文件时直接复用，导出脚本会拒绝覆盖。模型工厂先用此文件建立视觉分支，再加载完整 `model.pt` 恢复训练后的联合策略。ActionDiT 与 proprio encoder 的实际训练参数都来自 `model.pt`，当前配置 `model.action_dit_pretrained_path=null`，没有额外的 ActionDiT 模型下载或独立 proprio checkpoint。

#### AE：官方公开文件下载及原生格式转换

```bash
"$IMAGEWAM_PYTHON" scripts/setup/download_3090_ae.py \
  --proxy http://127.0.0.1:7897

PYTHONPATH="$IMAGEWAM_ROOT/third_party/flux2/src:$IMAGEWAM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$IMAGEWAM_PYTHON" scripts/flux2/convert_public_flux2_ae.py \
    "$IMAGEWAM_ROOT/checkpoints/3090/public_vae/vae/diffusion_pytorch_model.safetensors" \
    "$IMAGEWAM_ROOT/checkpoints/3090/ae.safetensors"
```

下载脚本固定上表 BFL revision，同时下载 `vae/config.json`；校验权重大小、SHA-256 及配置 Git blob SHA-1。已存在原文件可用 `--verify-only` 本地校验；无需代理时传 `--proxy ''`。转换脚本仅在需要生成原生格式时运行，运行加载的是转换后的 `ae.safetensors`。下载与转换细节见第 3.3 节。

#### Qwen3-4B：固定内容版本，支持两种传输源

当前服务器成功采用的 ModelScope 传输方式：

```bash
"$IMAGEWAM_PYTHON" scripts/setup/download_3090_qwen.py \
  --proxy http://127.0.0.1:7897 --workers 3 --transport modelscope
```

Hugging Face 连接正常时可用同一个脚本直接下载固定 revision：

```bash
"$IMAGEWAM_PYTHON" scripts/setup/download_3090_qwen.py \
  --proxy http://127.0.0.1:7897 --workers 3 --transport huggingface
```

该脚本先取得固定 HF revision 的文件列表与哈希；ModelScope 仅作为文件传输源，使用 `https://www.modelscope.cn/models/Qwen/Qwen3-4B/resolve/master/<filename>`，每个文件仍须与固定 HF revision 的大小和 SHA-256 / Git blob SHA-1 匹配。这里没有将 ModelScope `master` 描述为固定 revision：其独立 commit 未记录，内容一致性由逐文件校验保证。ModelScope 文件流量直连，`--proxy` 供 HF 元数据请求使用。已有文件可加 `--verify-only`，它仍需联网查询固定版本元数据。

#### 当前模型路径的一次性显式配置

普通启动脚本已内置以下默认值。需要明确确认或切换权重时，可以在启动前设置：

```bash
export CKPT_PATH="$IMAGEWAM_ROOT/../flux2_klein_4b/model.pt"
export DATASET_STATS_PATH="$IMAGEWAM_ROOT/../flux2_klein_4b/dataset_stats.json"
export FLUX2_SRC="$IMAGEWAM_ROOT/third_party/flux2"
export FLUX2_MODEL_PATH="$IMAGEWAM_ROOT/checkpoints/3090/flux2_imagewam_visual.safetensors"
export FLUX2_AE_MODEL_PATH="$IMAGEWAM_ROOT/checkpoints/3090/ae.safetensors"
export FLUX2_QWEN3_MODEL_SPEC="$IMAGEWAM_ROOT/checkpoints/3090/Qwen3-4B"
```

代码入口 `ImageWAM.from_flux2_klein_pretrained()` 负责视觉权重、ActionDiT 初始化、AE、Qwen/tokenizer 和 proprio 模块构造；随后 `ImageWAM.load_checkpoint()` 读取完整策略，恢复 `mot` 和 `proprio_encoder`。因此性能统计中的“ActionDiT 初始化”与“完整策略权重恢复”应单独解释，不能将初始化耗时误称为从独立 ActionDiT 文件加载的耗时。

### 3.1 FLUX.2 源码

已从服务器现有源码仓库创建独立副本，并使用原仓库 README 指定的固定版本。`--no-hardlinks` 避免与源仓库共享 Git 对象硬链接：

```bash
git clone --no-hardlinks /home/ubuntu/data_1/z_image/flux2 third_party/flux2
git -C third_party/flux2 checkout 50fe5162777813d869182b139e83b10743caef15
git -C third_party/flux2 remote set-url origin https://github.com/black-forest-labs/flux2.git
git -C third_party/flux2 rev-parse HEAD
export FLUX2_SRC="$IMAGEWAM_ROOT/third_party/flux2"
```

已有该目录时，先核验其 commit 和工作区状态；不要对包含其他改动的副本直接执行切换。源码通过 `PYTHONPATH` 引入，依赖由本仓库环境统一管理。

此前未完成的网络 clone 已移至 `artifacts/3090_setup/flux2-network-clone.partial` 留存；它不参与运行。

### 3.2 从完整策略导出视觉参数

新增脚本 `scripts/flux2/export_3090_flux2_weights.py` 读取 checkpoint 中的 `mot`，仅提取规范前缀 `mixtures.video.transformer.`，导出原生 FLUX.2 loader 可加载的视觉 transformer safetensors。这样可以避免为了初始化，再下载一份之后会被完整策略覆盖的 FLUX.2 base 4B 参数。

导出脚本使用 CPU `torch.load(..., weights_only=True, mmap=True)`，不量化、不转换 dtype、不调整张量形状。它使用固定版本 FLUX.2 的 meta 模型核验键名和形状，执行严格加载检查，并排除 wrapper 中重复注册的 block 别名。输出旁边的 JSON 记录源 checkpoint、训练 step、张量形状/dtype、导出大小和检查结果。

本次已导出 **149 个张量**，文件大小 **7,751,105,848 bytes**，严格加载检查通过。

```bash
export FLUX2_MODEL_PATH="$IMAGEWAM_ROOT/checkpoints/3090/flux2_imagewam_visual.safetensors"
"$IMAGEWAM_PYTHON" scripts/flux2/export_3090_flux2_weights.py \
  --checkpoint "$CKPT_PATH" \
  --output "$FLUX2_MODEL_PATH" \
  --flux2-src-path "$FLUX2_SRC" \
  --variant klein-base-4b
```

脚本拒绝覆盖已有输出。已完成导出时应检查 `flux2_imagewam_visual.safetensors.json`，而不是反复执行导出。JSON 的 `tensor_manifest_sha256` 是键名、形状和 dtype 清单的摘要，不等同于 safetensors 文件内容的 SHA-256；需要文件摘要时另行计算。

推理仍然走原生 factory，并随后调用 `model.load_checkpoint(model.pt)` 恢复完整 MoT、动作分支和 proprio 参数。导出的视觉文件只是初始化入口，不能代替完整策略 checkpoint。本次完整 MoT 加载已确认 `missing_keys=0 unexpected_keys=0`，proprio 参数严格加载成功。

### 3.3 公开 AE 下载、校验和格式转换

最初访问 `black-forest-labs/FLUX.2-dev` 的受限 AE 返回 403。此次实际采用 BFL 官方公开的 `FLUX.2-klein-base-4B` 仓库内 Diffusers 格式 VAE，下载后通过转换脚本恢复原生 FLUX.2 参数命名和 attention kernel 形状，不再依赖受限 AE 的下载权限。

| 字段 | 已确认值 |
|---|---|
| 来源 | `black-forest-labs/FLUX.2-klein-base-4B` |
| revision | `a3b4f4849157f664bdbc776fd7453c2783562f4d` |
| 文件 | `vae/diffusion_pytorch_model.safetensors` |
| 下载后大小 | `168120878` bytes |
| SHA-256 | `ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04` |
| 原始 dtype | BF16 |
| 原文件状态 | 下载完成且 SHA-256 校验通过 |

下载与校验由 `scripts/setup/download_3090_ae.py` 实现。它固定上述 revision，下载权重和 `vae/config.json`，校验权重 SHA-256、大小及配置文件 Git blob SHA-1，并将结果保存到 `artifacts/3090_setup/ae_download.json`。缓存和下载目录均位于本仓库；使用匿名公开访问，不读取已有 token。

```bash
"$IMAGEWAM_PYTHON" scripts/setup/download_3090_ae.py \
  --proxy http://127.0.0.1:7897
```

已有文件可增加 `--verify-only` 进行本地复核；需要直连时传入 `--proxy ''`。

`scripts/flux2/convert_public_flux2_ae.py` 对照 [Diffusers 官方 FLUX2 VAE 转换规则](https://github.com/huggingface/diffusers/blob/main/scripts/convert_flux2_to_diffusers.py) 执行逆转换：重命名参数，将 attention 的二维投影权重恢复为原生 1×1 卷积形状；保持公开文件的 BF16 dtype 和数值。它以固定 FLUX.2 源码中的 `AutoEncoder(AutoEncoderParams())` 为依据检查全部键名/形状。本次 **251 个张量严格加载检查已通过**，转换输出大小为 **168,118,958 bytes**；完整策略验证另见末尾实测结果。

```bash
export FLUX2_AE_MODEL_PATH="$IMAGEWAM_ROOT/checkpoints/3090/ae.safetensors"
PYTHONPATH="$FLUX2_SRC/src:$IMAGEWAM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$IMAGEWAM_PYTHON" scripts/flux2/convert_public_flux2_ae.py \
  "$IMAGEWAM_ROOT/checkpoints/3090/public_vae/vae/diffusion_pytorch_model.safetensors" \
  "$FLUX2_AE_MODEL_PATH"
```

转换脚本只需执行一次；已有输出时先检查此前日志和文件，避免覆盖。这里的 SHA-256 属于下载原文件，不能用来校验经过重命名和重新序列化后的 `ae.safetensors`。

### 3.4 Qwen3-4B 文本编码器

此次有界资产检查未找到可直接复用的普通文本版 Qwen3-4B，因此固定以下 revision 下载到本仓库。现有 Qwen TTS、VL 或 GGUF 文件不作为此文本编码器的替代。

下载脚本 `scripts/setup/download_3090_qwen.py` 固定 `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`，明确选择配置、tokenizer、索引和模型分片。默认 3 个并行下载 worker，可在 1～4 范围内调整；这些是下载线程，与推理 worker 数无关。

```bash
export FLUX2_QWEN3_MODEL_SPEC="$IMAGEWAM_ROOT/checkpoints/3090/Qwen3-4B"
"$IMAGEWAM_PYTHON" scripts/setup/download_3090_qwen.py \
  --proxy http://127.0.0.1:7897 --workers 3 --transport modelscope
```

Hugging Face 文件传输持续遇到 TLS 中断和读取超时，已改用官方 ModelScope 的 `Qwen/Qwen3-4B` 作为文件传输源；HF 固定 revision 仍作为内容校验依据。上面的 `--transport modelscope` 使 ModelScope 文件连接使用直连，`--proxy` 仅用于 HF 元数据查询。

本次 **10 个文件、8,060,896,856 bytes 已全部下载并通过校验**，包括 3 个权重分片。报告为 `artifacts/3090_setup/qwen_download_20261027T071414Z_1300641.json`，`status=complete`；下载日志为同目录 `qwen_modelscope_download_20261027T071414Z.log`。

脚本逐个核验文件大小、LFS 权重 SHA-256 或普通文件 Git blob SHA-1，检查分片索引引用完整性，并保存 `artifacts/3090_setup/qwen_download_<UTC时间>_<PID>.json`。即使 ModelScope 的 `master` 更新，内容不符合固定 HF revision 的文件也不能通过校验。增加 `--verify-only` 可跳过文件下载，但仍需联网取得固定 revision 的文件元数据进行比对。

两个下载脚本均禁止隐式 token、关闭 Xet、将 Hugging Face 缓存放在仓库 `.cache/huggingface`，失败时保留断点文件；日志仅输出异常类型/HTTP 状态，避免记录签名下载 URL。当前公开资产路径无需以取得原受限 AE 仓库权限为前提。

如文件大小正确但哈希校验失败，Qwen 下载阶段会按大小跳过已有文件；需要先将报告中不匹配的文件移至单独备份路径，再重新下载。不要把只有大小正确、报告仍为 `failed` 的目录作为已验证资产使用。本次 10 个文件均已通过哈希校验。

## 4. LIBERO 资产和仓库内配置

服务器原有 `/home/ubuntu/data_1/chenxing/LIBERO` 包含：

- 130 个 BDDL 任务；全部具有对应的 `.pruned_init`。
- 130 个 `.pruned_init`、120 个普通 `.init`。10 个 `libero_goal` 任务没有普通 `.init`，其 `.pruned_init` 齐全。
- 585 个仿真资产文件，合计 422,313,213 bytes，约 402.75 MiB。
- `libero_spatial`、`libero_object`、`libero_goal`、`libero_90`、`libero_10` 五个任务目录。

该目录缺少 `.git` 元数据，无法从本地确定其上游 commit 或工作区差异；包元数据版本为 `0.1.0`。已在 `artifacts/3090_setup/libero_files.json` 保存实际使用副本的 1,121 个文件及 SHA-256，包含本次加载兼容性补丁后的文件。此次验证是现有任务的仿真 rollout，不要求下载训练演示数据集。

已将现有 LIBERO 目录通过 `cp -a` 复制到仓库 `third_party/LIBERO`，副本约 427 MB；共享源目录保持不变。复现复制过程如下，已有副本时复用：

```bash
export IMAGEWAM_LIBERO="$IMAGEWAM_ROOT/third_party/LIBERO"
if [ ! -e "$IMAGEWAM_LIBERO" ]; then
  mkdir -p "$IMAGEWAM_ROOT/third_party"
  cp -a /home/ubuntu/data_1/chenxing/LIBERO "$IMAGEWAM_LIBERO"
fi
```

LIBERO 通过 `PYTHONPATH` 直接使用该副本，无需 editable 安装。已通过 `scripts/flux2/configure_3090_libero.py` 写入仓库 `.libero/config.yaml`，通过 `LIBERO_CONFIG_PATH="$IMAGEWAM_ROOT/.libero"` 使用它；`benchmark_root`、`bddl_files`、`init_states`、`assets` 均指向该副本，`datasets` 指向仓库 `data/libero`。配置脚本在既有配置与目标不一致时拒绝覆盖，不修改共享 `~/.libero/config.yaml`。

本地 LIBERO 副本已进行一处兼容性修改：`third_party/LIBERO/libero/libero/benchmark/__init__.py` 中的初始状态加载改为 `torch.load(init_states_path, weights_only=False)`，用于读取已确认来源的 NumPy 仿真初始状态。该修改只针对仓库内副本的一条加载语句，不改变 PyTorch 全局加载行为，也不修改共享 LIBERO 源目录。

```bash
export LIBERO_CONFIG_PATH="$IMAGEWAM_ROOT/.libero"
"$IMAGEWAM_PYTHON" scripts/flux2/configure_3090_libero.py \
  --libero-dir "$IMAGEWAM_LIBERO" \
  --config-dir "$LIBERO_CONFIG_PATH" \
  --patch-init-state-load
```

推理 wrapper 会执行同样的配置检查；一致的配置和已应用的补丁可重复使用。

无显示器环境采用 EGL：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

EGL smoke test 已以退出码 0 完成环境 reset、10 步执行和双相机渲染，两个图像输出均为 `256×256×3 uint8`；CUDA BF16 矩阵乘法检查也通过。该检查总耗时约 10.45 秒，这是环境验证耗时，不是策略推理延迟。证据在 `artifacts/3090_setup/simulator_smoke.log` 和 `simulator_smoke.json`。此次适配没有执行 `apt`，没有替换驱动或系统 OpenGL/CUDA 库。

## 5. 原生推理代码路径与参数

调用顺序：

```text
experiments/libero/eval_libero_single.py
  → Hydra: configs/sim_libero_omnigen2.yaml
  → task=libero_flux2_klein_4b_base_imagewam
  → imagewam.runtime.create_imagewam_flux2_klein
  → ImageWAM.from_flux2_klein_pretrained
  → 原生 FLUX.2 视觉参数、AE、Qwen3 文本编码器
  → model.load_checkpoint(model.pt)
  → LIBERO 观测预处理、统计归一化
  → ImageWAM.infer_action_flux2
  → 动作反归一化与环境执行
```

`sim_libero_omnigen2` 是现有 FLUX.2 评测脚本复用的配置容器名称；实际模型由 `task` 与 `model._target_` 决定。命令必须覆盖为 FLUX.2 4B task，不能仅凭配置文件名判断模型类型。

| 参数 | wrapper 默认值 | 含义 |
|---|---|---|
| GPU / worker / 环境 | `1 / 1 / 1` | 单进程、单环境，先验证最小链路 |
| suite / task / trial | `libero_spatial / 0 / 1` | 首次策略 rollout 配置 |
| `mixed_precision` | `bf16` | 模型 dtype，与当前 checkpoint 对齐 |
| 输入 | 高 224 × 宽 448 | 两个 224×224 相机图像横向拼接 |
| 动作维度 / proprio | `7 / 8` | 由 LIBERO processor 配置和统计决定 |
| `action_horizon` | `16` | 每次预测 16 个动作 |
| `replan_steps` | `12` | 执行 12 步后重新观测/规划 |
| `num_inference_steps` | `10` | 动作去噪迭代数 |
| Qwen context | `512` | FLUX.2 task 的文本上下文长度 |
| `seed` | `42` | wrapper 的默认随机种子 |
| `pack_proprio_after_text` | `true` | 与现有 FLUX.2 入口保持一致 |
| `text_cfg_scale` | `1.0` | 不额外放大文本引导 |
| `visualize_future_video` | `false` | 首次验证不额外生成预测视频 |

### 单任务命令：资产与 EGL 检查通过后执行

本次使用已提供的 `scripts/flux2/infer_3090_libero.sh`：检查必要权重文件/源码目录，准备仓库内 LIBERO 配置和初始状态加载补丁，然后调用原生单任务入口。它不使用原多任务 shell 的默认 8 GPU、多 worker 配置。

已执行 `bash scripts/flux2/infer_3090_libero.sh --cfg job`，退出码 0，解析配置保存在 `artifacts/3090_setup/native_config_check.log`。该命令只检查 Hydra 配置，不运行策略前向。

```bash
CUDA_VISIBLE_DEVICES=0 \
TASK_SUITE=libero_spatial TASK_ID=0 NUM_TRIALS=1 \
ACTION_HORIZON=16 REPLAN_STEPS=12 INFERENCE_STEPS=10 \
QWEN_CONTEXT_LEN=512 SEED=42 \
  bash scripts/flux2/infer_3090_libero.sh
```

这些值与 wrapper 默认值一致，可按需覆盖；额外 Hydra 参数可直接追加在脚本后面。checkpoint 和输出路径需通过 `CKPT_PATH`、`OUTPUT_DIR` 环境变量设置；wrapper 拒绝追加 `ckpt=`、`EVALUATION.output_dir=`、`hydra.run.dir=`，确保预检查、路径标签、控制台日志和实际输出一致。wrapper 显式设置 `data.train.dataset_dirs=[]`、`data.train.qwen_text_cache_dir=null`、`model.action_dit_pretrained_path=null` 和 `wandb.enabled=false`，使用本地文本编码器和完整策略，不要求训练数据、动作初始化 checkpoint 或训练服务。

wrapper 将 `PYTHONPATH` 指向本仓库和两个源码副本，设置 `PYTHONDONTWRITEBYTECODE=1`、`PYTHONNOUSERSITE=1`，并将 `HF_HOME`、`TORCH_HOME`、`XDG_CACHE_HOME`、`MPLCONFIGDIR`、`NUMBA_CACHE_DIR` 放在仓库 `.cache` 下，避免将运行缓存或字节码写入共享基础环境。渲染默认 `MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`。

默认输出在 `evaluate_results/3090/model/<时间_PID>/`，`model` 是当前 `model.pt` 的 checkpoint 标识；自定义 `OUTPUT_DIR` 时也会遵循原生入口的标识插入规则。日志中的实际路径是最终依据，目录中保留 `run.log`、Hydra 日志、解析后的 `eval_config_gpu0_task0.yaml`、`gpu0_task0_results.json` 和 rollout 视频。单任务单 trial 仅验证执行链路，不能替代完整 benchmark，也不能据此报告稳定成功率或与论文数字比较。

## 6. 文件修改与新增资产清单

本表记录此次新增文件与运行资产；实测结果单独列于下一节。

| 文件或目录 | 状态 | 目的 |
|---|---|---|
| `README_3090_inference.md` | 新增文档 | 记录适配、参数、复现命令与验证结果 |
| `scripts/flux2/export_3090_flux2_weights.py` | 已执行通过 | 从完整 checkpoint 导出原生视觉权重并严格校验 |
| `scripts/flux2/convert_public_flux2_ae.py` | 已执行通过 | 将公开 BF16 AE 逆转换为原生格式，严格检查键名/形状 |
| `scripts/setup/setup_3090_inference.sh` | 镜像安装成功，退出码 0 | 只在仓库 `.venv` 安装编译后排除 GPU 包的 lock |
| `scripts/setup/guard_3090_environment.py` | 已上传服务器 | 检查 GPU 包继承关系、版本/路径和项目配置摘要 |
| `requirements/3090-inference.in`、`requirements/3090-inference.lock` | 已使用，lock 为 100 项 | 固定直接及传递依赖，与原项目依赖文件分开 |
| `scripts/setup/download_3090_ae.py` | 已再次执行 `--verify-only` 通过 | 下载公开 AE、校验权重与配置文件并保存报告 |
| `scripts/setup/download_3090_qwen.py` | 下载与校验通过 | 固定 Qwen revision、断点重试和逐文件校验 |
| `scripts/flux2/configure_3090_libero.py` | 已执行 | 生成仓库配置并仅修改本地 LIBERO 初始状态加载 |
| `scripts/flux2/infer_3090_libero.sh` | 已提供 | 固化单 GPU 参数、本地路径、配置、缓存与输出目录 |
| `scripts/flux2/profile_3090_inference.sh` | 性能启动入口 | 复用 `.venv`、本地模型、EGL 与缓存配置，设置采样次数并保存控制台日志 |
| `scripts/flux2/benchmark_3090_inference.py` | 性能测量主程序 | 一次加载、真实观测、收敛预热、无插桩稳态采样、模块采样和 JSON/CSV 报告 |
| `scripts/flux2/profile_3090_timing.py` | 加载与计时工具 | 同步加载 wall time；运行时 CUDA events；统计均值、中位数、p95、极值、标准差 |
| `scripts/flux2/profile_3090_modules.py` | 临时模块插桩 | 包装原生调用，结束或异常时恢复；不复制模型前向、不改变权重或输入 |
| `evaluate_results/3090_profile/<RUN_ID>/` | 性能脚本自动生成 | 保存逐次样本、模块及加载明细、最终配置、参考动作和日志 |
| `.venv/` | 安装和导入检查通过 | 继承现有 Torch，容纳推理依赖 |
| `third_party/flux2/` | 已独立 clone、固定 commit | 官方 FLUX.2 源码 |
| `third_party/LIBERO/` | 已复制 | 复用资产，同时隔离共享目录 |
| `third_party/LIBERO/libero/libero/benchmark/__init__.py` | 本地副本单行补丁已应用 | 为已知 NumPy 初始状态明确指定 `weights_only=False` |
| `.libero/config.yaml` | 已生成 | 通过 `LIBERO_CONFIG_PATH` 隔离配置 |
| `checkpoints/3090/flux2_imagewam_visual.safetensors{,.json}` | 149 张量，7,751,105,848 bytes，严格加载通过 | 初始化视觉参数与导出元数据 |
| `checkpoints/3090/public_vae/` | 已下载并校验 | 保留官方公开 AE 原始文件 |
| `checkpoints/3090/ae.safetensors` | 251 张量，168,118,958 bytes，严格加载通过 | 原生 FLUX.2 AE 参数 |
| `checkpoints/3090/Qwen3-4B/` | 10 文件、8,060,896,856 bytes 校验通过 | 原生文本编码器 |
| `artifacts/3090_setup/run_*/` | setup 自动生成 | 保存安装、lock、环境保护与导入检查证据 |
| `artifacts/3090_setup/environment_preservation_verified.json` | `status=passed` | 对最初 19 个 GPU 相关包逐项比较版本、元数据路径和 RECORD SHA-256 |
| `pyproject.toml`、`uv.lock` | 保留原文件 | 避免改变原项目环境约定 |
| `experiments/libero/eval_libero_single.py` | `run_single_task` 增加 `try/finally: env.close()` | 确保成功、模型异常或视频写入异常时释放 EGL 仿真环境 |
| 原生 factory、模型实现 | 未修改 | 保持完整 checkpoint 的原生加载和推理路径 |

原仓库 tracked 文件仅修改 `experiments/libero/eval_libero_single.py` 的资源释放逻辑，原任务执行代码被包入 `try`，未改变预测或结果计算。其他适配通过新增脚本、profile/lock 和本 README 实现；`third_party` 默认被 Git 忽略，其中的 LIBERO 副本补丁已在表中单独记录。

## 7. 基础环境保护与验证记录

setup 已保存基础解释器和 `.venv` 的快照与比较结果，并比较 `pyproject.toml`、`uv.lock` 的摘要。镜像安装最终退出码为 0，基础环境和 `.venv` 的 guard 比较均通过，证据位于 `artifacts/3090_setup/run_20261027T070831Z_1294703/`。比较同时检查版本、包目录和 `torch.__file__`，不只检查版本字符串。

此外已对最初 `environment_before.json` 中的 **19 个受保护 GPU 包**逐项比较 `version`、`metadata_path` 和 `RECORD` 文件 SHA-256，全部一致；Torch 模块路径、版本与 CUDA 构建版本也一致。结果保存在 `artifacts/3090_setup/environment_preservation_verified.json`，状态为 `passed`。这里的 RECORD 校验是分发元数据文件的比较，不等同于重新哈希每个包内所有文件。

以下为可复用的检查命令；请把前后结果分别写入不同日志，不覆盖原始基线：

```bash
"$IMAGEWAM_BASE_PYTHON" - <<'PY'
import importlib.metadata as md
import sys
import torch, torchvision, torchaudio, triton
print("python", sys.version)
print("executable", sys.executable)
for name, module in [("torch", torch), ("torchvision", torchvision),
                     ("torchaudio", torchaudio), ("triton", triton)]:
    print(name, md.version(name), module.__file__)
print("torch.version.cuda", torch.version.cuda)
PY
"$IMAGEWAM_PYTHON" - <<'PY'
import importlib.metadata as md
import sys
import torch, torchvision, torchaudio, triton
print("python", sys.version)
print("executable", sys.executable)
for name, module in [("torch", torch), ("torchvision", torchvision),
                     ("torchaudio", torchaudio), ("triton", triton)]:
    print(name, md.version(name), module.__file__)
print("torch.version.cuda", torch.version.cuda)
PY
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nvcc --version
sha256sum pyproject.toml uv.lock
git status --short
git diff --stat
```

依赖完整清单可通过 `"$IMAGEWAM_UV_BIN" pip freeze --python "$IMAGEWAM_PYTHON"` 保存；检查结果应区分此推理覆盖环境与原项目全量依赖约束。

| 验证项目 | 当前记录 | 证据/后续检查 |
|---|---|---|
| 基础 PyTorch 栈与来源 | base / overlay guard 均通过；最初 19 个受保护包的版本、元数据路径和 RECORD SHA-256 一致 | `artifacts/3090_setup/environment_preservation_verified.json` 及镜像安装快照 |
| 原始项目配置保持不变 | guard 摘要比较通过 | 相同目录的 `project_manifests` 字段 |
| 仓库推理依赖安装 | 100 项安装成功，核心与数据导入通过，退出码 0 | `run_20261027T070831Z_1294703/installation.log`、`requirements.lock` |
| 官方 FLUX.2 commit | 已切换至固定版本 | `50fe5162777813d869182b139e83b10743caef15` |
| 视觉权重严格导出 | 149 张量、7,751,105,848 bytes，严格加载通过 | `flux2_imagewam_visual.safetensors.json` |
| 公开 AE 下载 | 下载与再次 `--verify-only` 均通过 | `artifacts/3090_setup/ae_download.json` |
| AE 原生转换 | 251 张量、168,118,958 bytes，严格加载通过 | `checkpoints/3090/ae.safetensors` |
| Qwen3-4B 下载/读取 | 10 文件哈希校验通过，3 个分片已由原生 loader 读取 | `qwen_download_20261027T071414Z_1300641.json`、原生推理日志 |
| LIBERO 副本与配置 | 已复制约 427 MB，配置和单行补丁已完成 | `third_party/LIBERO`、`.libero/config.yaml` |
| 原生 Hydra 配置检查 | `--cfg job` 退出码 0 | `artifacts/3090_setup/native_config_check.log` |
| EGL / CUDA BF16 smoke test | 退出码 0；reset + 10 步、双相机 `256×256×3 uint8`、BF16 矩阵乘法通过；总耗时 10.45 秒 | `artifacts/3090_setup/simulator_smoke.log`、`simulator_smoke.json` |
| 完整策略推理 | 首轮与修复后复测均为 task 0 的 1 个回合成功，退出码 0；复测无 EGL 析构异常 | `evaluate_results/3090/model/verified_3090_20261027/` |

## 8. 实际推理结果（2026-10-27）

首轮原生推理目录为 `evaluate_results/3090/model/native_20261027T071906Z/`。完整 MoT 加载日志为 `missing_keys=0 unexpected_keys=0`，proprio 参数严格加载成功，随后使用原生 LIBERO 观测和模型动作完成回合。

| 项目 | 首轮实测 |
|---|---|
| 任务 | `libero_spatial / task 0` |
| 描述 | pick up the black bowl between the plate and the ramekin and place it on the plate |
| 结果 | 1 / 1 成功，退出码 0 |
| 包含模型加载的总墙钟耗时 | 100.866 秒 |
| 原生结果 JSON 的 duration | 92.723 秒，也包含模型加载，不是单次前向延迟 |
| 整卡显存采样峰值 | 19,278 MiB，约 18.83 GiB |
| 运行前整卡显存 | 369 MiB |
| 采样方式 | 每 1 秒读取 nvidia-smi，包含桌面与其他进程；不是 PyTorch allocator 的精确峰值 |
| rollout 视频 | 76 帧，24 fps，256×512，135,306 bytes；已实际解码检查 |

目录中 `run.log` 保存控制台日志；`run_metrics.json` 和 `gpu_samples.csv` 保存外层计时与显存记录；`libero_spatial/gpu0_task0_results.json` 保存任务结果；`libero_spatial/videos/` 保存 MP4；`libero_spatial/eval_config_gpu0_task0.yaml` 保存实际运行配置。模型回合中已真实执行动作；此次没有额外插桩记录逐次 action tensor，因此不把成功 rollout 描述为独立的逐元素数值检查。

首轮退出后出现 `MjRenderContext.__del__` / `EGLGLContext.__del__` 的 `EGL_NOT_INITIALIZED` 警告，发生于成功输出之后，进程退出码仍为 0。原因是原生 `run_single_task()` 未显式关闭环境。现已增加 `try/finally: env.close()`；针对正常完成、reset 异常、模型异常、视频写入异常的隔离检查均确认环境关闭恰好一次，保留原结果或原异常。

相同默认参数的真实回合复测位于 `evaluate_results/3090/model/verified_3090_20261027/`，结果仍为 **1/1 成功，退出码 0**，不再出现 EGL 析构异常。模型加载、动作生成、仿真执行、结果 JSON 和 MP4 保存均完成。最终证据汇总为 `artifacts/3090_setup/final_inference_verification.json`，最终继承环境检查为 `artifacts/3090_setup/venv_final.json`。

复测总墙钟时间 **107.00 秒**，原生结果 JSON 的 duration 为 **97.351 秒**，两者均包含模型加载。修复前后视频都为 76 帧，逐帧解码后 SHA-256 相同（`fd22295ceaeae7b5eed0ff48e9e06c70bfb333d133351461bc03ef09a3e588cf`），确认清理修复未改变本次 rollout。退出后 GPU 回到 369 MiB、利用率 0%。两次运行使用相同任务、初始状态和随机种子，不是两个独立 benchmark 样本。

日志中的 robosuite 私有 macro 提示、旧 Gym 的维护提示及 Transformers `torch_dtype` 弃用提示均不阻止本次运行；未为消除提示更换仓库指定的仿真版本。当前单任务推理没有缺少的必需依赖，也未安装或编译 FlashAttention、DeepSpeed、TorchCodec 等训练/其他模型路径组件。

遇到 OOM 时记录发生在加载、文本编码、prefill 还是动作去噪阶段，以及当时 GPU 上的其他进程和可用显存。先保证单 worker 与既定输入大小；若后续采用文本缓存、CPU offload 或参数调整，需另行记录代码/配置变化及精度或行为影响。24 GB 的容量不能单独证明当前全部组件可以同时常驻显存。

## 9. 一次加载、多次推理性能统计

### 9.1 启动与参数

在服务器仓库目录执行，脚本使用现有 `.venv`，无需新增依赖：

```bash
cd /home/ubuntu/data_1/chenxing/ImageWAM/ImageWAM
bash scripts/flux2/profile_3090_inference.sh
```

指定重复次数及参数：

```bash
BENCH_ITERS=50 PROFILE_ITERS=20 \
WARMUP_MIN=6 WARMUP_MAX=30 \
STABILITY_WINDOW=3 STABILITY_TOLERANCE=0.10 \
INFERENCE_STEPS=10 ACTION_HORIZON=16 REPLAN_STEPS=12 \
TASK_SUITE=libero_spatial TASK_ID=0 QWEN_CONTEXT_LEN=512 SEED=42 \
RUN_ID="profile_$(date +%Y%m%d_%H%M%S)_$$" \
  bash scripts/flux2/profile_3090_inference.sh
```

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `BENCH_ITERS` | 20 | 无模块钩子的端到端测量次数 |
| `PROFILE_ITERS` | 10 | 同一模型上开启模块计时的测量次数 |
| `WARMUP_MIN` / `WARMUP_MAX` | 6 / 20 | 至少预热次数 / 最大预热次数 |
| `STABILITY_WINDOW` | 3 | 比较最近两个相邻窗口，每个窗口的样本数 |
| `STABILITY_TOLERANCE` | 0.10 | 窗口中位数变化和最新窗口变异系数的上限 |
| `RUN_ID` | 时间戳 + PID | 输出目录名称 |
| `OUTPUT_DIR` | `evaluate_results/3090_profile/<RUN_ID>` | 自定义时直接使用该目录，不插入 checkpoint 标签 |

模型路径环境变量与第 3.0 节一致；GPU、任务、seed、去噪步数、动作长度、文本长度沿用普通推理默认值。`REPLAN_STEPS` 记录在配置中，性能采样不执行环境动作，所以它不会让每次只生成 12 个动作。`NUM_TRIALS` 是普通回合测试参数，性能脚本用 `BENCH_ITERS` 控制重复次数。

性能入口使用 argparse，可追加 `--iterations 50`、`--profile-iterations 20` 等参数，追加值优先于环境变量；它不接受普通评测入口的 Hydra `key=value` 参数。输出路径统一通过 `OUTPUT_DIR` 指定，wrapper 拒绝追加 `--output-dir`。查看完整参数：

```bash
.venv/bin/python scripts/flux2/benchmark_3090_inference.py --help
```

同名输出目录会被拒绝，避免覆盖已有基线。正常完成且预热、稳态检查都通过时退出码为 0；完成但未满足稳定判据时退出码为 2，JSON 状态为 `complete_warmup_not_stable` 或 `complete_steady_not_stable`；进入模型加载阶段后的异常会保存失败状态并非零退出，参数、导入或资产预检查失败请查看 `run.log`。`warmup-min >= 2 * stability-window`，`warmup-max >= warmup-min`；要通过稳态检查，`BENCH_ITERS` 也需至少覆盖两个窗口。

### 9.2 流程和端到端边界

一个进程按以下顺序执行，工厂构造和完整 checkpoint 恢复各一次；测量循环内不重新加载模型：

1. 导入依赖、解析 Hydra 配置、显式初始化 CUDA；这些启动开销单独记录。
2. 使用原生 factory 构造视觉模型、ActionDiT、AE、Qwen/tokenizer、MoT 和 proprio；原生 `_load_model_checkpoint()` 恢复训练后的策略，并切换 eval 模式。
3. 创建 LIBERO 环境，从所选任务的第 0 个初始状态开始，执行配置中的 30 步等待动作，取真实双相机图像和本体状态；这一准备过程单独计时。
4. 在同一输入上预热，满足最小次数后检查稳定性，最多执行指定上限。
5. 使用同一已加载模型重复端到端推理，默认 20 次，关闭模块钩子。
6. 继续在同一模型上重复推理，默认 10 次，临时包装原生模块收集 CUDA events；结束后恢复原方法并关闭环境。

**一次端到端推理的输入是原始观测和任务文本，输出是 CPU 上反归一化的 `16×7` 动作块。** 包含双相机图像预处理、在线 Qwen 文本编码、本体编码、AE 图像编码、视觉 KV prefill、10 步动作去噪及动作后处理。两路原始图像均为 `256×256×3`，由原生 processor 处理成配置的 `224×448` 双视图输入。计时起止处均同步 CUDA，采用 `time.perf_counter()`。

仿真 physics/step/render、模型加载、预热和日志/报告写入均不计入稳态推理耗时。每次处理相同真实观测和文本，保持原生固定 seed 行为，不复用文本 embedding、图像 embedding 或跨推理的视觉 KV cache；每次仍重走编码和 prefill。这衡量固定输入下单个 action chunk 的延迟，不能把延迟除以 16 后当作闭环单步响应延迟，也不能替代完整 rollout 的耗时或成功率。

脚本会检查每次输出形状、有限数值，并与首次输出比较，容差 `rtol=atol=1e-5`；这些检查位于端到端计时区间之外。JSON 记录最大动作差异，保存第一份动作到 `reference_action.npy`，同时记录输入图像形状和 SHA-256。

### 9.3 预热、加载和模块统计的含义

预热稳定判据为：最近两个长度为 `STABILITY_WINDOW` 的窗口，中位数相对变化不超过阈值，且最新窗口的标准差/均值不超过阈值。满足后结束预热；达到最大次数仍不满足时保留全部数据并明确标记。稳态测量结束后用相同判据检查最后两个窗口，同时提供全部样本的标准差、极值和 p95。该判据是短窗口检查，不代表整个过程永远无抖动。

`warmup.total_wall_ms` 是整个预热阶段耗时，包含阶段内结果检查与进度打印；`warmup.latency_ms` 是逐次推理耗时的统计。首轮推理单独保存为 `first_call_ms`。稳态及模块统计不会把预热样本混入均值。

加载使用同步 wall time，覆盖文件读取、参数赋值、设备/精度转换；Python 导入和 CUDA 上下文初始化另列。`model_load_total` 是整体模型加载时间，`factory.total` 包含视觉、动作初始化、AE、Qwen 等子项，随后才是 `policy.load_checkpoint_total`。加载保留系统页缓存状态，没有清除缓存，因此不将结果称为冷盘加载性能。

| 加载标签 | 具体含义 |
|---|---|
| `visual.total` | FLUX 视觉分支构建，含初始化、视觉 safetensors 读取、参数赋值和设备转换 |
| `action.total` | ActionDiT 构造与转移；训练参数稍后从完整策略恢复 |
| `ae.initialize_meta` / `read_weights` / `apply_weights` / `to_device_dtype` | AE 空参数构造、权重读取、赋值和转换，独立记录 |
| `qwen.load_pretrained` / `to_device_dtype` / `load_tokenizer` | Qwen 三个分片的加载、设备转移和 tokenizer 加载 |
| `mot.assemble` / `imagewam.assemble` | 联合模型与策略对象组装 |
| `policy.read_checkpoint` | 读取完整 `model.pt` |
| `policy.apply_mot_weights` / `apply_proprio_weights` | 将训练后的联合策略与本体编码参数恢复到实例 |

同一模型在构造、组装和最终 `.to()` 时可能出现多条设备转换记录；每条保留实际调用的时间及父级关系。完整策略中的视觉和动作参数由同一个 `load_state_dict` 恢复，当前记录为联合的 `policy.apply_mot_weights`，不虚构两个独立文件读取时间。

模块阶段使用 CUDA event，在当前 stream 记录每个边界，单次推理结束后统一同步和读取，避免在每个模块之间强行同步。`cuda_ms` 是该 stream 上的经过时间，可包含 CPU 提交间隙，**不等同于纯 GPU kernel 活跃时间**。`wall_ms` 则是未在模块边界同步的主机调用时间，某些同步等待可能落在后一个调用中；不要将它当作独立 GPU 模块耗时。纯 CPU 的模板、tokenizer 和动作反归一化只报告 wall time。

| 运行标签 | 对应原生边界 |
|---|---|
| `policy.action_chunk` | 评测入口 `_predict_action_chunk`，含预处理和后处理 |
| `observation.preprocess` | 观测整理及 processor 图像预处理 |
| `policy.model_forward` | `ImageWAM.infer_action_flux2()` |
| `text.encode` | 文本模板、tokenize、Qwen 前向和上下文整理 |
| `text.chat_template` / `text.tokenize` / `text.qwen_forward` | 文本编码子步骤；Qwen 前向沿用原生实现，含其原有 LM head 计算 |
| `proprio.encode_and_pack` / `proprio.linear` | 本体状态整理、编码并打包到文本上下文 |
| `image.encode_tokens` / `image.ae_encode` | AE 编码及图像 token 整理 |
| `visual.prepare` / `visual.prefill` | 视觉位置/条件准备及视觉分支 KV prefill |
| `action.denoise_loop` | 全部动作去噪迭代 |
| `action.denoise_step` | 单步动作去噪，默认每次推理调用 10 次 |
| `action.prepare` / `action.transformer` / `action.project` / `action.scheduler` | 每步动作准备、MoT action forward、输出投影和 scheduler 更新 |
| `action.denormalize` | CPU 动作反归一化 |

以上是 **inclusive** 范围，父行包含子行，不能把整张表相加。`module_summary.csv` 同时提供“每次推理内该模块所有调用之和的均值”和“单次调用均值”：例如默认 `action.transformer` 的前者包含 10 次调用，后者只是一轮动作 Transformer。AE 解码、未来视频生成属于未启用路径，不列为实际执行模块。

无模块钩子的 `steady` 均值是主要端到端指标；带钩子的 `profile` 均值及其相对差异另行报告。这个差异同时受到插桩开销、设备时钟和运行波动影响，不能当作精确隔离的钩子成本。各阶段默认报告样本数、均值、中位数、线性插值 p95、最小值、最大值和总体标准差。

### 9.4 输出与复查

```text
evaluate_results/3090_profile/<RUN_ID>/
├── run.log                  # 完整控制台输出
├── summary.md               # 加载、预热、稳态、模块的可读表格
├── benchmark.json           # 最终状态、配置、路径、硬件、全部统计及原始 span
├── iterations.csv           # 每次 warmup / steady / profile 的同步 wall ms
├── spans.csv                # 每次模块/加载调用、parent_id、wall/cuda ms
├── module_summary.csv       # 每模块每次推理和每次调用的均值
├── load_spans.json          # 模型加载完成后立即保存的加载明细
├── resolved_config.yaml     # 实际 Hydra 配置
├── reference_action.npy     # 首次推理动作，用于检查重复结果
└── progress.json            # 启动阶段快照；最终结果以 benchmark.json 为准
```

性能脚本仅新增外层测量代码，复用原生工厂、checkpoint loader、processor 和 `_predict_action_chunk()`；临时包装在退出时恢复。没有安装新依赖，没有修改 CUDA/PyTorch、原生模型结构、精度、权重或去噪算法，也没有对原仓库 README 写入内容。

### 9.5 RTX 3090 实测记录（2026-10-27）

最终脚本实测目录：[`evaluate_results/3090_profile/profile_verified_20261027/`](evaluate_results/3090_profile/profile_verified_20261027/)，启动命令：

```bash
RUN_ID=profile_verified_20261027 bash scripts/flux2/profile_3090_inference.sh
```

该目录已存在，复现时使用新的 `RUN_ID`。本次 `loading_count=1`，退出码 0，状态 `complete`；task 0、seed 42、BF16、上下文 512、动作块 16×7、10 步去噪，未启用文本缓存。

| 总体指标 | 实测 |
|---|---:|
| 模型加载总时间 | 87.501 秒 |
| 预热次数 / 阶段总时间 | 6 次 / 3.476 秒 |
| 第一次推理 | 1,488.673 ms |
| 预热收敛检查 | 通过 |
| 稳态端到端样本数 | 20 |
| **稳态端到端平均** | **397.133 ms / 动作块** |
| 中位数 / p95 | 397.185 / 399.066 ms |
| 最小值 / 最大值 | 394.983 / 401.019 ms |
| 总体标准差 | 1.397 ms |
| 稳态最后两个窗口检查 | 通过 |
| 带模块钩子的 10 次平均 | 398.557 ms |
| 带钩子均值相对稳态的差异 | +0.36% |
| 所有重复推理与首次动作的最大差异 | 0 |

加载明细按同步 wall time 摘录如下；包含关系见第 9.3 节，不能将总项和子项相加。

| 组件或阶段 | wall ms |
|---|---:|
| 原生 factory 总计 | 46,661.490 |
| FLUX 视觉分支构建总计 | 20,441.951 |
| ActionDiT 初始化及设备转移 | 3,187.350 |
| AE 初始化 / 读权重 / 参数赋值 | 15.972 / 623.340 / 2.854 |
| AE 三次 `.to()` 调用合计 | 2.572 |
| Qwen `from_pretrained` | 1,103.019 |
| Qwen 第一次设备转移 | 20,967.193 |
| Qwen tokenizer 加载 | 305.405 |
| 完整策略 checkpoint 恢复总计 | 40,258.273 |
| 其中：`model.pt` 读取 | 37,429.062 |
| 其中：MoT 训练权重赋值 | 2,810.705 |
| 其中：proprio 训练权重赋值 | 0.100 |

AE/Qwen 的三次 `.to()` 来自原生初始化、组装和最终设备转换检查，并未重复读取权重。Qwen 三分片读取和设备转移的分布受到加载器及磁盘页缓存影响；不能仅凭 `from_pretrained` 的时间判断整个文本模型已经就绪。首轮独立进程实测加载总时间为 78.596 秒，与本次的差异保留在原始报告中，未跨运行合并加载统计。

主要模块采用 10 次插桩推理的 CUDA event 平均，单位为 ms。父子模块有重叠，以下摘要不是可直接相加的时间分解：

| 模块 | 每次推理调用数 | 每次推理平均 ms | 单次调用平均 ms |
|---|---:|---:|---:|
| 观测预处理 | 1 | 2.388 | 2.388 |
| 在线文本编码整体 | 1 | 90.411 | 90.411 |
| 其中 Qwen 前向 | 1 | 89.639 | 89.639 |
| 本体状态编码与打包 | 1 | 0.330 | 0.330 |
| 图像 AE 编码 | 1 | 14.118 | 14.118 |
| 视觉条件准备 | 1 | 1.893 | 1.893 |
| 视觉 KV prefill | 1 | 143.653 | 143.653 |
| 10 步动作去噪整体 | 1 | 141.673 | 141.673 |
| 其中 Action Transformer | 10 | 136.055 | 13.605 |
| 其中 scheduler 更新 | 10 | 0.268 | 0.027 |

文本模板、tokenizer、动作反归一化的 CPU wall 平均分别为 0.100、0.319、0.084 ms。完整 20 个模块及全部加载子阶段见本次 [`summary.md`](evaluate_results/3090_profile/profile_verified_20261027/summary.md)，原始记录见 [`benchmark.json`](evaluate_results/3090_profile/profile_verified_20261027/benchmark.json)。本次共记录 650 个模块 span，5 个去噪子模块各 100 次调用，其余 15 个模块各 10 次；全部与 10 次推理 × 10 步去噪配置匹配。

首轮独立验证目录 `profile_initial_20261027` 的稳态 20 次平均为 398.816 ms，最终脚本的 20 次平均为 397.133 ms。两次各自只加载一次，最终基线采用后者，未把不同进程的样本混成一个统计结果。

性能测试后再次保存 `artifacts/3090_setup/venv_after_profile.json`，与此前 `venv_final.json` 比较通过：PyTorch 仍为 `2.8.0+cu128`，来自 `/home/ubuntu/miniconda3/envs/py312/lib/python3.12/site-packages/torch/__init__.py`；受保护 GPU 包、Python 版本、`pyproject.toml`、`uv.lock` 均保持一致。原仓库 `README.md` SHA-256 仍为 `301ab40295495f385fdb4cd1a73ca24544fe04893ddb63f1c3e46ef8f213221d`。


## 2026-10-28：独立动作加载器

新增 [优化加载适配与实测记录](README_3090_optimized_loading.md)：保留共享 embedding 与 FLUX 别名，避免视觉重复加载，严格比较中间结果及最终动作；包括正常入口、性能对比和完整动作文件。原执行模式数据继续保留。
