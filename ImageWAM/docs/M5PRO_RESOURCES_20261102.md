# M5 Pro RoboTwin 开发资源

工作目录：`/Users/qklj/work/cx/ImageWAM`。

## 代码来源

代码由 3090 `/home/ubuntu/data_1/chenxing/ImageWAM/ImageWAM` 直接上传，保留其 Git 历史、工作区修改、`src/imagewam/optimized_action_loader.py`、3090 推理/性能脚本和本地说明，以及 RoboTwin/FLUX2 依赖源码。1,580 个普通文件已经逐文件 SHA-256 验证，详见 `.setup/robotwin_20261102/code_verified.json`。

此次代码包没有携带 Linux `.venv`、缓存、模型、历史评测输出、LIBERO 仿真资产和 FLUX2 示例图片。旧完整归档传输已暂停，断点保留在工作目录外；不会自动解压覆盖本开发目录。

## 下载目录

资源由本机直接下载，后台任务位于 `.setup/robotwin_20261102/`。

| 目录 | 内容 |
|---|---|
| `models/robotwin_flux2_4b/` | RoboTwin 策略 model.pt、原始训练 config.yaml、归一化 dataset_stats.json、模型卡 |
| `models/flux2_klein_base_4b/` | FLUX2 视觉基础权重和官方 Diffusers VAE |
| `models/Qwen3-4B/` | 完整文本编码器分片、索引、配置和 tokenizer |
| `data/robotwin2.0_archives/` | 官方完整 RoboTwin 数据集 8 个分卷、数据卡、统计 |
| `data/robotwin2.0/` | 完成所有下载与校验后自动解压的数据集；实际根目录见 dataset_extracted.json |
| `configs/local/m5pro_robotwin.env` | 新增的本地路径配置，不覆盖上游或 3090 配置 |

模型固定版本：RoboTwin `ca99fe8a80bb4d7252b733996772347ba361d4f1`；FLUX2 `a3b4f4849157f664bdbc776fd7453c2783562f4d`；Qwen `1cfa9a7208912126459214e8b04321603b3df60c`。
数据集：`yuanty/robotwin2.0-fastwam@aac262c35d02cc71b2f6ef670bd65fd9f2bb2547`，压缩分卷约 79.1 GB。

下载先完成模型，再下载数据。已有同名文件只有内容校验一致才复用，内容不同会报错；不会覆盖。后台采用独立进程、任务锁、curl 续传及重试，下载临时文件以 `.download-part` 结尾。大文件校验 SHA-256，小文件校验 HF Git blob SHA-1。数据集仅在全部文件验证后解压；解压前及逐文件过程中检查磁盘空间。

## 查看进度及恢复

```bash
cd /Users/qklj/work/cx/ImageWAM
cat .setup/robotwin_20261102/download_status.json
tail -n 20 .setup/robotwin_20261102/download.log
tail -n 20 .setup/robotwin_20261102/curl.log
```

`download_completed.json` 仅在全部模型、数据下载校验及数据解压完成后生成。未出现该文件不应宣称资源准备完成。中断后可在旧进程退出后重新运行：

```bash
nohup /usr/bin/python3 -u .setup/robotwin_20261102/download_on_m5pro.py \
  >> .setup/robotwin_20261102/download.log 2>&1 < /dev/null &
```

## 后续推理前置工作

此次只准备代码与资源，未安装 macOS 推理环境，也未执行三图推理。系统 Python 3.9.6 低于项目 Python>=3.10 的要求，应建立独立 arm64 环境，不能沿用 Linux CUDA 的安装命令/虚拟环境。

```bash
source configs/local/m5pro_robotwin.env
```

下载的是公开 Diffusers VAE，不能直接当原生 `ae.safetensors` 加载。建立兼容 PyTorch/safetensors 环境后，可用已复制的转换脚本生成原生 AE（文件已存在时跳过，避免覆盖）：

```bash
if [ ! -e "$FLUX2_AE_MODEL_PATH" ]; then
  python scripts/flux2/convert_public_flux2_ae.py \
    "$FLUX2_AE_DIFFUSERS_PATH" "$FLUX2_AE_MODEL_PATH"
fi
```

三相机入口：`experiments/robotwin/imagewam_policy/deploy_policy.py`。标准 RoboTwin 路径将 head/left/right 三张同步 RGB 拼成 `[1,3,288,256]`，接收匹配的 14 维状态和文本指令。使用对应策略统计进行归一化/反归一化；不要混用 LIBERO 的 7 维动作权重、8 维状态或统计。

MPS 设备/同步/精度及峰值内存仍需适配和验证。此处的数据集可用于离线观测输入测试；完整 RoboTwin 仿真还需要单独安装仿真依赖和场景/机器人资产，不代表复制依赖源码后即能在 macOS 运行。
