# 本地代码副本：权重与 RoboTwin 数据下载

更新时间：2026-11-06。本地代码目录为 `/Users/chenxiang/ImageWAM` 与 `/Users/chenxiang/ImageWAM_MLX`，来自M5Pro适配后的项目，不是重新克隆的未适配上游。

本次只复制代码、配置、文档、轻量tokenizer资源。**没有复制模型权重、RoboTwin数据/示例、测试输出、虚拟环境、编译产物和大型归档，也没有执行下面的大文件下载。** 下载前先检查磁盘空间；复制前本机仅剩约14GiB，不足以恢复所有模型或完整数据。远端参考占用：ImageWAM/models约18GiB，MLX/weights约21GiB，RoboTwin/data含压缩包与解压数据约148GiB。

## 下载来源与固定版本

以下仓库和版本来自M5Pro已有下载记录，仓库页面已核对。精度实验使用的是FastWAM预处理版RoboTwin，不能用任意同名数据替换后声称复现同一结果。

| 内容 | 官方发布地址 | 本项目使用的revision | 目标位置（相对ImageWAM） |
|---|---|---|---|
| ImageWAM RoboTwin 检查点、模型配置、归一化统计 | https://huggingface.co/yuyangalin/ImageWAM-FLUX.2-4B-RoboTwin | `ca99fe8a80bb4d7252b733996772347ba361d4f1` | `models/robotwin_flux2_4b/` |
| Qwen3-4B 文本编码器与tokenizer | https://huggingface.co/Qwen/Qwen3-4B | `1cfa9a7208912126459214e8b04321603b3df60c` | `models/Qwen3-4B/` |
| FLUX.2 klein-base-4B **仅VAE** | https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B | `a3b4f4849157f664bdbc776fd7453c2783562f4d` | `models/flux2_klein_base_4b/vae/` |
| FastWAM预处理RoboTwin数据及统计 | https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam | `aac262c35d02cc71b2f6ef670bd65fd9f2bb2547` | `data/robotwin2.0_archives/`，解压见下文 |

上游ImageWAM代码：https://github.com/yuyangalin/ImageWAM 。上游RoboTwin：https://github.com/RoboTwin-Platform/RoboTwin 。上述代码链接仅用于追溯；本地已复制适配代码，不要直接覆盖。

## 1. 准备独立下载/转换环境

在有足够磁盘空间的机器执行，建议Python3.12。此环境用于下载和CPU权重转换，不是性能测试用的MLX环境。

```bash
cd /Users/chenxiang/ImageWAM
python3.12 -m venv .venv-download
.venv-download/bin/python -m pip install --upgrade pip
.venv-download/bin/python -m pip install huggingface_hub torch safetensors einops
```

公开仓库可先匿名下载；如服务返回需要授权，在浏览器完成相应许可后执行 `.venv-download/bin/hf auth login`。不要把访问token写进README或代码。

## 2. 下载推理所需权重与配置

```bash
cd /Users/chenxiang/ImageWAM

.venv-download/bin/hf download yuyangalin/ImageWAM-FLUX.2-4B-RoboTwin \
  --revision ca99fe8a80bb4d7252b733996772347ba361d4f1 \
  --include 'model.pt' 'config.yaml' 'dataset_stats.json' 'README.md' \
  --local-dir models/robotwin_flux2_4b

.venv-download/bin/hf download Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --include '*.safetensors' '*.json' 'merges.txt' \
  --local-dir models/Qwen3-4B

.venv-download/bin/hf download black-forest-labs/FLUX.2-klein-base-4B \
  --revision a3b4f4849157f664bdbc776fd7453c2783562f4d \
  --include 'vae/config.json' 'vae/diffusion_pytorch_model.safetensors' \
  --local-dir models/flux2_klein_base_4b
```

**不需要下载`flux-2-klein-base-4b.safetensors`或完整基础transformer目录**：当前动作推理的视觉/动作主干来自ImageWAM的model.pt，只额外使用FLUX.2的VAE。

VAE下载的是Diffusers格式，现有代码需要原生`ae.safetensors`，不能简单重命名。使用仓库已有转换器：

```bash
cd /Users/chenxiang/ImageWAM
# 避免覆盖已有转换结果；已有文件时此命令不执行。
test -e models/flux2_klein_base_4b/ae.safetensors || \
PYTHONPATH="$PWD/third_party/flux2/src${PYTHONPATH:+:$PYTHONPATH}" \
.venv-download/bin/python scripts/flux2/convert_public_flux2_ae.py \
  models/flux2_klein_base_4b/vae/diffusion_pytorch_model.safetensors \
  models/flux2_klein_base_4b/ae.safetensors
```

转换器按原生AutoEncoder结构校验键名和形状，不修改下载源权重。

## 3. 生成MLX权重

本项目的MLX裁剪权重是从上面下载的原始权重本地转换得到，没有另一个已核验的公开MLX下载仓库。目标目录必须不存在，脚本会拒绝覆盖。

```bash
cd /Users/chenxiang/ImageWAM_MLX
mkdir -p weights
../ImageWAM/.venv-download/bin/python scripts/convert_weights.py \
  --source ../ImageWAM --output weights/bf16

../ImageWAM/.venv-download/bin/python scripts/compact_action_weights.py
```

`native-w4`、`native-w1`从这些BF16源权重在加载时量化；`action-bf16`保留原始动作BF16精度，无需额外下载W4/W1模型包。

轻量`assets/models`中的tokenizer与统计文件已复制。如需从已下载源恢复，可执行下面的无覆盖复制：

```bash
cd /Users/chenxiang/ImageWAM_MLX
mkdir -p assets/models/Qwen3-4B assets/models/robotwin_flux2_4b
for name in config.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do
  test -e "assets/models/Qwen3-4B/$name" || \
    cp "../ImageWAM/models/Qwen3-4B/$name" "assets/models/Qwen3-4B/$name"
done
test -e assets/models/robotwin_flux2_4b/dataset_stats.json || \
  cp ../ImageWAM/models/robotwin_flux2_4b/dataset_stats.json assets/models/robotwin_flux2_4b/
```

## 4. 下载并解压RoboTwin评估数据

该发布版提供8个分卷。只有全部下载完成后才能拼接解压。它的数据元信息标记train，是本项目离线精度样本的来源，**不是独立闭环测试集**。

```bash
cd /Users/chenxiang/ImageWAM
.venv-download/bin/hf download yuanty/robotwin2.0-fastwam \
  --repo-type dataset \
  --revision aac262c35d02cc71b2f6ef670bd65fd9f2bb2547 \
  --include 'robotwin2.0.tar.gz.part-*' 'dataset_stats.json' 'README.md' \
  --local-dir data/robotwin2.0_archives

# 先检查8个分卷都存在。解压目标应为空，避免覆盖已有数据。
for part in 00 01 02 03 04 05 06 07; do
  test -s "data/robotwin2.0_archives/robotwin2.0.tar.gz.part-$part" || exit 1
done
mkdir -p data/robotwin2.0
if test ! -e data/robotwin2.0/robotwin2.0; then
  (set -o pipefail; cat data/robotwin2.0_archives/robotwin2.0.tar.gz.part-* | tar -xzf - -C data/robotwin2.0)
fi
test -e data/robotwin2.0/dataset_stats.json || \
  cp data/robotwin2.0_archives/dataset_stats.json data/robotwin2.0/dataset_stats.json
```

目录应为`ImageWAM/data/robotwin2.0/robotwin2.0/{data,meta,videos}`。若解压中断，不要把残留目录当作完整数据。

## 5. 恢复本项目72组真实输入

本次复制排除了examples和records里的测试样本及参考输出；下载数据后用保留的采样脚本重新生成输入。目录已存在时脚本拒绝覆盖。

```bash
cd /Users/chenxiang/ImageWAM
.venv-download/bin/python -m pip install numpy pyarrow av
cd ../ImageWAM_MLX
mkdir -p records
../ImageWAM/.venv-download/bin/python experiments/robotwin_accuracy_20261106/prepare.py
```

生成位置为`records/robotwin_accuracy_20261106/inputs/`，包含36个三图样本、示范动作、两份噪声；manifest包含对应prompt。可将这些输入传给推理接口。六版本参考输出需重跑生成，不是数据仓库中可下载的模型预测。

## 6. 本地运行环境与路径说明

本次复制不包含`.venv`、`.venv-mps`或`records/.../venv`。原README里的`/Users/qklj/...`是M5Pro实测路径，迁移后应改为`/Users/chenxiang/...`。权重下载与转换完成并不意味着已具备原性能环境。

原`native-w4`入口严格检查MLX `0.32.5.dev20261106`；它是项目验证过的源码构建，不是可随意用最新pip版本替代的环境。恢复该环境需依照`docs/optimization/README_XCODE_MLX_VALIDATION_20261106.md`及`experiments/background_20261105/`的构建记录/脚本重新构建。源码基点为`cab962d052e65befda7e14c573a955baa30b7114`，构建选项亦影响性能；本次没有安装、重建或验证本地推理环境。旧MLX环境依赖记录见`ImageWAM_MLX/requirements-lock.txt`。

要在本地复用正式启动命令，还需提供自己的三图输入/完整prompt，或使用第5步恢复的输入。原`examples/robotwin_ep0_frame0`特定历史参考样本未随代码复制，不能直接运行旧命令后假定文件已存在。
