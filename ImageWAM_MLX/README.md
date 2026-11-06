> 本地代码副本：未复制模型权重、RoboTwin数据/测试样本、虚拟环境或大型实验记录。下载地址、固定版本命令和恢复步骤见 [权重与数据下载说明](README_DOWNLOAD_ASSETS.md)。原文中的 `/Users/qklj/` 路径是远端测试路径，本地需调整。

# ImageWAM MLX

M5Pro上的RoboTwin三图推理。项目目录：`/Users/qklj/work/cx/ImageWAM_MLX`。

## 使用入口

- **[版本选型总表](README_VERSION_SELECTION.md)**：一张表查看性能、多样本精度、主要优化与启动命令。优先native-w4；更低平均误差选择action-bf16。
- [版本启动与复测细节](README_INFERENCE_VERSIONS.md)
- [保留版本列表](README_RETAINED_VERSIONS_20261106.md)
- [RoboTwin六版本精度报告](README_ROBOTWIN_ACCURACY_20261106.md)
- [项目目录说明](README_PROJECT_LAYOUT.md)

## 开发与历史记录

- [优化过程文档索引](docs/optimization/README.md)
- [过程代码与配置说明](development/README.md)

正式代码在`imagewam_mlx/`，使用入口在`scripts/infer.py`和`scripts/infer_version.py`。原始权重、数据、环境与测试证据均保留。详细实验结论不在此重复维护。
