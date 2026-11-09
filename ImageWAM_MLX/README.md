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


## 本地开发副本（2026-11-09）

本目录从m5pro同步源码、配置、文档、轻量模型元数据、日志和非数据类统计报告，保留本地 [模型资产下载说明](README_DOWNLOAD_ASSETS.md)。权重、原始测试输入/输出、GPU原始采样、虚拟环境、编译二进制与归档包未复制。排除规则见 `SOURCE_COPY_EXCLUDES.txt`，逐文件SHA256及复制验证见 `sync/SOURCE_FILES_20261109.json` 和 `LOCAL_COPY_MANIFEST.json`。

旧版本被覆盖文件备份在本目录之外；未删除本地独有文档。远端README中的 `/Users/qklj/...` 和虚拟环境路径属于原测试环境，本地尚未安装MLX运行环境或下载模型，因此本次同步不代表本地已完成推理验证。
