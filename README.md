# ImageWAM 与 ImageWAM MLX

本仓库汇总 M5Pro 适配后的两个项目代码目录，保留相邻布局，便于代码中使用 `../ImageWAM` 相对路径。发布整理日期：2026-11-06。

| 目录 | 内容 | 入口 |
|---|---|---|
| `ImageWAM_MLX/` | Apple Silicon MLX 推理、量化优化与精度/性能记录 | [版本选型总表](ImageWAM_MLX/README_VERSION_SELECTION.md) |
| `ImageWAM/` | 原 ImageWAM/PyTorch 代码、适配工具与配置 | [项目 README](ImageWAM/README.md) |

## 使用说明

- [权重与 RoboTwin 数据下载、转换和恢复命令](ImageWAM_MLX/README_DOWNLOAD_ASSETS.md)
- [六个版本的 RoboTwin 多样本精度结果](ImageWAM_MLX/README_ROBOTWIN_ACCURACY_20261106.md)
- [MLX 项目目录说明](ImageWAM_MLX/README_PROJECT_LAYOUT.md)

优先使用 `native-w4` 兼顾速度与数值精度；`action-bf16` 平均数值偏差更低；W1 系列仅保留速度极限实验用途。测试结论限于报告中的设备、输入和运行时；离线误差不是机器人闭环成功率。

## 仓库边界

此仓库包含代码、配置、文档及轻量 tokenizer 资源，不包含模型权重、RoboTwin 数据/测试样本、原始测试输出、虚拟环境和编译产物。报告中指向 `records/` 的原始证据保存在开发服务器，不随本仓库发布。权重和数据的来源、固定 revision 与下载命令见下载说明。

旧文档中的 `/Users/qklj/...`、`/Users/chenxiang/...` 为已有测试/开发路径，使用时按本机路径调整。`native-w4` 的已验证 MLX 源码构建环境未包含在仓库中；下载代码不等于已具备同样的运行环境。

第三方目录保留源码及其原有许可证，移除了嵌套 `.git` 历史。本仓库整理不改变上游和第三方许可证；使用相应代码前请参阅各目录的 LICENSE/NOTICE。
