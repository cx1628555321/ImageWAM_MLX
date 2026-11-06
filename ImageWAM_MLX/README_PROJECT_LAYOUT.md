# 项目目录说明

```text
ImageWAM_MLX/
  README.md                         项目导航
  README_VERSION_SELECTION.md       面向使用者的选型与命令
  README_INFERENCE_VERSIONS.md       启动与复测细节
  README_RETAINED_VERSIONS_20261106.md 版本清单
  README_ROBOTWIN_ACCURACY_20261106.md 正式精度总结
  README_PROJECT_LAYOUT.md          本目录说明
  imagewam_mlx/                     正式实现
  scripts/                         正式入口与历史验证工具（见目录README）
  tests/                           检查代码
  docs/optimization/               分阶段优化分析、测试过程与旧总结
  experiments/                     已按实验批次分组的过程代码/配置
  development/legacy_experiments/   原根目录experiments_*历史脚本
  development/results/             原根目录零散JSON结果
  development/maintenance/          整理清单和修改前代码/文档备份
  records/                         实测证据、输出及已验证源码构建环境
  assets/ examples/ weights/        推理资产、示例、权重（远端）
  .venv/                           已验证旧运行时（远端）
```

`records/background_20261106_xcode/builds/.../venv`是实际使用的新MLX环境，不是可删除的临时结果。移动虚拟环境会破坏绝对路径，因此records保持原位。experiments仍为实验复现入口；不再被正式native-w1/native-w4启动路径导入。

仅删除源码目录中的__pycache__/pyc/pyo及.DS_Store。没有删除权重、原始数据、实验输出、失败证据或虚拟环境。根目录JSON与旧实验目录均归档，非删除。历史脚本的旧实验目录字面路径和根目录JSON引用已更新；历史实验没有全部重跑，不将目录整理称为所有旧实验再次验证。

本地是开发镜像，assets/weights/环境和大体积结果可能仅在M5Pro存在。整理清单及备份在`development/maintenance/layout_20261106/`，可按moves.json反向恢复归档路径。正式启动验证见该目录的validation记录。
