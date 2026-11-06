# 脚本入口

正式使用：`infer.py`（accuracy等原接口）、`infer_version.py`（保留版本）、`benchmark.py`（性能测试）。

资产准备：`convert_weights.py`、`prepare_assets.py`、`compact_action_weights.py`。核验工具包括`analyze_accuracy.py`、`compare_traces.py`、`capture_environment.py`及validate/test/audit脚本。

其余experiment/diagnose/sweep/run_*工具是历史验证入口，保留原位置以避免改变__file__路径及脚本间导入。阶段性实验实现集中在根目录experiments及development/legacy_experiments。正式推理不从实验目录导入候选。

使用命令与选择建议见[版本总表](../README_VERSION_SELECTION.md)。
