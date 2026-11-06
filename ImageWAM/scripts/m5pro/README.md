# M5 Pro RoboTwin 开发入口

正式代码位于 scripts/m5pro；历史归档不作为第二个维护入口。

| 文件 | 职责 |
|---|---|
| benchmark_robotwin.py | 数据准备、加载、首次/预热/稳态/模块测量及结果保存 |
| robotwin_io.py | 三图拼接和状态/动作归一化 |
| timing.py | MPS 同步计时、统计、原子 JSON 写入 |
| profiling.py | 复用原 FLUX.2 模块边界，使用 MPS Recorder |
| audit.py | 设备和模块调用核查、输入扰动、历史精度对比、自动源码归档 |
| validate_io.py | 与原 RoboTwin 入口逐元素核对预处理和后处理 |

旧 scripts/flux2/benchmark_m5pro_robotwin.py 和 validate_m5pro_robotwin.py 保留为兼容入口。
原权重、模型实现和 action_only 加载器不修改。

## 运行

在项目根目录使用已有 .venv-mps 环境；依赖锁位于 requirements/m5pro_inference_20261103.txt。
每次指定不存在的新结果目录。若更换 episode/frame，省略 --reference-dir。

```bash
cd /Users/qklj/work/cx/ImageWAM
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv-mps/bin/python -u scripts/m5pro/benchmark_robotwin.py \
  --output-dir evaluate_results/m5pro_robotwin/NEW_RUN \
  --episode 0 --frame 0 --warmup 6 --iterations 20 --profile-iterations 5 \
  --audit --reference-dir evaluate_results/m5pro_robotwin/20261103_100525

.venv-mps/bin/python scripts/m5pro/validate_io.py \
  evaluate_results/m5pro_robotwin/NEW_RUN
```

## 记录

每轮结果位于 evaluate_results/m5pro_robotwin/<run>/；开发说明位于 development_records/m5pro_robotwin/<date_task>/。
脚本自动保存源码快照、环境版本和输入/输出哈希，不依赖临时目录或手工 finalize 脚本。
历史 docs/M5PRO_ROBOTWIN_INFERENCE_20261103.md 和 20261103_baseline 归档保持原义。

- benchmark.json / iterations.csv / spans.csv / summary.md：原始计时和汇总。
- execution_audit.json：--audit 时生成，所有审计在性能测量之后执行。
- source_snapshot.tar.gz / source_manifest.json / artifact_sha256.json / requirements-lock.txt：复现资料。
- inputs.npz / input_manifest.json / initial_noise_fp32.npy：真实三图、状态、指令与噪声。
- reference_action.npy、normalized_first_000.npy、每轮 actions_* / normalized_*：精度比较输出。

端到端从已解码三图、状态、文本到 CPU 动作，不含视频读取、加载、文件写入或仿真。
模块是同步 inclusive wall 时间，父子项不可相加；同步本身会增加开销。
未控制 OS 页缓存和 Metal 编译缓存；固定单观测重复性不等于跨平台精度或任务成功率。
内存数据为迭代结束采样，不是峰值。

## Dynamic text optimization

Use --optimization dynamic for untruncated variable-length text. Baseline remains the default. See ../../development_records/m5pro_robotwin/20261103_dynamic_optimization/README.md for tested coverage, numerical differences and commands.

- [各测试启动命令、硬件环境与端到端耗时](../../README_M5PRO_TEST_COMMANDS.md)
