# ImageWAM MLX 后台优化计划与续接提示词

## 可直接交给后续会话的提示词

你在 m5pro `/Users/qklj/work/cx/ImageWAM_MLX` 继续 ImageWAM MLX 推理优化。首先读取本 README、`records/background_20261105/status.json`、`REPORT.md` 和具体任务日志，检查 runner.pid 对应命令是否仍运行。不要重复启动仍在运行的测试，不要仅凭 PID 文件判断存活。

当前正式可选速度基线是 `--profile speed --prefix-precision all_int8 --text-cache-size 1`：ABBA 157.703 ms；前缀 66.640 ms，动作 10 步 62.518 ms，图像编码 27.756 ms。原 speed 169.287 ms。所有数值是历史同条件实测，不是本轮保证值。默认 accuracy/speed 的策略不可悄悄替换。前缀 INT8＋动作 W4 筛选 158.857 ms，标准参考 RMSE 0.007745；W1 speed 的参考 RMSE 约 0.51844，不得将 speed 输出逐值一致宣传为相对原模型无损。

目标：在三图每次在线编码、H16、10步、动态文本、文本缓存命中同一口径下，找出下一批可以复现的端到端收益。模型加载暂不优化。必须区分已有优化、重复失败实验、新内核/新形状，以及新的上游代码。

按以下方向分析并验证：

1. **GPU/CPU 诊断**：检查完整 Xcode 与 xctrace；获取 GPU capture、命令缓冲提交/开始/结束和 CPU profile。只有真实解析到逐内核时间线时，才宣称得到了 kernel 级瓶颈；命令缓冲时间不能代替 kernel 时间，含同步的模块和不能代替端到端。缺工具链要明确标 blocked，不要改系统 Xcode 选择或停止其他用户服务。
2. **新增前缀 INT8 融合**：针对新增43个矩阵，分别测试 GEMM 输出 bias 融合、双流 QKV/MLP 的 LayerNorm＋调制＋动态量化融合、二者组合。保留原38个已优化矩阵、动态行数/文本长度和BF16输出舍入；归约次序变化必须记录误差。不要固定文本长度，图片沿用RoboTwin尺寸。必须确认中间张量是否实际减少，而非仅改变函数名。
3. **动作原生内核和最新分派**：在原环境测 native W1/W4、qqmm MXFP4/MXFP8/NVFP4；在独立环境测试包含 PR #4796 的固定提交，与该环境自身基线及旧环境对照。#4796 修复多行 qqmm 的错误向量分派；同时检查 #4682 staged NAX/Split-K 的实际源码覆盖。算子宣传倍数不是当前端到端收益。记录 dtype、原始权重来源、group size、dispatch 和激活量化/反量化路径。
4. **affine INT8 PR #4802**：独立环境测试 W4/W8 的 qqmm 动作路径，记录 M16 是否进入回退；不得绕过 M>=32 的条件后宣称原生INT8快路径。此PR尚未合并；fetch后记录完整SHA。新环境只做实验兼容覆盖，不改生产runtime版本检查。
5. **局部数学模式**：测试 selected GEMM/量化/epilogue 的 relaxed、fast，与safe比较。禁止全局修改masked softmax；记录数值变化、NaN/Inf。不能把API可接受视为正确或更快。

测试方法：每个候选独立进程、10次预热、50次正式、另50次同步模块统计；输入/噪声相同，GPU计时同步，输出保存。验证51个现有案例以及H1/H64/长文本三组动态输入，保存action、normalized与动态逐步轨迹。时间统计保留所有样本。可重现收益候选做独立ABBA，保持每进程只驻留一个模型。测试时如果其他Python模型任务出现，停止自己的子进程并等待设备空闲，不能终止其他任务。进程检测不能证明GPU绝对空闲，报告这一限制。

对所有候选汇总：加载、首次、steady mean/P95、前缀、动作、参考RMSE、相对本轮speed误差、动态输入、源码SHA、命令、是否进入预期内核、被采用/拒绝/阻塞原因。有效方案经复测才考虑集成；微测收益和不足1ms的单次改善不得直接推广。此次后台任务不自动切换生产默认。

## 已编排后台任务

脚本：`experiments/background_20261105/runner.py`。

- profile：工具链检测、当前all_int8基线的命令缓冲时间线/CPU profile/GPU capture。
- stock_baseline：本轮基线。
- stock：fuse_bias、fuse_norm、fuse_both、math_relaxed、math_fast、native_w1、native_w4、qq_mxfp4、qq_mxfp8、qq_nvfp4。
- route4796：独立构建 `refs/pull/4796/head` 并记录SHA；基线、native W1/W4、qqmm三种浮点量化。
- affine4802：独立构建 `refs/pull/4802/head` 并记录SHA；基线、动作 affine W4/W8、前缀 affine W4/W8。
- 对筛选至少快1%的候选，最多选择两个执行独立ABBA复测。数值误差照实记录，进入复测不代表精度验收。

执行前源码哈希保存到source_hashes.json。上游独立环境位于records下，不替换`.venv`。实验进程对上游新版本的runtime gate做内存内兼容覆盖，真实MLX版本保存在environment.json；该覆盖不是新版本兼容性保证。若构建或测试失败，后续独立任务继续，失败日志保留。

后台是已经编写好的测试队列，不是常驻AI开发代理。它能够执行、等待、重试、保存和汇总；无法在会话暂停后自行研究并修复未知编译错误。遇到缺少完整Xcode、上游API不兼容等问题会明确记录，恢复会话后根据上面的提示词继续处理。不能承诺所有方向都会通过或有收益。

## 查看与停止

在m5pro项目根目录：

```bash
cat records/background_20261105/status.json
cat records/background_20261105/REPORT.md
tail -50 records/background_20261105/launcher.log
ps -p "$(cat records/background_20261105/runner.pid)" -o pid,etime,args
```

安全停止本队列：

```bash
touch records/background_20261105/STOP
```

runner在下一次轮询停止自己的进程组，保留结果。不要使用pkill python。首次启动使用独立session、关闭stdin、文件日志；SSH和本地会话断开不影响队列，但服务器关机/重启会停止任务。本脚本使用排他锁防并发启动；不要在暂停后直接重跑脚本覆盖已存在输出。恢复时按status挑出未完成任务并使用新的输出目录。

## 官方来源

- https://github.com/ml-explore/mlx/pull/4796 ：qqmm路由修复，已合并main；不是0.32.4已含功能。
- https://github.com/ml-explore/mlx/pull/4682 ：staged NAX/Split-K。
- https://github.com/ml-explore/mlx/pull/4802 ：affine qqmm INT8，未合并，M16回退需记录。
- https://github.com/ml-explore/mlx/pull/4722 ：已有INT8实现来源，避免重复算收益。
- https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html#math-mode ：safe/relaxed/fast。
