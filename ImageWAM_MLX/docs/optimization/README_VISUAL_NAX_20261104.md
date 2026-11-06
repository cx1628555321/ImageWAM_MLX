# BF16 NAX 视觉投影分块与输出融合（2026-11-04）

完成两类投影各七种分块的微测、三个候选的端到端验证和动态边界验证。本轮没有确认新的稳定加速，不修改正式 `--profile speed` 或 accuracy 分支。此前最佳独立均值 226.641 ms 为历史数据，不与本轮后台负载下的数据直接比较。

## 方法与环境

M5 Pro：15 CPU / 16 GPU 核，24 GiB 统一内存，MLX 0.32.4，Python 3.12.13。远端 `/Users/qklj/work/cx/ImageWAM_MLX`，所有代码及记录均在此项目内。元数据、源码哈希和结束时进程列表保存在 records/visual_nax_20261104。

复用安装版本 MLX `steel/gemm/gemm_nax.h` 的 BF16 NAX 累加循环，保留上游 LICENSE 和源文件哈希。不是重新实现标量 GEMM，也没有把 BF16 权重降精度。

- 输出候选替换单流 0–18 层的 linear2，把 BF16 输出、gate、residual 合入内核。视觉权重实际上无 bias，通用内核传零 bias。最后 KV-only 层维持原样。
- 输入候选仅替换单流 0–18 层的 linear1。输入、权重均 BF16，FP32 累加、BF16 输出；没有新增权重反量化。
- 显式检查有效 NAX fragment 组合和 N/K 对齐；M 来自实际输入，末尾不足一块时使用原生安全加载和边界写回。未固定文本长度。

两类微测各七种 `(BM, BN, WM, WN, BK)`，每种预热 3 次、40 轮随机交错测试。真实输入来自标准样本的首个单流块，不代表其他层全部相同。输入投影最佳均值约 2.912 ms，基线 3.129 ms，七种输出均逐值一致。输出融合微测最好均值约 1.671 ms，基线 1.718 ms，但视觉激活最大绝对差 0.0625；没有证据把差异唯一归因于某个舍入步骤。

端到端使用正式最快 speed、相同三张 RoboTwin 图像/状态/文本/噪声、16×14 动作和 10 步去噪，每配置预热 5 次，100 轮随机顺序配对。包含在线编码、预处理到 CPU 反归一化动作，不含读取文件、模型加载或落盘；计时前后同步。另保存每配置 5 次同步模块记录，模块时间不可直接求和作为端到端。

## 实测结果

“节省”为基线减候选；95% 区间由逐轮差值 10000 次 bootstrap 计算，仅描述本轮，未校正时序相关性或多重比较。输出误差相对同轮正式 speed，不是相对 PyTorch 或闭环机器人成功率。

|候选|同期基线 ms|候选 ms|节省 ms|节省 95% 区间 ms|样本动作 max abs|
|---|---:|---:|---:|---|---:|
|nax64|262.949|269.379|-6.429|[-12.235, -0.596]|0.01600122|
|nax128|262.949|264.770|-1.820|[-6.979, 3.220]|0.01600122|
|input64|266.204|264.386|1.818|[-2.815, 6.326]|0.00000000|

- nax64：输出融合，64×64，2×2 SIMD groups，BK=32。本轮明显变慢。
- nax128：输出融合，64×128，2×2 SIMD groups，BK=64。未确认收益。
- input64：仅输入投影，64×64，2×2 SIMD groups，BK=64。均值改善约 0.68%，但区间跨零，保留为低负载时复测候选。

测试期间 mediaanalysisd 约 115% CPU，视频编解码服务和虚拟机同时活动。未停止这些进程，也未测 GPU timeline，因此不能确定各服务 GPU 占用或将所有波动归因于某个服务。

## 动态输入与数值结果

输出融合完成 31、79、199 token × horizon 1、16、64 的 9 种唯一组合；脚本额外重复 31 token 三种 horizon，总共执行 12 组。两种输出融合最大动作绝对差约 0.0169076，所有输出有限，不宣称无损。

输入投影完成 31、39、79、199 token × horizon 1、16、64 的 12 组，最大动作绝对差为 **0.0**。测试使用相同三张图像和状态，变更文本及 horizon；不是遍历整个 RoboTwin 数据集。每组保存 token IDs、动作和归一化动作，以供后续复核。

原始记录：
- [输出投影微测](../../records/visual_nax_20261104/probe/probe.json)
- [输入投影微测](../../records/visual_nax_20261104/input_probe/probe.json)
- [输出融合端到端](../../records/visual_nax_20261104/paired/results.json)
- [输入投影端到端](../../records/visual_nax_20261104/input_paired/results.json)
- [输出动态验证](../../records/visual_nax_20261104/validation/results.json)
- [输入动态验证](../../records/visual_nax_20261104/input_validation/results.json)

## 复现

输出目录需不存在，重跑时将脚本内 root 改成新目录；不要删除或覆盖本轮结果。测试顺序执行，不并行运行 GPU 工作。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
.venv/bin/python experiments_visual_nax_20261104/build.py
.venv/bin/python experiments_visual_nax_20261104/probe.py
.venv/bin/python experiments_visual_nax_20261104/input_probe.py
# 各候选端到端耗时见上表
.venv/bin/python experiments_visual_nax_20261104/paired.py
.venv/bin/python experiments_visual_nax_20261104/input_paired.py
.venv/bin/python experiments_visual_nax_20261104/validate.py
.venv/bin/python experiments_visual_nax_20261104/input_validate.py
```

## 后续判断

本轮证明复用 NAX 并融合输出，不足以保证完整图提速；不能将微测每层节省简单乘 19 当作模型收益。大 GEMM 的调度、原生路径的编译融合及共享系统负载可能改变结果，尚无 kernel timeline 分离这些原因。

优先在稳定负载下复测 input64；若收益仍不稳定，继续保留原生 GEMM。进一步实验可调整线程块遍历顺序以改善权重/输入缓存局部性，或仅对实际获益形状启用候选；这些方向未在本轮实现，不计入已有收益。当前所有候选均未进入默认路径。
