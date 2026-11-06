# 预展开 INT8 GEMM 线程块调度实验（2026-11-04）

本轮基于当前正式 speed（含 gate/concat 量化融合、冻结权重预展开 INT8、动作 W1）测试输出 tile 的遍历顺序。此前 BF16 NAX swizzle 与旧 Cider 分块的结论不直接代表当前 INT8 内核，因此对当前路径重新验证。

## 改动

只改变逻辑线程块坐标：

```text
logical_m = group_y * swizzle + group_x % swizzle
logical_n = group_x / swizzle
```

测试 swizzle=2/4/8，尝试让相邻 M tile 更接近地读取同一组权重。保留 BM=64、BN=64、BK=64、threadgroup 权重暂存、行 padding80、INT8 权重和激活、INT32 累加、BF16 输出。每个元素的累加顺序和量化不变，没有增加反量化或请求级缓存。动态 M 计算 grid；多出的整个线程组在 barrier 前统一退出。两类单流投影 linear1/linear2 都使用新调度，保留其他既有优化。

候选位于 `experiments/int8_schedule_20261104/candidate.py`，生产代码和权重未改动。继承的内核沿用生产 `metal/staged_int8_upstream.json` 及 MLX MIT 许可来源。

## 环境与实验口径

M5 Pro：15 CPU / 16 GPU 核、24 GiB 统一内存；MLX 0.32.4；`MLX_MAX_OPS_PER_BUFFER=256`。项目根目录 `/Users/qklj/work/cx/ImageWAM_MLX`。

测试开始时 mediaanalysisd 约 105% CPU，另有 VideoToolbox 编解码和虚拟机活动。未停止其他服务；CPU 进程列表不能量化 GPU 干扰。本轮不与历史约 198–200 ms 跨时段相减计算收益。

每次完整推理重新处理三图、状态和文本，执行视觉前缀、10 步动作去噪并返回 CPU 动作；标准 horizon16、固定初始噪声。不包含模型加载、文件读取和保存。计时前后同步 GPU。四个实例共享未改动权重，避免额外模型权重工作集。

## 随机交错完整推理

每配置预热 5 次、测量 100 次。节省=基线−候选；区间为 10,000 次 paired bootstrap，未校正时序相关性、候选选择与多重比较。

|配置|端到端平均 ms|节省 ms|节省的 95% 区间 ms|
|---|---:|---:|---|
|正式 speed|223.577|0|—|
|sw2|221.037|2.541|[-2.793, 7.936]|
|sw4|223.193|0.385|[-5.835, 6.484]|
|sw8|225.787|-2.210|[-8.601, 4.065]|

全部区间跨零，不能据此认定提速。sw2 进入独立进程复核。

## 复现

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
.venv/bin/python experiments/int8_schedule_20261104/full.py
.venv/bin/python experiments/int8_schedule_20261104/abba.py baseline a1
.venv/bin/python experiments/int8_schedule_20261104/abba.py sw2 b1
.venv/bin/python experiments/int8_schedule_20261104/abba.py sw2 b2
.venv/bin/python experiments/int8_schedule_20261104/abba.py baseline a2
.venv/bin/python experiments/int8_schedule_20261104/dynamic.py 2
```

重跑需使用新的输出目录/标签，保留历史记录。完整计时与逐样本动作输出在 `records/int8_schedule_20261104/`；`full/results.json` 保存逐次计时，`full/validation.json` 保存 51 样本验证结果。模块独立同步剖析会改变调度，其各项不能相加替代端到端。

## 精度与动态性

sw2/sw4/sw8 均通过已有 51 个真实样本及文本长度案例的 action/normalized 逐值一致检查，各案例输出保存在 `full/`。测试输入索引沿用 `../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full/comparison.json`，根据标签读取对应三图和状态。标准初始噪声来自 `examples/robotwin_ep0_frame0/initial_noise_fp32.npy`。

一致性仅针对当前 speed，不表示其原有量化误差消失，也不代表闭环任务成功率已验证。矩阵维度和动态文本接口未改变。

## 如何解释结果

该调度改变的是相邻线程块访问输入/权重的顺序，不减少 GEMM 算术量，也不减少每块内部的 K 循环或必要屏障。增加 M 方向邻近性可能改善权重复用，同时减少 N 方向邻近性并影响激活复用和并行调度。因此需要完整推理测试，不能仅依据少量访问或更好的局部计时判断收益。本轮没有 GPU counter/timeline，不能声称已证明具体缓存命中率变化。

## 独立进程最终复核与决策

ABBA 各使用新进程，每组预热 10 次、稳态 100 次、另 5 次同步模块诊断：

|进程|实现|平均 ms|P95 ms|
|---|---|---:|---:|
|A1|正式 speed|226.014|262.782|
|B1|sw2|229.338|263.769|
|B2|sw2|225.826|255.966|
|A2|正式 speed|221.016|256.177|

合并均值：正式 speed **223.515 ms**，sw2 **227.582 ms**，候选慢 **4.067 ms（约 1.82%）**。四组最终动作/归一化输出逐值一致。本轮没有确认新增端到端收益，不固化任何候选。生产 `imagewam_mlx/` 文件哈希与上一轮记录全部一致。

sw2 动态边界：39 tokens/horizon1、189 tokens/horizon16、39 tokens/horizon64，均与正式 speed 逐值一致。该项输出、初始噪声及结果位于 `dynamic/`。

独立同步模块诊断各汇总 10 次（单位 ms）：

|模块|正式 speed|sw2|
|---|---:|---:|
|文本编码|31.481|33.930|
|图像编码|35.249|30.529|
|视觉前缀|90.966|92.170|
|动作去噪|72.248|69.249|

未改动的模块也出现波动，不能将它们的差异归因为 swizzle；模块同步改变调度，表格不能相加替代端到端。视觉前缀和动作去噪仍是主要耗时部分。本次仅修改遍历顺序，没有改变这两部分的算术工作量。

后续优先验证当前预展开 INT8 内核的 N 方向 tile 扩大（例如 BN64→128）：它与本轮遍历重排不同，会改变块内激活复用、线程组数量和寄存器占用。此前旧 Cider 内核的分块结果不能替代当前内核验证；需先比较真实投影形状，再检查整网及动态输入，不能预先承诺收益。

本轮源码、标准输入、设备信息及 SHA256 位于 `source_snapshot/` 和 `metadata.json`；完整记录归档为 `records/int8_schedule_20261104_review.tar.gz`。
