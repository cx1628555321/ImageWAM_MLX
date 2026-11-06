# 2026-11-05：当前预展开 INT8 内核扩大 N 分块

实验目录沿用批次标识 `int8_wide_20261104`。正式 speed 基线包含 gate/concat 量化融合、预展开 INT8 视觉权重和动作 W1。当前生产代码、模型权重及 accuracy 路径均未修改。

## 实现

针对实际使用的预展开 INT8 NAX 内核，将 BN64 改为 BN128，保留 BM64、BK64、threadgroup 权重暂存和行 padding80；测试 WN2 与 WN4。与此前旧 Cider 内核的 tile 测试不同，本轮直接基于当前正式内核。

|配置|BN|WM×WN|线程/组|每组暂存权重|每 simdgroup 输出片大小|
|---|---:|---|---:|---:|---|
|正式 speed|64|2×2|128|5120 bytes|32×32|
|wn2|128|2×2|128|10240 bytes|32×64|
|wn4|128|2×4|256|10240 bytes|32×32|

扩大 N tile 的目标是增加块内激活复用、减少输出线程组数量；代价是更多暂存空间，以及 wn2 更多累加寄存器、wn4 更多线程。保持 INT8 输入/权重、INT32 累加、BF16 输出，未引入新的量化或反量化步骤。M/token 长度动态；尾块保留边界判断。两类 single-block 投影 linear1/linear2 都替换，其他既有优化不变。

源码 `experiments/int8_wide_20261104/candidate.py`，原内核来源和 MIT 许可见生产 `imagewam_mlx/metal/staged_int8_upstream.json`。

## 完整推理交错测试

M5 Pro（15 CPU、16 GPU，24 GiB 统一内存），MLX 0.32.4，`MLX_MAX_OPS_PER_BUFFER=256`。每配置预热 5 次，随机交错测量 100 次；三图、在线文本、状态、horizon16、10 步去噪、固定初始噪声。每次调用重新执行完整 predict 并返回 CPU 动作，计时前后同步；不含加载、文件读取和输出落盘。

候选共享未改变的权重。节省=基线−候选；区间为 10,000 次 paired bootstrap，未校正时序相关性或多重比较。

|配置|端到端平均 ms|节省 ms|节省 95% 区间 ms|
|---|---:|---:|---|
|正式 speed|198.641|0|—|
|BN128 / WN2|200.040|-1.399|[-1.492,-1.306]|
|BN128 / WN4|202.761|-4.120|[-4.180,-4.058]|

本轮完整替换均退化，不能加入默认路径。绝对值不可与其他负载时段的结果相减归因。减少线程组数量不等于实际 GPU 工作更快；寄存器、暂存容量及调度占用是潜在原因，但没有硬件 counter，不能确定具体主因。

## 命令和记录

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
.venv/bin/python experiments/int8_wide_20261104/full.py
.venv/bin/python experiments/int8_wide_20261104/probe.py
```

`full.py` 对应上表端到端；`probe.py` 从真实标准样本前向中捕获第一组 linear1/linear2 的量化输入和权重，只测 GEMM，用于诊断，不是完整推理。重跑应指定新输出目录，保留现有记录。

记录在 `records/int8_wide_20261104/`，包括逐次计时、输出 NPZ 和验证 JSON。后台进程列表单独保存，没有停止其他服务。

## 真实投影诊断与按投影选择

使用真实三图前向捕获的第一层投影输入，分别测量每个 GEMM；量化激活作为编译函数参数传入，避免无参数常量输出复用。各配置预热 10 次、随机交错测量 100 次，包含调用、求值和同步开销。输出经无损 BF16→FP32 转换后比较，全部逐值一致。

|投影输出维度|正式 BN64 ms|BN128/WN2 ms|BN128/WN4 ms|
|---|---:|---:|---:|
|linear1：27648|1.710554|1.824582|1.879323|
|linear2：3072|1.030982|0.963741|1.066473|

WN2 对 linear2 平均节省 0.067241 ms，95% 区间 [0.057569,0.077250]；对 linear1 反而慢 0.114028 ms。这解释了为何不应全量统一改分块，也提供了选择性替换的依据。单算子结果不能直接乘层数作为端到端收益。

`partial.py` 保持 linear1 的 BN64，仅对已有预展开 INT8 输出维度 3072 的 linear2 使用 BN128/WN2。51 样本 action/normalized 逐值一致。100 组完整配对实测：

|配置|端到端平均 ms|
|---|---:|
|正式 speed|198.038137|
|仅 linear2 BN128/WN2|197.320076|

节省 **0.718060 ms（约 0.36%）**，bootstrap 95% 区间 **[0.627786,0.811707] ms**。

```bash
.venv/bin/python experiments/int8_wide_20261104/full_partial.py
```

原始微测 `probe.json` 包含逐次计时、实际形状与编译模板。`probe.log` 保留最初 BF16 转 NumPy 的诊断错误；`probe_v2.log` 保留 JSON dtype 序列化错误，两者都不涉及推理内核修复。最终有效、完整落盘的是 `probe_v3.log` 和 `probe.json`。

## 独立进程复核：不固化

A→B→B→A 各独立进程、单模型驻留，每进程预热 10 次、测量 100 次，另 5 次同步模块诊断：

|进程|配置|平均 ms|P95 ms|
|---|---|---:|---:|
|A1|正式 speed|197.735|198.489|
|B1|仅 linear2 BN128/WN2|205.558|207.157|
|B2|仅 linear2 BN128/WN2|205.920|208.211|
|A2|正式 speed|198.910|200.276|

合并均值：基线 **198.322477 ms**、候选 **205.738680 ms**；候选慢 **7.416203 ms（约 3.74%）**。四个进程输出逐值一致。虽然多模型交错中局部候选有正收益，但独立部署条件下未复现，因此本轮所有候选均不加入默认代码。

每 20 次请求的均值也显示候选连续偏慢：B1 约 205.34–206.21 ms，B2 约 205.02–206.66 ms，不能简单解释为一两个异常峰值。

独立模块诊断各配置汇总 10 次，单位 ms：

|模块|正式 speed|局部候选|
|---|---:|---:|
|文本编码|27.555|28.147|
|CPU 图像预处理|2.601|4.561|
|图像编码|27.912|28.010|
|视觉前缀|83.500|81.779|
|动作去噪|62.505|62.793|

视觉前缀在额外同步的诊断中减少约 1.72 ms，但完整推理反而变慢；CPU 预处理等未改动模块也有不同表现。不能把模块时间相加解释端到端，更不能把视觉前缀收益当作整网收益。图间调度、内存工作集和 command-buffer 行为可能参与其中，当前没有 GPU timeline 证据，尚未确定主因。下一步若继续该候选，应定位正常无额外同步路径的调度差异，而非继续按微基准扩大分块。

## 精度、动态输入与复现补充

- 全量 wn2/wn4：两种候选各 51 个样本，动作和归一化输出逐值一致。
- 局部采用：51 个样本逐值一致。
- 局部候选的 39 tokens/horizon1、189 tokens/horizon16、39 tokens/horizon64 全部逐值一致。
- 此处相对当前 speed；未消除原 speed 与 accuracy/PyTorch 的量化误差，也未验证机器人闭环成功率。

```bash
.venv/bin/python experiments/int8_wide_20261104/abba_partial.py baseline a1
.venv/bin/python experiments/int8_wide_20261104/abba_partial.py partial b1
.venv/bin/python experiments/int8_wide_20261104/abba_partial.py partial b2
.venv/bin/python experiments/int8_wide_20261104/abba_partial.py baseline a2
.venv/bin/python experiments/int8_wide_20261104/dynamic_partial.py
```

ABBA 重跑须使用新标签；full/partial/dynamic/probe 也需使用新输出目录以保留历史。正式启动继续使用现有 `scripts/infer.py --profile speed` 及输入参数，无新增默认参数。

所有结果位于 M5 Pro 的 `/Users/qklj/work/cx/ImageWAM_MLX/records/int8_wide_20261104/`。`full/` 为全量两种分块，`partial/` 为局部采用，`abba/` 为单模型复核，`dynamic/` 为动态边界，`metadata.json` 保存硬件与源码/输入哈希，`source_snapshot/` 保存正式源码和标准输入。归档 `records/int8_wide_20261104_review.tar.gz` 同步到本地并校验 SHA256。
