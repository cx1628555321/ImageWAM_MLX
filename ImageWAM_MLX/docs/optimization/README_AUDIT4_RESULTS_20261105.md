# 四项优化方向适配与实测汇总（2026-11-05）

## 结论

本轮完成布局/复制审计、原生 GPU 时间戳、CPU 调用边界与图导出/C++ 投影执行、动态文本与 shapeless 四项实验。**没有发现足以切换正式默认的新稳态提速方案**。生产推理代码未改动，新增代码全部隔离在 `experiments/audit4_20261105/`。

当前 speed 路径在本轮标准输入下，预热 10 次后 30 次完整推理平均 **192.752 ms**。这是当前版本一次新的测量，不是本轮优化收益；不能与历史 198.66 ms 相减宣称提速。保持三图、10 步、无跨请求文本/图像特征缓存、原始动态文本、CPU 动作输出。推理边界不包含加载和文件读写。

设备：Mac mini M5 Pro，15 CPU 核、16 GPU 核，24 GiB 统一内存；macOS 27.2、Python 3.12.13、MLX 0.32.4。项目位于 `/Users/qklj/work/cx/ImageWAM_MLX`。本轮没有重测加载耗时或机器人任务成功率。

|方向|适配/测试结果|处理|
|---|---|---|
|真实 stride / 连续化边界|10 种内核输入签名、1,035 次自定义内核调用；默认连续化内核的已观测输入均连续|未找到可安全删除的新复制点|
|Metal 原生时间戳|30 请求、8,639 个 command buffer；全部 completed，输出一致|保留为诊断工具，不注入正式入口|
|CPU 边界 / 图导出 / C++|真实 W1 投影可导出并在独立 C++ 程序执行，结果一致；交替微测未见实质收益|不切换入口；整网 C++ 移植尚未实施|
|动态长度 / 预热 / shapeless|31/39/79/189 token 混合输入正常；显式预热输出一致；两处 shapeless 均失败|沿用按实际形状编译和现有预热接口|

## GPU 执行证据

|指标|每请求平均|
|---|---:|
|无记录端到端|192.752 ms|
|启用时间戳端到端|192.881 ms|
|有效 command buffer 执行区间并集|191.480 ms|
|首个开始到最后结束跨度|191.914 ms|
|跨度内未被执行区间覆盖的间隙|0.434 ms|
|command buffer 数量|287.967|

8,639 条记录中 6,829 条 end > start，1,810 条 start == end，没有零开始时间；状态均为 Metal completed。合并重叠区间后计算并集，未重复相加。记录结束 pending callbacks = 0。

这说明**命令缓冲区之间的空闲较少**，优先减少 GPU 内部的计算、访存和同步成本。区间包含缓冲区内部等待，不能称为 GPU 活跃率、ALU 利用率或实际带宽，不能据此判定计算/带宽谁占主导。没有取得逐 kernel 时间戳和硬件计数器。前后两组均值差 0.129 ms 也不是随机配对得到的探针开销结论。

工具通过进程内 Objective-C runtime 包装 `AGXG17XFamilyCommandBuffer.commit`，记录 `GPUStartTime/GPUEndTime`。未修改系统或 MLX 安装。该方法依赖实现类，仅供此环境诊断。

## 布局与精度

在关闭编译的诊断路径中，读取真实输入 strides，按 kernel/shape/dtype 首次出现采样。唯一发现的非连续输入为门控量化输入 `[328,18432]`，strides `[27648,1]`，该内核已使用 `ensure_row_contiguous=False` 并处理 stride。其余需连续化内核的已观测输入均连续。

因此不能凭参数名就批量关闭连续化，也不能说编译整网不存在物理复制。逻辑图 `logical_step.dot` 不是物理执行图。此次 eager 诊断的 action 和 normalized 输出与标准 probe 输出逐值相等，max_abs = 0。

## CPU 边界与 C++ 实验

另一个带边界计时的 30 请求实验：平均 wall 200.332 ms，调用线程 CPU 时间 75.759 ms。边界数据如下，不与无探针端到端直接比较：

|边界|每请求次数|wall ms|调用线程 CPU ms|
|---|---:|---:|---:|
|async_eval|11|164.667|60.155|
|compiled_step|10|4.814|4.808|
|MLX → NumPy|1|20.056|0.090|

`async_eval` 不是保证立即返回的接口；上述 wall 包含原生工作/等待。NumPy 边界的约 20 ms 主要表现为等待，不能把它全算作复制成本。线程 CPU 时间不含其他原生工作线程，也不等于 Python 解释器时间。cProfile 中 denoise 的累计时间同样不能解释成纯 Python 开销。

实际热点投影 `(M,N,K)=(16,1024,7168)`，现有 W1 Split-K + residual epilogue：导出 `real_projection.mlxfn`，传入实际捕获的权重、激活、残差、门控及 bias，Python 导入和独立 C++ 导入均与原投影逐值一致。

连续微测（各 200 次）出现较大时段差异，Python compiled/imported 为 0.378/0.233 ms，C++ imported 为 0.384 ms；不据此宣称导入图提速。随后 **300 组随机顺序交替复测**：compiled **0.177295 ms**，imported **0.176657 ms**，差仅 **0.000638 ms**。当前没有值得推进正式替换的局部收益证据。C++ 数据属于单独进程、单投影同步微测，不能外推整网速度。

## 动态输入与 shapeless

每个成功模式运行 32 次混合长度请求，包含 RoboTwin 范围内的 31/39/79 token 和额外 189 token。每种实际长度只观察到一次 prefill 和一次 step Python tracing，后续重复没有继续 tracing。这里统计的是 Python trace，不是驱动每次编译计数；不是本轮重新跑完整 51 样本或所有可能文本长度。

|模式|前 4 次延迟，31/39/79/189 token（ms）|后 28 次混合均值|
|---|---|---:|
|未显式预热|838.755 / 200.881 / 234.676 / 280.972|216.394 ms|
|预热 31/39/79|187.942 / 204.234 / 232.758 / 291.478|216.207 ms|

预热独立花费 790.340 + 201.204 + 234.588 = **1226.132 ms**，未计入请求延迟；未预热的 189 token 仍会正常 trace。32 组 action/normalized 均逐值一致。显式预热只是转移首次成本，不是稳态提速，两组不是随机配对。

`shapeless_step` 和 `shapeless_prefill` 均在首个请求失败：

```
ValueError('[Primitive::output_shapes] Split cannot infer output shapes.')
```

失败路径没有有效输出，精度状态为未验证，不计作通过。要继续需逐项替换/验证形状敏感原语，不能直接全局开关启用；普通编译已覆盖重复形状，不值得仅为减少一次 trace 大规模修改数值路径。

## 复现和记录

以下命令在远端项目根目录执行。脚本使用现有模型与标准三图输入，不下载、不覆盖模型。为保护证据，部分脚本通过 `exist_ok=False` 或独占创建拒绝覆盖；重新运行时请先把脚本的记录目标改成新目录。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
clang -dynamiclib -fobjc-arc -fblocks -framework Foundation -framework Metal experiments/audit4_20261105/gpu_probe.m -o experiments/audit4_20261105/gpu_probe.dylib
.venv/bin/python experiments/audit4_20261105/probe_run.py
.venv/bin/python experiments/audit4_20261105/layout.py
.venv/bin/python experiments/audit4_20261105/boundary.py
.venv/bin/python experiments/audit4_20261105/dynamic_driver.py
PYTHONPATH=. .venv/bin/python experiments/audit4_20261105/export_test.py
PYTHONPATH=. .venv/bin/python experiments/audit4_20261105/native_prepare.py
clang++ -std=c++20 -I.venv/lib/python3.12/site-packages/mlx/include -L.venv/lib/python3.12/site-packages/mlx/lib -Wl,-rpath,/Users/qklj/work/cx/ImageWAM_MLX/.venv/lib/python3.12/site-packages/mlx/lib -lmlx experiments/audit4_20261105/native.cpp -o experiments/audit4_20261105/native
experiments/audit4_20261105/native
.venv/bin/python experiments/audit4_20261105/summarize.py
```

原始数据与输出：[summary.json](../../records/audit4_20261105/summary.json)、[原生投影结果](../../records/audit4_20261105/export/native_summary.json)、[实验目录](../../experiments/audit4_20261105)、[记录目录](../../records/audit4_20261105)。归档 `records/audit4_20261105.tar.gz` 不包含重复的大投影 inputs.safetensors；真实输入仍保存在远端 capture 和 export 目录。

## 下一步优先级

1. **对 GPU 内部热点定点测量**：以实际前缀/动作投影形状做隔离 kernel 微测，结合分块、Split-K 分区和融合边界；局部有收益后必须做正式 CLI 配对和独立进程复核。现有 command buffer 时间戳不足以定位单个内核。
2. **研究缓冲区内部同步与访存**：比较自定义内核调用数量、临时数组生命周期、Split-K 部分和读写量；不要把“buffer 间无空闲”误解为内部没有等待。需更细粒度追踪才能决定收益最大的改动。
3. **低优先级**：仅为了删 Python 而全量 C++ 重写、全局 shapeless、盲目取消连续化。本轮证据未支持它们优先于 GPU 内核工作。

精度声明只涵盖与现有量化 speed 路径的数值一致性，不能等同于原 BF16/PyTorch 精度或 RoboTwin 任务成功率。
