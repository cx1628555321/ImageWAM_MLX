# 2026-11-05：整图调度与额外 CPU 开销诊断

目标是解释此前 BN128 局部 GEMM 变快、独立进程端到端却变慢的现象。当前正式 speed 及权重保持不变。所有实验位于 `experiments/graph_dispatch_20261105/`，结果位于 `records/graph_dispatch_20261105/`。

环境：M5 Pro（15 CPU / 16 GPU，24 GiB 统一内存），MLX 0.32.4。每个测试进程只驻留一个模型，按序运行，不并发争用 GPU。候选沿用仅 linear2 BN128/WN2，未引入新的算术变化。

## 编译入口与 command-buffer 扫描

每进程预热 10 次、测量 40 次，另 3 次独立同步模块诊断。三图、在线文本、状态、固定噪声、10 步、horizon16；计时前后同步并返回 CPU 动作，不含文件加载/保存。

|顺序/配置|ops/buffer|端到端平均 ms|P95 ms|
|---|---:|---:|---:|
|原生正式入口|256|197.696|198.013|
|仅用 lambda 重新封装原生 prefill|256|197.478|197.835|
|BN128，bound method 编译|256|205.658|207.438|
|BN128，lambda 编译|256|197.072|197.488|
|BN128，bound method 编译|128|197.066|197.518|
|BN128，bound method 编译|512|196.986|197.596|
|原生正式入口，再测|256|206.846|208.659|

这组单次顺序筛选不能确认最优配置：原生入口也出现约 206 ms 慢态，bound 编译入口在其他阈值下又可进入约 197 ms 快态；不能只归因为某一种包装或 BN128。

慢态 CPU 图像预处理约 4.5–5 ms，快态约 2–2.6 ms。图像编码和动作去噪在额外同步剖析中差异较小，提示需要检查 CPU 调度/提交路径，而不是继续只修改 GEMM。模块同步改变实际调度，不能简单相加解释端到端。

## 当前线程 QoS 实验

依据远端 Apple SDK `pthread/qos.h`、`sys/qos.h` 核对 API，原始头文件归档。仅使用 `pthread_set_qos_class_self_np` 修改测试进程当前线程的请求 QoS，在导入 MLX 前执行；未修改其他服务、系统调度配置或固定 CPU 核心。

默认 QoS 实测为十进制 21（0x15）；USER_INITIATED 为 25（0x19），USER_INTERACTIVE 为 33（0x21）。请求返回值及读回值均记录。每进程 10 次预热、60 次计时，另 3 次模块诊断。

|配置|平均 ms|每次进程 CPU 时间 ms|硬缺页计数|
|---|---:|---:|---:|
|原生，继承 QoS|197.697|37.364|0|
|原生，initiated|197.707|38.210|0|
|BN128 bound，initiated|205.829|100.918|0|
|原生，interactive|197.756|38.203|0|
|BN128 bound，interactive|205.673|100.321|0|
|原生，继承 QoS，再测|200.166|38.679|2|

QoS 请求生效，但未消除候选慢态，不据此固化 QoS 设置。进程 CPU 时间包含所有线程累计 CPU 消耗，不等同于主线程占用或 CPU 核心利用率。缺页计数涵盖计时与后续模块诊断；本轮几乎没有硬缺页，不能把约 7–8 ms 退化直接归因于交换文件。探测时系统 swap 使用约 1.87 GiB，`pmset -g therm` 未记录热或性能警告；这并不能排除其他系统干扰。

## 可复现入口

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
.venv/bin/python experiments/graph_dispatch_20261105/driver.py
.venv/bin/python experiments/graph_dispatch_20261105/qos_driver.py
.venv/bin/python experiments/graph_dispatch_20261105/sample_driver.py
```

重跑需修改为新的结果目录/标签。`sample_driver.py` 只采样自己创建的推理子进程，保留 PID 校验；采样期间的数据标注 diagnostic_only，不作为无插桩性能收益。

## CPU 采样与提交简化候选

`sample_native.sample.txt`、`sample_partial_bound.sample.txt` 保存各 3 秒的 CPU 调用栈。可见 MLX eval、Metal command encoding、BufferCache 管理、Pillow resize，以及许多睡眠等待线程。**等待栈采样计数不能当作消耗 CPU 的时间，更不能当作 GPU 时间**。采样期间原生进程也表现为较高 CPU 时间，因此不能将采样值直接与无插桩计时比较，也尚未唯一确定快/慢态根因。

根据现有代码，`workspace_denoise` 每步会 `async_eval(x,cache)`：除动作外，再提交 25 层 K/V，即 50 个 KV 数组作为求值根节点。新增 `submit.py` 仅调整提交方式，不改编译单步内核、权重、时间步长或更新公式：

- xonly：每一步只 `async_eval(x)`；动作计算依赖该步各层 KV。
- every2 / every5：同样只提交动作，但分别每两步/五步提交，最后一步始终提交；步数仍为动态参数。

这些是异步提交频率实验，不是减少去噪步数，也不是之前两步/五步联合编译。需要检查中间 KV 作为后续步骤输入时的依赖正确性。

`submit_full.py` 在同一个模型实例内随机切换原版和三种候选，避免多实例/权重驻留差异；每种预热 5 次、正式 120 次，并执行 51 案例动作与归一化输出回归。所有配置都重新计算真实三图、文本和状态，无跨请求缓存。

## 提交策略实测及最终决策

同一实例内的 120 组随机顺序配对：

|配置|端到端平均 ms|节省 ms|节省 bootstrap 95% 区间 ms|
|---|---:|---:|---|
|原版，动作+KV 每步提交|198.210862|0|—|
|xonly|197.756372|0.454491|[0.393749,0.515369]|
|every2|197.720589|0.490273|[0.435017,0.545739]|
|every5|197.670696|0.540166|[0.486110,0.595123]|

上述区间未校正时序相关性和多重比较，不能作为独立部署收益。随后 A-B-C-C-B-A 六个独立进程，每进程预热 10 次、正式 100 次、另 3 次同步模块剖析；A=原版、B=xonly、C=every5：

|配置|两次进程均值 ms|合并平均 ms|合并 P95 ms|每次进程 CPU 时间 ms|
|---|---|---:|---:|---:|
|原版|197.692 / 200.023|198.857681|201.320115|38.347690|
|xonly|206.582 / 208.116|207.348603|210.198025|100.596500|
|every5|209.904 / 210.542|210.223290|212.310677|101.647320|

独立复核 xonly 慢约 8.49 ms，every5 慢约 11.37 ms，**均不固化**。QoS、command-buffer 及编译封装筛选也没有足够稳定证据改变正式配置。当前正式 speed 在本轮独立复核约 **198.86 ms**，源码与上一轮完全一致。

本轮仍未解释同实例配对与独立进程方向相反的全部原因。证据表明 CPU 执行状态/开销是重要关联因素，不能直接断言 BN128 或 root 数量本身必然退化，也不能将原生入口慢态全部称为 GPU 算子变慢。CPU 采样不是 GPU timeline，尚不足以区分 GPU 空闲、提交阻塞和不同 CPU 调度状态；不据此引入新的默认线程/系统策略。

## 正确性与归档

- 7 个编译/阈值进程、6 个 QoS 进程、2 个采样进程、6 个提交复核进程，标准动作与归一化动作均逐值一致。
- 三种提交策略各通过 51 案例动作/归一化输出一致性检查。
- xonly/every5 各通过 4 组动态条件，共 8 组：39 tokens/horizon1/1步、39 tokens/horizon8/3步、189 tokens/horizon16/10步、39 tokens/horizon64/11步，全部逐值一致，覆盖最后不足一组的情况。
- 比较基准为当前 speed；原有量化误差仍在，未验证机器人闭环成功率。
- 所有原始计时保留，没有为了获得更快数字删除慢态样本。插桩采样单列，不与正常计时混用。
- `audit.json` 汇总一致性、合并统计和 production_unchanged；正式 Python 与 Metal 源文件逐文件比较上一轮快照无改动。模型权重和配置未修改。

补充命令（重跑须使用新目录/标签）：

```bash
.venv/bin/python experiments/graph_dispatch_20261105/submit_full.py
.venv/bin/python experiments/graph_dispatch_20261105/submit_abba.py native a1
.venv/bin/python experiments/graph_dispatch_20261105/submit_abba.py xonly b1
.venv/bin/python experiments/graph_dispatch_20261105/submit_abba.py every5 c1
.venv/bin/python experiments/graph_dispatch_20261105/submit_abba.py every5 c2
.venv/bin/python experiments/graph_dispatch_20261105/submit_abba.py xonly b2
.venv/bin/python experiments/graph_dispatch_20261105/submit_abba.py native a2
.venv/bin/python experiments/graph_dispatch_20261105/submit_dynamic.py
```

代码、样本、输出、CPU 栈、设备信息及源码哈希保存在 `/Users/qklj/work/cx/ImageWAM_MLX/records/graph_dispatch_20261105/`，归档为 `records/graph_dispatch_20261105_review.tar.gz`，本地镜像校验 SHA256 后解包。后续应借助无额外同步的 GPU timeline 进一步分离主机提交与 GPU 执行，再决定是否重试已有局部候选。
