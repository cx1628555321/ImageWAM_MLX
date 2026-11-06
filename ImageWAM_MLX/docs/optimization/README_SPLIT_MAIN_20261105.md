> 后续状态：空闲窗口整网验证现已完成，见本文末尾。以下中止原因保留为首次尝试记录。

# Split-K 主循环优化进展（2026-11-05）

状态：三个候选已实现并完成真实投影微测；整网测试因并发负载中止，**未完成整网精度及性能验证，未修改默认实现**。

## 改动

1. `unroll`：将 split_qmm 内 BK=64、SK=32 的两次矩阵运算显式展开，保留求和顺序。
2. `edges`：仅跳过 k=0 的加载前屏障，以及最后寄存器累加结果写回前的屏障。每轮加载后的屏障和后续迭代复用共享内存前的屏障保留。
3. `combined`：组合两项。

不修改通用 qmm 或权重 loader，不改变 W1 权重、Split-K 分区、FP32 partial、BF16 输出舍入、动作步数和输入长度。候选代码在 `experiments/split_main_20261105/candidate.py`，正式代码未变。

## 已完成的真实投影测试

实际捕获的 `(M,N,K)=(16,1024,7168)`，显式张量参数进入 mx.compile，预热15次、200轮随机交替，各次 eval 同步。测量为主矩阵乘+现有归约的 host wall，不是纯 kernel GPU 时间。

|模式|平均 ms|相对基线变化|局部输出|
|---|---:|---:|---|
|baseline|0.196646|—|参考|
|unroll|0.182841|降低约7.02%|逐值一致|
|edges|0.199767|增加约1.59%|逐值一致|
|combined|0.183553|降低约6.66%|逐值一致|

局部输出一致仅针对捕获的真实投影，不能替代全模型精度。测试过程中发现同机另一 Pi05 GPU 模型任务，不能据此宣称稳定加速或外推端到端收益。完整原始样本：`records/split_main_20261105/micro.json`。

## 整网测试中止原因

同机发现 `pi05/backends/mlx/.venv/bin/python ... q8-subnets-20261105/trial.py ... expert-mlp` 测试进程。系统 vm.swapusage 报告 used=15211.81 MiB（约14.86 GiB），ImageWAM 整网仍在加载阶段，没有产生端到端测量记录。

为避免继续相互干扰，核验 PID 78043 的命令为本轮 `experiments/split_main_20261105/test.py` 后只终止该进程。未终止 Pi05 任务。串行命令链返回143，后续51样本脚本没有启动。

因此本轮**没有**整网速度、51样本回归或动态形状通过结论，也没有新的默认速度记录。并发证据保存在 `records/split_main_20261105/concurrent_processes.txt`。

## 无并发模型任务时继续验证

两个脚本支持新的记录目录，避免覆盖本轮局部证据。在服务器没有其他模型测试、内存状态稳定后执行：

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
export WAM_RECORD_DIR="records/split_main_resume_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python experiments/split_main_20261105/test.py && .venv/bin/python experiments/split_main_20261105/validate.py
```

test.py 包含三个候选与相同包装基线的60轮随机交替完整推理、H=1/16/64和长文本测试；validate.py 保存51用例基线及所有候选输出。若收益达到 max(2 ms,1%) 且输出一致，再安排独立进程正式入口复核。当前没有留下后台等待或自动切换任务。

## 空闲窗口续测完成

后台监控于16:47:16完成 attempt_2 并退出；成功运行期间每15秒检查未发现其他模型任务（不能排除短于采样间隔的争用）。

|模式|完整推理均值 ms|P95 ms|
|---|---:|---:|
|baseline|199.531|206.127|
|unroll|199.570|205.803|
|edges|199.475|205.361|
|combined|199.224|205.476|

60轮随机顺序交替，最快组合只节省0.307 ms（约0.15%）。局部约7%收益没有转化为足够的端到端改善，未达到 max(2 ms,1%) 门槛，不更新默认、不宣称稳定提速。

51用例×3候选，共153组 action/normalized 比较全部逐值一致；动态长度和长文本检查全部通过。结果：[summary.json](../../records/split_main_watch_20261105_163027/summary.json)。
