# ImageWAM MLX 后续优化分析（2026-11-03）

本轮检查当前默认实现并增加内存/延迟实测，没有修改默认推理路径。下面是待验证的优化方向，不是已获得的加速。

## 本轮测量

环境：M5 Pro，24 GiB 统一内存，MLX 0.32.4。当前默认 compact + fused VAE + overlap；真实 RoboTwin 三图、原文本、固定输入噪声、16 帧动作、10 步去噪。预热 8 次，连续测量 50 次，每次前后同步，包含 CPU 输入处理及输出转换，不含模型加载与磁盘读取样本。没有改变 wired limit 或系统设置。

| 指标 | 结果 |
|---|---:|
| 模型构造/加载（非冷磁盘保证） | 2.618 s |
| 稳态端到端均值 | 286.451 ms |
| 中位数 | 281.353 ms |
| P95 | 330.217 ms |
| 最大值 | 337.917 ms |
| 稳态 MLX active | 14.050 GiB |
| 稳态 MLX allocator cache | 2.502 GiB |
| MLX peak active | 15.564 GiB |
| active + cache | 16.552 GiB |

上述内存不是进程 RSS；peak active 不能与稳态 cache 相加当作同时发生的峰值。设备推荐工作集为 19,069,665,280 字节。

稳态前后：进程主缺页 204→204；系统 Swapins/Swapouts、Compressions、Pageouts 均无增长；Pageins +51、Decompressions +522。系统已有约 1322 MiB swap 使用量，但未增加。系统计数包括其他进程，不能据此归因具体 GPU 操作。当前证据不支持“稳态抖动由换页造成”。加载阶段系统压缩计数显著增加，值得单独研究加载瞬态，但不等于推理阶段持续换页。

运行命令（脚本使用固定记录目录，重复执行前需修改为新的目录；不会覆盖已有结果）：

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
.venv/bin/python scripts/audit_memory_latency.py
```

原始 50 次延迟、内存快照：`records/memory_latency_audit_20261103/results.json`；最终动作：同目录 `action.npy`。本次仅检查输出有限值，未新增完整精度回归；当前默认实现的完整回归见 `README_REMAINING_ADAPTATION_20261103.md`。

## 推荐的后续实验顺序

### 1. 动作分支小 M 矩阵乘法与权重布局

动作分支默认 M=16，重复 10 步，权重约 1.196 GiB。实际矩阵包括 QKV `[9216,1024]`、single linear1 `[17408,1024]`、linear2 `[1024,7168]`。这与大批量 GEMM 的适用条件不同，下一步应测真实形状和真实中间激活，比较原 `x @ W.T` 与预先连续化的转置权重路径，再考虑专用 kernel。

先仅处理少量热点权重；测 M=1/8/16/32/64，分别记录预处理开销、内存增量、单层耗时和全模型耗时。`.T` 本身不等于每次完整复制，不能把转置视图误判为瓶颈。权重体积乘迭代次数也不代表实测 DRAM 流量。不同 GEMM 路径可能改变归约顺序，必须做逐层和最终动作精度验证；不直接启用量化或改变 BF16 舍入。

### 2. 加载阶段峰值与首次执行

加载时 MLX active 已达 14.05 GiB，系统压缩明显增长。应细分文件读取、权重求值、VAE 布局转换、tokenizer 初始化，以及首次 GPU 执行，查找原始映射与转换后张量的重叠生命周期。优先缩短临时张量存活时间；仅在证据支持时测试分阶段加载/有限驻留预算。

MLX 提供进程级 `set_wired_limit`，必须遵守设备推荐上限并恢复原设置；当前无稳态换页证据，故它不是首选的稳态加速方案。参考：[官方 API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html)。

### 3. 动态形状的首次调用开销

单独测同长度重复、31–79 token 实际样本长度切换以及 193 token 扩展用例的首次/再次调用。可预热常见实际形状，未知长度继续正常编译，禁止截断或固定文本长度。评估图导出/导入能否减少 Python tracing；导出图不应当作持久化 Metal 机器码，仍须实测冷启动。参考：[MLX Exporting Functions](https://ml-explore.github.io/mlx/build/html/usage/export.html)。

这一方向主要改善启动与长度切换延迟，不应拿来宣称固定输入稳态提升。

## 暂不重复的方向及验收

已有记录显示跨步大图、普通 split-KV、自写 split attention、部分 RMS/RoPE 融合未稳定获益；QKV 合并存在输出变化及额外内存；Qwen RoPE 与 VAE group inverse 保留可选开关，不升级默认。不能因为“融合更多”就推断更快。

新候选先做局部测试，通过后使用交替配对端到端测试和重复运行，不删除尾延迟；报告均值、P95、差值置信区间及内存。精度沿用 51 样本、输入扰动与动态长度、162 数组对照，并记录首次出现差异的中间层。保持三张图、原文本、原始去噪步数与输出语义。这里的数值一致性不代表已验证机器人任务成功率。
