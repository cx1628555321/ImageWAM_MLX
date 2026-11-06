# 正式 speed 的 200 ms 核验与 INT8 共享内存布局实验

结论：历史约 200 ms 确实来自正式 --profile speed，而非不能部署的候选。当前正式实现未改变，但本轮负载下不能确认稳态平均仍为 200 ms。不能用最快一次代替平均，也不能把不同时间段的差异归因于代码优化。

## 正式入口重新测量

M5 Pro（15 CPU / 16 GPU）、24 GiB 统一内存，macOS 27.2，MLX 0.32.4。正式预展开 INT8 speed，buffer256，三张已解码图片 + state + 在线文本，10 步、horizon16；计时至 CPU 动作输出，无跨请求特征缓存，不包括文件读取、模型加载或输出写盘。10 次预热后统计 100 次稳态。

| 指标 | 本轮正式 speed |
|---|---:|
| 平均 | 222.480 ms |
| P50 | 217.502 ms |
| P95 | 256.737 ms |
| 最小 / 最大 | 200.208 / 258.060 ms |
| 加载 | 3.601 s |
| 首次前向（含首次编译等） | 4358.745 ms |

同期观察到 mediaanalysisd 约占一个 CPU 核，并有 VideoToolbox 编解码服务活动。过程快照保存在本轮记录内；没有暂停其他服务，也没有 GPU 时间线证明其与每次延迟尖峰的因果关系。历史正式均值 200.071 ms 和 203.354 ms 的记录保留有效，但不是当前负载下的保证值。

## 继续优化：保留块内复用，改变权重共享内存行跨度

原内核 INT8 权重 tile 的 K=64，threadgroup 行跨度为 80 字节。候选只将跨度改为 64、96、128，测试 bank conflict/对齐/占用率的性能影响；数学运算、权重量化、激活精度和输入处理均不变。没有移除 threadgroup 权重复用。

每个候选 100 组随机顺序配对：

| 路径 | 平均 ms | 相对基线节省 ms | bootstrap 95% 区间 ms |
|---|---:|---:|---|
| baseline | 225.581 | 0.000 | [0.000, 0.000] |
| pad64 | 222.575 | 3.006 | [-3.092, 9.182] |
| pad96 | 225.563 | 0.018 | [-6.282, 6.214] |
| pad128 | 225.149 | 0.432 | [-6.559, 7.334] |

三个候选收益区间都跨零。进一步对配对均值最好的 64 字节候选进行单模型独立进程 ABBA，顺序 80/64/64/80，各 10 次预热、60 次稳态：

| 进程 | 平均 ms | P95 ms |
|---|---:|---:|
| 0_80 | 225.658 | 260.480 |
| 1_64 | 231.106 | 268.290 |
| 2_64 | 228.066 | 264.975 |
| 3_80 | 223.831 | 259.625 |

合并平均：当前 80 字节路径 224.745 ms，64 字节候选 229.586 ms，候选差值 +4.841 ms。未确认稳定收益，不更改正式默认。独立进程使用相同 benchmark 计时逻辑，实验入口仅替换候选 kernel；这是实验路径，不是正式默认发生变化。

三个候选都通过 51 个代表样本的 action/normalized 逐值一致性验证；独立进程标准样本也逐值一致。此处仅说明本轮未增加已测数值差异，原 speed 相对 accuracy/PyTorch 的量化误差依然存在。本轮未重新执行完整 accuracy，因为其实现没有改变；SHA256 已确认 runtime.py、prepacked_visual.py、infer.py、benchmark.py 与上轮正式版本相同。

## 复现与记录

工作目录 /Users/qklj/work/cx/ImageWAM_MLX。正式路径重测（本轮平均 222.480 ms）：

```bash
.venv/bin/python scripts/benchmark.py --profile speed --reference examples/robotwin_ep0_frame0 --output records/speed_verify_next --warmup 10 --iterations 100 --profile-iterations 5
```

64 字节实验候选，仅供复查：

```bash
.venv/bin/python experiments_runtime_tuning_20261104/padding_bench_entry.py 64 --profile speed --reference examples/robotwin_ep0_frame0 --output records/padding64_next --warmup 10 --iterations 60 --profile-iterations 3
```

输出目录应使用新名称。records/speed_verify_followup_20261104 保存正式 100 次结果、输入、输出、模块诊断与源码快照；records/int8_padding_followup_20261104 保存三个候选、51 例输出、进程快照和 ABBA。实验脚本为 experiments_runtime_tuning_20261104/padding_*.py。大型 trace.npz 保留远端，审阅归档不含该文件。

下一步优先量化相邻归一化/激活处理与 INT8 量化的单算子成本，再选择融合边界；在当前明显抖动的环境下，小于数毫秒的端到端改善需要多轮单模型确认，不能仅按最低耗时推广。
