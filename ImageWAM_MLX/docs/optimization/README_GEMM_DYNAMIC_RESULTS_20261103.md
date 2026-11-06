# 动作 GEMM 与动态文本预热适配结果

2026-11-03；远程根目录 `/Users/qklj/work/cx/ImageWAM_MLX`。按要求只推进矩阵乘法与动态长度方向，未修改模型加载逻辑。

## 结论与代码

新增显式 `ImageWAM.warmup(samples, repeats=1)`，实现位于 `imagewam_mlx/warmup.py`，建议服务开始接收请求之前用一个真实样本预热。默认 `predict` 数学路径不变。预热不保存图像/文本特征或动作结果，后续未知文本长度正常执行，不做截断、填充或长度限制特化。运行时编译缓存仍由 MLX 管理；此接口不是持久化编译二进制缓存，不保证消除所有新形状开销。

连续转置权重方案精度不满足要求且更慢，未适配到主推理路径。二维 GEMM 调用方案数值一致，但未获得显著局部收益，也未引入主路径。

## 环境与协议

M5 Pro（15 CPU / 16 GPU）、24 GiB 统一内存，macOS 27.2、Python 3.12.13、MLX 0.32.4。默认 compact + fused VAE + overlap。真实 RoboTwin 三图、固定样本状态与初始噪声，动作长度 16，去噪 10 步。时间包含输入处理和 CPU 输出转换，计时前后 GPU 同步，不包含模型构造或磁盘样本读取。

动态文本组使用 48 个实际样本长度（31–79 token，缺 78）、193 token 扩展文本、未预热的 158 token 文本，再回到 31 token，共 51 个案例，每例测三次。三组独立进程串行执行，不是随机交替配对试验；后续混合长度均值不能作为严格的稳态加速比较。

## 动态文本结果

| 模式 | 请求前预热成本 | 首个正式请求 | 后续长度首次请求均值 | 各长度重复请求均值 |
|---|---:|---:|---:|---:|
| 未预热 | 0 | 1276.086 ms | 310.279 ms | 296.485 ms |
| 全部 49 个文本预热 | 16040.693 ms | 284.791 ms | 312.030 ms | 300.379 ms |
| 仅首个真实样本预热 | 1244.312 ms | 269.905 ms | 316.526 ms | 300.301 ms |

首个样本预热把首次执行开销转移到接收请求之前，**不减少启动加首请求的总工作量，也没有证明稳态计算加速**。遍历全部文本长度没有显示持续收益，不推荐作为默认启动策略。未知 158 token 长度正常运行；单样本预热组该请求为 371.459 ms，说明未承诺消除未知形状首次开销。

每种预热方案与未预热组比较 51 案例 × 2 数组（action、normalized），各 102 个数组逐字节一致；两种方案合计 204 次数组对比。所有案例组内重复输出也逐字节一致。当前比较固定图像、状态与噪声，改变文本；不替代机器人任务成功率评测。

推荐调用方式：

```python
from imagewam_mlx import ImageWAM

model = ImageWAM("weights/bf16", "assets")
# images/state/prompt/noise 使用正常推理的真实样本。
sample = dict(images=images, state=state, prompt=prompt, noise=noise, steps=10)
warmup_report = model.warmup([sample])
actions, normalized = model.predict(**sample)
# 后续直接传入新文本、新观测；无需重复预热或命中已知长度。
```

`warmup` 与 `predict` 不应在同一模型实例上并发调用。输入验证仍走原 `predict`；`repeats` 必须为正整数。预热后清空 trace/profile，不保留诊断输出。

## GEMM 实验

取真实首步动作激活，测试三个代表层；M=1/8/16/32/64，其中 M>16 通过重复真实激活行构造，属于局部基准，不是对应完整动作轨迹。每点预热后交替测 40 对，包含单算子同步开销，不能把局部时间相加当作端到端耗时。

默认 M=16 的连续转置权重结果：

| 层 | 原始 ms | 连续权重 ms | 中间输出最大绝对差 |
|---|---:|---:|---:|
| double 0 QKV | 0.213 | 0.250 | 0.015625 |
| single 0 linear1 | 0.289 | 0.319 | 0.0625 |
| single 0 linear2 | 0.209 | 0.252 | 0.125 |

15 个形状测试中仅 2 个输出一致。没有把它推广到全模型或声称最终动作误差等于上表；局部已不满足严格一致要求，停止该候选。

另一方案把输入展平为二维后执行原始转置权重 GEMM，再还原输出维度，无额外权重副本。15/15 局部测试输出一致。M=16 的配对“原始减候选”差值 bootstrap 95% 区间：

| 层 | 原始 ms | 二维调用 ms | 节省耗时 CI，ms |
|---|---:|---:|---:|
| double 0 QKV | 0.1962 | 0.1981 | [-0.01014, 0.00532] |
| single 0 linear1 | 0.2855 | 0.2835 | [-0.00320, 0.00775] |
| single 0 linear2 | 0.1632 | 0.1658 | [-0.00749, 0.00203] |

区间都跨 0，没有足够证据继续推广。两轮基准环境状态不同，应只比较同一行配对值。

## 启动命令与记录

以下测试输出目录必须不存在；重测请使用新目录，不覆盖已有结果。动态测试本身会测量端到端时间，GEMM 命令只测局部算子。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
.venv/bin/python scripts/test_gemm_layout.py --variant packed --output records/gemm_shapes_new
.venv/bin/python scripts/test_gemm_layout.py --variant flatten --output records/gemm_flatten_new
.venv/bin/python scripts/test_dynamic_warmup.py --mode cold --output records/dynamic_cold_new
.venv/bin/python scripts/test_dynamic_warmup.py --mode warm --output records/dynamic_warm_new
.venv/bin/python scripts/test_dynamic_warmup.py --mode first --output records/dynamic_first_new
```

本次完整记录在 `records/gemm_shapes_20261103`、`records/gemm_flatten_20261103`、`records/dynamic_{cold,warm,first}_20261103`。动态目录中的 `case_*.npz` 保存最终输出；原始逐次时间在各自 `results.json`。汇总 `records/gemm_dynamic_summary_20261103.json`，可运行 `.venv/bin/python scripts/summarize_gemm_dynamic.py` 重算本次汇总（脚本路径固定到本次目录）。代码修改前备份在 `records/gemm_dynamic_source_20261103/runtime_before.py`。

进一步改善稳态速度需要更底层的 GEMM kernel/调度证据；当前两个简单调用/布局改写均不支持直接获得收益。预热接口已完成适配，不自动在构造函数中运行，以便调用方决定何时准备真实样本与开始接收请求。
