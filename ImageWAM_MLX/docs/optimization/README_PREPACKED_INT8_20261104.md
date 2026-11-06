# 冻结权重预展开 INT8：2026-11-04

已固化到 speed 默认路径。本轮正式 ABBA 独立进程测试：旧 speed 210.446 ms → 新 speed 203.354 ms，减少 7.091 ms（3.37%）。不应把上一轮 230 ms 与本轮直接相减：后台负载与时段不同，本轮只对同期对照归因。

瓶颈：上一轮虽采用 INT8 矩阵乘，但每个 M 方向 tile 都重复将冻结 W4 权重解包，并把 scale/bias 折叠为 INT8。现在加载时用相同整数规则完成一次转换，前向直接读取 INT8 权重；保留在线激活量化、INT32 累加和 BF16 输出。这是以常驻内存换取前向时间，不是进一步降低比特数，也不会把整个模型变成全 INT8。

38 个 single_blocks.linear1/linear2 使用预展开 INT8，来源仍是先前 W4 group64；Qwen/VAE/其余视觉层和动作 W1 路径保持原逻辑。磁盘权重未重写。内核继续使用同一 MLX PR #4722 mode-2 算术，来源/许可见 staged_int8_upstream.json 和 LICENSE.mlx。

## 测试条件与结果

M5 Pro（15 CPU / 16 GPU）、24 GiB 统一内存、macOS 27.2、MLX 0.32.4、Python 3.12.13。命令缓冲阈值 256。三张已解码 RGB + state + 在线可变文本 → CPU 动作；10 步、默认 horizon16。计时不含文件读取、加载和结果写盘，无跨请求输入/特征缓存。

100 组随机顺序配对：基线 206.118 ms，只预计算整数系数 206.807 ms（慢 0.690 ms，未采用），预展开 INT8 199.730 ms（节省 6.387 ms，配对 bootstrap 95% 区间 6.316–6.463 ms）。

正式入口另做 old/new/new/old 四进程测试，各 4 次预热、40 次稳态；四块均值依次为 205.600, 198.989, 207.720, 215.292 ms。期间有整机时间漂移，前半段和后半段比较均支持提速；不声称已消除温度/后台负载影响。

| 指标 | 本轮旧 speed | 本轮新 speed |
|---|---:|---:|
| 稳态平均 ms，各 80 次 | 210.446 | 203.354 |
| 合并 P50 / P95 ms | 209.323 / 216.232 | 202.504 / 208.956 |
| 加载平均 s，各 2 次 | 3.052 | 3.112 |
| 首次平均 ms，受编译缓存影响 | 868.787 | 1039.877 |
| MLX active GiB | 9.850 | 11.071 |
| MLX peak GiB | 14.192 | 14.192 |

MLX active 增加约 1.22 GiB；该指标不等于系统进程 RSS。peak 包含模型加载阶段，本次 peak 相同不代表常驻内存没有增加。

模块单独同步诊断每路径 6 次，会改变调度；不能直接相加代替稳态端到端。

| 模块 | 旧 ms | 新 ms |
|---|---:|---:|
| text_encoder | 28.593 | 28.566 |
| image_encoder | 27.921 | 27.980 |
| visual_prefill | 90.288 | 83.939 |
| action_denoise | 62.812 | 62.466 |

## 精度与输入可变性

两个实验候选均对 51 个案例的 action/normalized 与上一版 speed 逐值一致。正式默认版本再次对全部 51 个案例与上一轮保存的结果逐值一致；accuracy 标准样本与历史精度参考逐值一致，关闭 staged 的原路径回归通过。infer CLI 与 benchmark 输出逐值一致。
补充生产路径与 --no-prepacked-visual 路径比较：39 tokens / horizon 1, 189 tokens / horizon 16, 39 tokens / horizon 64，action/normalized 均逐值一致。未固定或截断文本长度。该优化没有新增已测数值误差，但 speed 本身已有先前 W1/W4 量化误差，不能称其与 accuracy/PyTorch 等精度，未验证机器人任务成功率。

## 使用与回退

项目目录 /Users/qklj/work/cx/ImageWAM_MLX。新 speed 同期平均 203.354 ms：

```bash
.venv/bin/python scripts/benchmark.py --profile speed --reference examples/robotwin_ep0_frame0 --output records/prepacked_retest_new --warmup 4 --iterations 40 --profile-iterations 3
```

回退本轮预展开（保留上一轮 staged INT8 + buffer256），同期平均 210.446 ms：

```bash
.venv/bin/python scripts/benchmark.py --profile speed --no-prepacked-visual --reference examples/robotwin_ep0_frame0 --output records/prepacked_retest_old --warmup 4 --iterations 40 --profile-iterations 3
```

```bash
.venv/bin/python scripts/infer.py --profile speed --inputs examples/robotwin_ep0_frame0/inputs.npz --prompt-file examples/robotwin_ep0_frame0/prompt.txt --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy --output records/prepacked_infer --warmup 1
```

输出目录需新建。Python API 用 prepacked_visual=False 回退；--no-staged-visual 关闭整个 staged 路径；accuracy 仍是未指定 profile 时的默认路径。自动启用范围仍受原 M5/MLX0.32.4 策略限制。

## 记录与后续

records/staged_fold_20261104 保存前版源码、100 组原始耗时、51 样本输出、正式 ABBA 输入/输出/源码快照、精度与动态长度结果。experiments_runtime_tuning_20261104/fold_*.py 保存复现脚本。最初一次内核尺寸变量编译错误已修复，原始失败日志保留为 records/staged_fold_20261104.log。
下一步可围绕预展开 INT8 权重的分块布局、重复从 device 到 threadgroup 的复制和 activation quantize 的额外 kernel 做专项实验；需继续以端到端验证，因为单算子加速不必然带来整体收益。无需重新量化或下载权重。
