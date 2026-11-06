# 门控激活与 INT8 量化融合：2026-11-04

已将 BF16 SiLU 函数表、门控乘法与 INT8 量化融合接入 speed 默认路径。收益很小，两轮各 200 组随机顺序配对均节省约 0.12 ms（约 0.06%），不应解释为明显的整体加速。保留 --no-fused-gate 回退，accuracy 不变。

## 同期完整模型确认

| 实验 | 原 speed ms | gate 融合 ms | 节省 ms | 配对 bootstrap 95% 区间 ms |
|---|---:|---:|---:|---|
| 第一轮，200 组 | 198.275707 | 198.147051 | 0.128656 | [0.087670, 0.170456] |
| 独立复测，200 组 | 198.557893 | 198.437472 | 0.120421 | [0.068345, 0.169385] |

测试共享冻结模型权重、随机交错调用，每次重新执行在线编码和推理，不缓存请求特征。第二轮使用不同随机顺序；区间只反映该次实验条件内的采样。亚毫秒级收益容易被系统负载掩盖，不承诺每次请求都更快。

## 实现与数值边界

原 speed 已融合拼接和量化，但仍先生成 gated(mlp) 的 BF16 中间张量。本轮将其省去：一次性用原 MLX 运算生成覆盖 65536 个 BF16 编码的 SiLU 表（BF16，128 KiB），量化 kernel 直接读取原始 MLP 的两半，查表后完成门控乘法、行尺度归约和 INT8 转换。保留必要 BF16 舍入。

采用每行 1024 线程，按连续通道分配读取；使用输入 strides，避免为原始 MLP 切片额外做连续化拷贝。行数和文本长度动态变化，不固定或截断文本。函数表是静态函数值，不是实际输入或文本特征缓存。其索引域覆盖全部 BF16 编码；实际数值回归范围见下文，不宣称穷举验证所有模型输入。

量化权重和 GEMM 不变：预展开 INT8 权重、INT8 激活、INT32 累加、BF16 输出；其他模型模块的精度策略也不变。只增加 128 KiB 常驻函数表，不修改磁盘模型文件。新增 gate_quant_visual.py，无运行时 AST/exec 或实验脚本依赖。

## 局部候选筛选

真实投影维度的合成输入 M=328、attention=3072、原始 MLP=18432。每组 500 次随机顺序同步墙钟计时，非 GPU counter。第一组为连续输入，后两组刻意使用非连续切片；必须各自与同期基线比较。

| 实验/候选 | 同期基线 ms | 候选 ms | 量化结果检查 |
|---|---:|---:|---|
| 原分块布局/exp | 0.285775 | 0.350507 | 出现差异，拒绝 |
| 原分块布局/lut | 0.285775 | 0.360919 | 逐值一致 |
| sigmoid 表/连续线程访存/v128 | 0.299004 | 0.421882 | 逐值一致 |
| sigmoid 表/连续线程访存/v256 | 0.299004 | 0.350341 | 逐值一致 |
| sigmoid 表/连续线程访存/v512 | 0.299004 | 0.297560 | 逐值一致 |
| SiLU 表/连续线程访存/v256 | 0.308843 | 0.352871 | 逐值一致 |
| SiLU 表/连续线程访存/v512 | 0.308843 | 0.306956 | 逐值一致 |
| SiLU 表/连续线程访存/v1024 | 0.308843 | 0.301064 | 逐值一致 |

直接手写浮点 sigmoid 公式产生了 27382 个 INT8 编码差异和 13 个行尺度差异，且更慢，因此未进入完整模型测试。仅 sigmoid 查表虽恢复已测一致性，但多数布局仍慢。最终只有完整 SiLU 表 + 1024 线程显示约 0.008 ms 的局部改善，因而进入完整模型验证；不能将局部计时按层数简单相加当作端到端收益。

## 正式入口与精度回归

设备 Apple M5 Pro（15 CPU / 16 GPU）、24 GiB 统一内存，macOS 27.2、Python 3.12.13、MLX 0.32.4，buffer256。三张已解码 RGB + state + 在线文本到 CPU 动作；10 步、默认 horizon16；不含模型加载、输入文件读取与结果写盘。

正式部署后，10 次预热、100 次稳态：

| 指标 | 最新默认 speed |
|---|---:|
| 平均 ms | 198.123 |
| P50 / P95 ms | 197.937 / 200.368 |
| 最小 / 最大 ms | 197.268 / 201.206 |
| 加载 s | 3.116 |
| 首次前向 ms（含首次编译等） | 836.782 |

该绝对值与配对测试时段不同，不能直接相减归因。单独同步的模块诊断各 5 次，不能相加代替未插桩端到端：

| 模块 | 平均 ms |
|---|---:|
| text_encoder | 27.805 |
| image_encoder | 27.878 |
| visual_prefill | 82.590 |
| action_denoise | 62.498 |

两轮候选各 51 个代表样本、正式默认再次 51 个样本，action 与 normalized 均与原 speed 逐值一致。accuracy 标准样本与历史参考逐值一致，原 staged 关闭路径验证通过；实际 infer CLI 与 benchmark 结果一致。
关闭/启用本轮融合的动态边界比较：39 tokens / horizon1, 189 tokens / horizon16, 39 tokens / horizon64，逐值一致。没有新增已测数值误差，不代表 speed 与 accuracy/PyTorch 等精度；原量化误差仍存在，未验证机器人任务成功率。

## 启动与回退

在 /Users/qklj/work/cx/ImageWAM_MLX 运行，最新单独测试平均 198.123 ms：

```bash
.venv/bin/python scripts/benchmark.py --profile speed --reference examples/robotwin_ep0_frame0 --output records/gate_quant_next --warmup 10 --iterations 100 --profile-iterations 5
```

关闭本轮门控融合，保留上一轮拼接量化融合：

```bash
.venv/bin/python scripts/benchmark.py --profile speed --no-fused-gate --reference examples/robotwin_ep0_frame0 --output records/gate_quant_old_next --warmup 10 --iterations 100 --profile-iterations 5
```

```bash
.venv/bin/python scripts/infer.py --profile speed --inputs examples/robotwin_ep0_frame0/inputs.npz --prompt-file examples/robotwin_ep0_frame0/prompt.txt --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy --output records/gate_quant_infer_next --warmup 1
```

Python API 用 fused_gate=False 回退。--no-fused-concat、--no-prepacked-visual 或 --no-staged-visual 也会退出本融合依赖的路径。默认不指定 profile 仍是 accuracy。输出目录需使用新名称。

## 记录

records/gate_quant_20261104 保存 micro/vector/silu 筛选、full/repeat 各 200 组及 51 例、before 原入口、final_validation、dynamic、final_entry、cli_infer。输入、输出、计时及源码快照保留；大型 trace.npz 保留远端但不进入本地审阅归档。实验在 experiments_runtime_tuning_20261104/gate_*.py；full/repeat 已显式 fused_gate=False 构造旧基线。
后续优先考虑更高耗时的归一化/调制到量化边界；本轮说明继续压缩门控激活的收益已经很小，应以模块占比指导后续投入。
