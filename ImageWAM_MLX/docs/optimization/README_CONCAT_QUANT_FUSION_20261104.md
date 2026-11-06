# 视觉拼接与 INT8 激活量化融合：2026-11-04

已固化到 --profile speed：将每个视觉 single block 的 attention/MLP 拼接合并进 INT8 量化读取，不再生成完整拼接张量。固定 200 组共享权重配对确认：200.196 → 199.253 ms，节省 0.943 ms（0.47%）；95% bootstrap 区间 [0.881, 1.008] ms。这是一项约 1 ms 的小幅改善，不能把后台负载变化带来的更大下降算入优化。

默认 accuracy 不变；保留 --no-fused-concat 和 Python API fused_concat=False 回退。仅在当前已验证的 speed/prepacked INT8 分支启用。磁盘权重和原配置不修改。

## 实现与局部证据

旧路径先计算 gated(mlp)，与 attention 拼成 BF16 张量，再量化为 INT8。新路径保留原 gated 算术及 BF16 舍入，在同一个量化 kernel 内按位置读取两个独立输入；行最大值仍覆盖两段全部通道，输出仍采用相同块布局。后续 INT8 矩阵乘、INT32 累加、BF16 输出完全沿用原实现。不是缓存输入，也不改变模型权重精度。

改动位于 imagewam_mlx/split_quant_visual.py、runtime.py 和两个 CLI。生产代码不读取实验源码、不使用 AST/exec。源内核与许可沿用 staged_int8_upstream.json / LICENSE.mlx。文本长度不固定，不截断文本；每次按实际行数分配输出，模型通道维度保持其原结构。

真实投影尺寸的合成输入专项：M=328，attention=3072，原 MLP=18432。500 组同步墙钟计时，旧拼接+gate+量化 0.396414 ms，新路径 0.363341 ms，节省 0.033073 ms；量化输出/尺度/指数逐值一致。19 层累计约 0.6 ms 量级，只作机制诊断，不直接等同端到端收益；这不是 GPU counter 测量。

同时测试了把投影后的缩放/残差相加并入 INT8 GEMM 输出内核，保留 BF16 中间舍入。单独残差候选通过 51 个样本，但完整性能证据不足，未默认采用；组合候选也未固化。

## 测试条件与证据边界

Apple M5 Pro（15 CPU / 16 GPU），24 GiB 统一内存，macOS 27.2，Python 3.12.13，MLX 0.32.4，buffer256。三张已解码 RGB + state + 在线文本 → CPU 动作，10 步、默认 horizon16。不含加载、文件读取和结果写盘，不缓存跨请求特征。

最初两项 160 组配对均受明显抖动影响：拼接融合节省 2.694 ms，区间 [-3.081,8.547]；残差融合节省 1.237 ms，区间 [-4.431,6.819]。这些数据本身不足以确认收益。

随后进行了六进程比较与四进程确认。前者每进程 50 次、后者每进程 100 次，均预热 10 次；基线自身存在跨进程漂移，因此未将合并均值差作为净优化量。最终采用预先固定 200 组、不同随机顺序的共享权重配对确认，并与局部算子证据相互核对。全部原始结果保留，未删除慢样本。

| 四进程确认 | 平均 ms |
|---|---:|
| 0_baseline | 198.995 |
| 1_split | 200.565 |
| 2_split | 200.819 |
| 3_baseline | 208.324 |

正式代码部署后再次直接运行 scripts/benchmark.py（10 次预热、100 次稳态）：

| 指标 | 最新正式 speed |
|---|---:|
| 稳态平均 ms | 206.797 |
| P50 / P95 ms | 207.051 / 208.562 |
| 最小 / 最大 ms | 203.054 / 208.745 |
| 模型加载 s | 3.143 |
| 首次前向 ms（含首次编译等） | 856.306 |

该部署后绝对耗时与配对实验时段不同，不能直接相减归因。模块单独同步诊断各 5 次，改变了调度，仅供定位：

| 模块 | 平均 ms |
|---|---:|
| text_encoder | 28.636 |
| image_encoder | 27.997 |
| visual_prefill | 82.970 |
| action_denoise | 62.559 |

## 精度回归

正式入口对 51 个代表样本的 action/normalized 与此前保存的 speed 输出逐值一致。accuracy 标准样本与历史精度参考逐值一致，关闭 staged 的回退路径验证通过。实际 infer CLI 与 benchmark 输出逐值一致。
补充关闭/开启本次融合的动态边界比较：39 tokens / horizon1, 189 tokens / horizon16, 39 tokens / horizon64，均逐值一致。已测无新增数值误差，不代表 speed 与 PyTorch/accuracy 等精度；原量化误差仍存在，未进行机器人任务成功率测试。

## 启动与回退

工作目录 /Users/qklj/work/cx/ImageWAM_MLX。最新正式实测平均 206.797 ms：

```bash
.venv/bin/python scripts/benchmark.py --profile speed --reference examples/robotwin_ep0_frame0 --output records/fused_concat_next --warmup 10 --iterations 100 --profile-iterations 5
```

关闭本次融合；配对对照平均 200.196 ms（对应开启 199.253 ms）：

```bash
.venv/bin/python scripts/benchmark.py --profile speed --no-fused-concat --reference examples/robotwin_ep0_frame0 --output records/unfused_concat_next --warmup 10 --iterations 100 --profile-iterations 5
```

```bash
.venv/bin/python scripts/infer.py --profile speed --inputs examples/robotwin_ep0_frame0/inputs.npz --prompt-file examples/robotwin_ep0_frame0/prompt.txt --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy --output records/fused_concat_infer_next --warmup 1
```

输出目录需使用新名称。--no-prepacked-visual 或 --no-staged-visual 也会退出本次融合所依赖的分支。默认不指定 profile 仍是 accuracy。实验脚本已显式设置 fused_concat=False 构造旧基线，避免新默认污染后续复现。

## 文件与后续

records/concat_quant_20261104、records/visual_epilogue_20261104 保存初测；records/fusion_confirm_20261104 保存 before 旧源码、paired_repeat、micro、两轮独立进程、正式 51 例/accuracy 回归、动态输入、final_entry 和 cli_infer。含原始输入/输出、计时和源码快照。trace.npz 保留远端，审阅归档省略。实验源码为 experiments_runtime_tuning_20261104/fusion_*.py、concat_quant_test.py、visual_epilogue_test.py。
下一步应先量化单个归一化/调制到量化之间的开销，再决定是否进一步融合；LayerNorm 的归约与 BF16 舍入需要单独精度验证。未验证的融合不默认启用。
