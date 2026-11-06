# ImageWAM MLX 量化可行性与实际支持检查

2026-11-03，M5 Pro，MLX 0.32.4。结论：框架与当前设备支持量化推理，值得单独建立混合精度分支；本轮只做精度审计和小规模 API 可执行性检查，没有转换或替换现有模型，也没有测量量化 ImageWAM 的速度/精度。

## 当前精度

`weights/bf16/action_only/manifest.json` 标记 `dtype=bfloat16`、`quantized=false`。逐一读取 safetensors 头部确认所有已保存张量均为 BF16。不是 FP16，也不是 INT8/INT4。主路径权重、主要激活及 KV 使用 BF16，RMS/GN 的统计、部分位置编码和输出转换等显式使用 FP32，故准确描述为 BF16 主导的混合精度推理。

| 权重部分 | 原始张量体积 GiB |
|---|---:|
| 视觉专家 | 6.9895 |
| 动作专家 | 1.1959 |
| Qwen 前 27 层及 embedding | 5.8004 |
| VAE encoder | 0.0641 |
| proprio | 0.0002 |
| 合计 | 14.0502 |

这是张量数据体积，不是文件系统占用、进程 RSS 或峰值 GPU 工作集。模型名称中的 4B 不包含全部辅助编码器参数。

## MLX 支持情况

官方依据：[quantize](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)、[quantized_matmul](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html)、[v0.32.4 量化层源码](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.4/python/mlx/nn/layers/quantized.py)。

| 路径 | 已安装版本支持 | 用途 |
|---|---|---|
| affine weight-only | 1/2/3/4/5/6/8-bit，group 32/64/128 | 优先评估 W8A16，再 W6A16/W4A16 |
| MXFP4 | 4-bit，group 32 | 微缩放浮点权重，非 INT4 的同一种编码 |
| MXFP8 | 8-bit，group 32 | 低位浮点权重；也支持 qqmm 动态激活量化 |
| NVFP4 | 4-bit，group 16 | 低位浮点权重；也支持 qqmm 动态激活量化 |
| nn 层 | QuantizedLinear、QuantizedEmbedding | 模块模型可通过 nn.quantize 选择层及不同配置 |

`quantized_matmul` 可直接消费压缩权重、scale/bias，并返回浮点结果。W8A16/W4A16 不意味着全部计算都成为整数，也不能把内存压缩比视为加速比。`qqmm` 会在调用时量化输入激活，支持 MXFP8/NVFP4；激活量化有额外开销及误差来源，应在 weight-only 之后单独评估。

实际设备执行检查：随机 BF16 权重 128×128，输入 `[1,M,128]`，M=1/16/39/193。上述各模式的 quantized_matmul、mx.compile 路径均执行成功并输出有限 BF16 值；qqmm 的 MXFP8/NVFP4 也通过相同长度及编译检查。affine 实测 group=64，其他模式使用各自固定 group；没有宣称测试了全部维度/设备/组合。API 可执行不等于已验证硬件吞吐、完整模型速度或动作准确性。

当前公共 nn 量化层没有可直接替换的 QuantizedConv2d。VAE 的 conv2d、GN、attention 不会因为量化线性层自动变快；将卷积转 im2col+量化 matmul 需要额外适配与临时内存，不能保证收益。KV 量化也不是权重量化自动附带的能力，当前 SDP attention 路径仍应保留浮点 KV。

## 适配当前代码

本项目使用自定义 `Weights.w` 字典、`Weights.linear` 和函数式算子，不是完整的 `mlx.nn.Module` 树。因此不能直接对 ImageWAM 调用 nn.quantize 就完成转换。建议保留当前结构，增加 QuantizedWeights.linear，加载离线打包权重及 scale/bias 后调用 mx.quantized_matmul，最后按现有 BF16 边界添加原 bias。

不要在每次推理前反量化整块权重到 BF16 再做普通 matmul，那样不能直接获得压缩权重 kernel 的收益。另建权重目录、量化 manifest、来源哈希及混合精度层配置，原模型保持完整；删除不再需要的驻留 BF16 副本后才能真正节省常驻内存。

矩阵最后一维须满足 group_size 分组约束；例如 `action_encoder.weight=[1024,14]`、proprio `[7680,14]` 不适合直接使用 group=64，第一轮保留 BF16。文本序列长度 M 不需要固定为分组长度，tokenize 不截断，仍测试全部实际文本长度。

## 推荐优先级

1. **动作专家的大线性层 W8A16，group=64**：权重约 1.20 GiB、10 步重复调用，默认小 M=16，作为相对局部的首个实验。输入/输出头、norm、时间条件等敏感小层先保留 BF16。直接影响连续动作，不能仅看权重重建误差。
2. **视觉专家 W8A16，再选择性 W6/W4**：约 6.99 GiB，历史同步 prefill 约 114.65 ms，有较大体积和时间优化潜力；实际 M 较大，BF16 GEMM 可能已高效，需实测。前缀 KV 被全部去噪步骤复用，误差会传递。
3. **Qwen W8/W6/W4**：先保持 embedding、norm 为 BF16，逐步量化 attention/MLP 投影。此处取第 9/18/27 层隐藏特征，不能仅用语言模型生成质量来判断是否通过。
4. **VAE 暂保留 BF16/FP32**：权重仅 0.064 GiB，主要问题是激活计算与访存，不是大权重驻留。最后再研究专门的低精度卷积路径。

这是待测优先级，不是已证明哪一模块量化一定更快。可先建立 W8 基准，再依据分模块敏感度选择 mixed W8/W6/W4；不建议直接全模型 INT4 或首先量化激活。

## 存储估算与精度验收

以所有最后一维可被 64 整除的二维线性权重为候选，保留 embedding、卷积和其他张量；候选 BF16 数据共约 13.261 GiB。假设 affine group=64，scale/bias 均为 BF16：

| 方案 | 估算全部权重数据 GiB |
|---|---:|
| 当前 BF16 | 14.05 |
| 全候选 W8A16 | 7.83 |
| 全候选 W6A16 | 6.18 |
| 全候选 W4A16 | 4.52 |

包含分组 scale/bias；不包含激活、KV、allocator cache、编译缓存、加载瞬态等，不是实际峰值内存预测，也不是性能预测。若保留更多敏感矩阵为 BF16，实际权重更大。

量化通常引入数值误差，与此前追求逐字节一致的优化不同。8-bit 也不能预先保证无损。应独立报告量化相对当前 MLX BF16 的增量误差，以及相对 PyTorch 的总误差；不能把既有跨框架误差直接当作新的量化容差。

验收至少包括：51 个代表案例、动态文本及更长文本、图像/状态扰动、不同 horizon/步数；中间特征、每步动作、最终动作的 max_abs/MAE/RMSE、两个夹爪及逐维误差；最后需要 RoboTwin 闭环成功率。性能用配对端到端、模块与内存测试，保持输入/噪声/步数不变，量化转换不计入稳态推理。未定义业务容差且没有闭环验证前，不应宣称量化后“精度完全没有影响”。

## 本轮记录

`records/quantization_support_20261103/results.json`：权重 dtype/形状、实际设备信息、所有 weight-only 可执行性检查。

同目录 `qqmm.json`：动态激活量化检查；`scripts/audit_quantization_support.py`：头部审计和小规模检查脚本（固定输出目录，复跑应改用新目录）。本轮未修改默认推理、任何原始模型或配置。
