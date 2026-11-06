# ImageWAM MLX 默认精度分支与极限速度量化实验

2026-11-03。开发目录：`/Users/qklj/work/cx/ImageWAM_MLX`。默认仍为此前已验证的 BF16 精度路径；速度路径是允许数值退化的独立配置，不替换原权重。当前目录没有 Git 仓库，这里的分支指运行配置，不是 Git branch。原代码快照：`records/accuracy_frozen_20261103/`。

## 启动方式

在 M5 Pro 执行；两个入口仅需改变 `--profile`。不传该参数等价于 `--profile accuracy`。输出目录要求不存在，重复运行请更换目录名。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
.venv/bin/python scripts/infer.py \
  --profile accuracy \
  --inputs examples/robotwin_ep0_frame0/inputs.npz \
  --prompt-file examples/robotwin_ep0_frame0/prompt.txt \
  --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy \
  --output records/my_accuracy_run
```

改成 `--profile speed`、输出改为 `records/my_speed_run` 即启用速度配置。`run.json` 保存实际配置、量化模式、加载时间、首次推理、预热和请求耗时；`action.npy/csv`、`normalized.npy`、`initial_noise.npy` 保存输出与噪声。默认预热 1 次；首次推理并未隐藏，单独记录在 `first_inference_ms`。

## 测试环境与口径

M5 Pro，15 核 CPU、16 核 GPU、24 GiB 统一内存（没有独立显存），MLX 0.32.4、Python 3.12.13、macOS 27.2。实际硬件/系统原始输出在正式记录的 `hardware.txt` / `os.txt`，各 benchmark 包含 device_info、源码快照与 SHA256。

按 accuracy → speed → speed → accuracy 顺序启动独立进程；每次 8 次预热、100 次稳态、10 次额外同步模块 profile。每种配置共 200 次稳态。端到端为已解码的三张 RGB 图、状态和动态文本到 CPU 最终反归一化动作；不含加载、磁盘读取/写输出或视频解码。保持 16×14 动作、10 步去噪、不截断文本、不复用旧输入特征。首次含图编译，不是清空系统缓存后的严格冷启动。

| 指标 | 默认 accuracy | speed |
|---|---:|---:|
| 稳态平均 ms | 291.082 | 277.018 |
| 稳态中位数 ms | 284.488 | 277.548 |
| P95 ms | 322.286 | 297.941 |
| 加载/加载并打包 s，两次 | 2.613 / 2.609 | 2.836 / 2.800 |
| 首次推理 ms，两次 | 1123.888 / 1172.152 | 1111.726 / 1048.376 |

本轮稳态延迟降低 **4.83%**，吞吐对应 **1.051×**。这是当前被测试组合的结果，不代表所有可能内核/设备的全局最优。机器存在时间波动，保留全部逐次数据；候选筛选值不能与另一次历史基线直接相减归因。

| 同步模块 profile，ms | accuracy | speed |
|---|---:|---:|
| tokenization | 0.310 | 0.153 |
| text_encoder | 43.429 | 28.907 |
| preprocess | 4.171 | 2.522 |
| image_encoder | 43.539 | 29.371 |
| visual_prefill | 124.450 | 130.759 |
| action_denoise | 87.806 | 89.080 |
| postprocess | 0.220 | 0.157 |

同步 profile 每个模块强制完成，改变流水重叠，不能直接将其求和等同于正常端到端。主要剩余瓶颈是视觉 prefill 和 10 步动作去噪。VAE 快速融合是明确收益来源；动作 W1 相对其他混合候选的微小优势存在波动，不能将全部组合收益归因于量化。

## 最终速度配置

- 文本 Qwen 和视觉专家的大矩阵：BF16；整体低位量化实测更慢。
- 动作专家 65 个符合条件的线性层：affine W1、group_size=64，激活保持 BF16。小输入/输出头和 norm 保留原类型。
- VAE：BF16 卷积，GroupNorm 用分组布局上的原生 LayerNorm，整体编译；允许改变原 FP32 统计/舍入边界。
- 归一化统计、位置编码及输出转换中需要的 FP32 保留；KV 和注意力保持浮点。
- 量化仅在模型初始化时执行一次，不写入或覆盖原模型；转换时间计入加载，前向直接读取 packed weight + scale/bias。
- 不默认启用 QQMM 激活量化、Qwen 合并投影、量化行切块或视觉条件层量化。实验配置保留在 Python 接口和扫描脚本中。

## 算子支持与反量化审计

实际类型探测覆盖 27 类操作、135 个输入类型组合（两个 audit 记录合并）；量化 API 的位宽、分组与动态 M 检查另存 `records/quantization_support_20261103/`。成功执行不等价于正确的低位量化语义。

| 算子族 | 已验证可用路径 | 量化决策 |
|---|---|---|
| 普通 matmul、conv2d、原生 LayerNorm、SDPA | BF16 / FP16 / FP32；整数路径拒绝 | 不把 packed 整数直接传入 |
| quantized_matmul | packed affine 1/2/3/4/5/6/8-bit；MXFP4/8、NVFP4，输出浮点 | 动作专家采用 W1A16；不是全图 INT1 |
| QQMM | API 可运行，但当前 Metal 后端对激活 quantize-dequantize | 不进入最终速度路径 |
| RMS/GN、均值、rsqrt、SiLU、门控、sin/cos、幂 | 浮点计算；部分整数输入会晋升或丢失数值意义 | 保留浮点与必要 FP32 统计 |
| reshape/transpose/split/concat/gather/pad | 保持张量类型或索引 | 不要求反量化；索引仍是整数 |
| RoPE 自定义 Metal、残差、Euler 更新 | 生产路径 BF16/FP32 | 保持浮点，不把整数当成量化激活 |

因此不存在把当前整图统一切成 INT1/INT4 的直接可用路径。最低公共浮点主路径是 16 位；权重可以独立低位打包。整数 softmax 在小测试中未报错，但不能据此将整数结果视为正确概率或量化 attention。

源码依据：[MLX v0.32.4 Metal quantized.cpp](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.4/mlx/backend/metal/quantized.cpp)。`QQMatmul::eval_gpu` 调用 `quantize_dequantize_input`，后者产生浮点结果；它不是本项目想要的持续低位激活流水。`QuantizedMatmul::eval_gpu` 分派 QMV/QMM 内核直接读取打包权重。最终前向无显式 `mx.dequantize`、无每层完整 BF16 权重展开、无 QQMM 的激活格式往返；内核内部必要的解包/缩放仍然存在，不能宣称零反量化或纯整数乘法。

未取得 GPU kernel 时间线来精确拆分解包、拷贝和 GEMM 的贡献；源码分派和模块耗时用于定位方向，不把全部速度退化都归因于反量化。

## 候选扫描

下表是首轮独立进程、每个 12 次稳态的筛选值，后续有 20/30/40 次、正反顺序和最终 200 次复测。

| 全候选线性层统一模式 | 平均 ms |
|---|---:|
| accuracy | 287.937 |
| mxfp4 | 320.542 |
| mxfp8 | 365.833 |
| mxfp8a | 1744.828 |
| nvfp4 | 298.146 |
| nvfp4a | 915.849 |
| w1 | 312.097 |
| w2 | 309.898 |
| w3 | 317.648 |
| w4 | 314.892 |
| w6 | 336.497 |
| w8 | 339.767 |

进一步测试：BF16/FP16 快速 VAE、Qwen QKV/gate-up 合并、动作独立 W1/W2/W4、group 32/64/128、NVFP4、条件投影选择性量化、小 M 分块强制 QMV 分派。全模型位宽越低并不越快；分块引入额外调用/拼接，没有保留。

缓存补验：加载后调用 `mx.clear_cache()` 的额外 100 次稳态为 277.203 ms，与正式 speed 277.018 ms 接近，没有支持启动时清缓存能提速的证据，因此未加入生产路径。记录：`records/speed_cache_check_20261103/`。

## 数值验证

- 默认 accuracy：162 个保存数组与此前默认版本逐字节一致，最大差 0。正式 CLI 也验证了不传 `--profile` 进入 accuracy。
- speed：51 个离线真实文本/观测用例输出有限，token IDs 完全不变；相对本轮 accuracy 的最终动作 max_abs=2.292184，MAE=0.385081，RMSE=0.517981。
- 覆盖实际文本长度、非单调切换、额外 193-token 输入和三相机/状态/文本扰动；动作 horizon=1/8/16/32/64 及多种去噪步数另做有效性检查。
- W1 动作偏差显著，只作为本轮忽略精度的速度实验；未做 RoboTwin 闭环成功率，不代表可用于真实控制。默认 BF16 分支继续作为后续精度对照。

原始动作/归一化动作、噪声、各步 trace 和采样输入保存在正式记录中。`summary.json` 包含逐关节最大误差/RMSE、逐用例误差及代码 SHA256；完整验证还保存了相对 PyTorch 的误差。

## 记录目录

- `records/speed_operator_audit_20261103/`、`speed_operator_inventory_20261103/`：算子类型。
- `records/speed_sweep_20261103/`：统一量化。
- `records/speed_mixed_20261103/`：融合和混合方案。
- `records/speed_tuning_20261103/`：分组大小。
- `records/speed_repeat_20261103/`：正反顺序复测与条件投影。
- `records/speed_chunk_20261103/`：小 M 内核分派。
- `records/speed_final_20261103/`：正式 ABBA 性能、源码快照、完整离线验证、CLI、动态形状及总表。

复测命令（使用新的输出目录）：

```bash
.venv/bin/python scripts/benchmark.py --profile speed \
  --reference records/final_default_20261103 \
  --iterations 100 --warmup 8 --profile-iterations 10 \
  --output records/my_speed_benchmark
```

下一步需针对大 M 的视觉计算研究更高吞吐的原生低位 GEMM/融合内核，并用真实 kernel trace 验证；继续降低存储位宽或强制量化所有算子没有实测依据。
