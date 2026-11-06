# 动作推理冗余加载与计算核查（2026-11-03）

后续状态：下述剩余 Q/MLP 与 VAE 输出通道裁剪已适配并通过精度验证，见 [裁剪版验证记录](README_COMPACT_ACTION_20261103.md)。本文保留裁剪前核查证据。

范围：M5 Pro 的 `/Users/qklj/work/cx/ImageWAM_MLX` 默认 MLX 动作推理；三路 RoboTwin 图像、动态文本。此裁剪不适用于图像生成路径。

## 结论

不能声称当前已经删除全部冗余。已适配 PyTorch 跳过的视觉末层 attention / 输出分支，MLX 默认路径同样跳过；但此前 MLX 仍加载了末层未使用的 linear2。本次在校验原始 manifest 后、mx.eval 前移除该权重，运行时权重集合减少 75,497,472 字节（72 MiB）。原始权重文件与配置保持不变。没有测量磁盘实际读取量，也不把此数值当作进程峰值内存或耗时下降。

| 部位 | 当前加载与计算 | 核查依据 |
|---|---|---|
| Qwen | 仅 embedding 与层索引 0–26；取第 9/18/27 层原始隐藏状态；不加载/运行后续层、最终 norm、LM head | convert_weights.py、qwen.py |
| 视觉输出头 | final_layer 权重已排除，动作路径不调用 | convert_weights.py、model.py |
| VAE decoder | 权重已排除，只有 encoder 执行 | convert_weights.py、vae.py |
| 视觉最后 single block 的 attention、linear2、残差输出 | 不执行；本次也从运行时加载集合移除 linear2 | model.py 的 prefill 与 load |
| 视觉最后 single block 的 linear1 | **仍加载完整 [27648,3072]，默认仍构造完整投影后取 K/V** | model.py、ops.py |
| 视觉末层 Q norm/RoPE | Python 路径仍构造，返回缓存不依赖 Q；是否实际执行受 MLX 惰性图消除影响，不将其当作实测 kernel 开销 | ops.py、runtime.py |
| VAE quant_conv | **仍计算 64 输出通道后取前 32** | vae.py |
| 动作专家 | 全部 5 个 double 和 20 个 single block 及动作 head 必需 | runtime.py 的 _step |

## 与已适配 PyTorch 对照

`../ImageWAM/scripts/m5pro/action_prefill.py` 在视觉 single 最后层取得 K/V 后 break，跳过 mixed attention 与 block._out。MLX `model.py::prefill` 的 `if i<19` 等效跳过这段输出计算。PyTorch 的这份适配同样仍调用末层完整 QKV/MLP prepare，因此不应把它描述成已经裁去整个最后一层。

默认 MLX `runtime.py` 继承该 prefill，然后编译；已有 `selective_kv_experiments.py` 属于实验实现，并不是默认启用。该实验只计算末层 K/V，但仍加载完整投影权重，不能据此声称加载也已裁剪。

## 尚可裁剪的边界

1. 末层 linear1 只需第 3072:9216 行，即 K/V 的 6144 行。其余 21504 行占 126 MiB BF16 权重；query_norm.scale 另有 256 字节。真正减少加载需要配套权重布局/读取逻辑，不能只在 matmul 前切片就宣称没有读取完整权重。
2. VAE quant_conv 可尝试仅输出均值的 32 通道。它的前 32 个卷积核仍依赖上游全部 64 通道，不能因此把 encoder.conv_out 也减半。需要验证卷积输出维度改变是否影响浮点舍入。
3. Qwen 的第 9/18/27 层输出依赖此前全部层；视觉前 24 个 block 的输出用于下一层 K/V；动作最后一层输出用于动作 head。这些不能按“没有直接返回最终结果”判断为冗余。

已有 selective K/V 回归曾通过全部 162 个数组的逐位比较，但重复性能测试的置信区间跨零（见 README_SELECTIVE_KV_RETEST_20261103.md）。本次不把该实验升级为默认速度优化，也不宣称新增速度收益。

## 本次修改与验证

修改 `imagewam_mlx/model.py`：保留完整 checkpoint 键名/形状校验，随后从 visual 运行时字典删除 single_blocks.19.linear2.weight 与可选 bias，再执行 mx.eval。default 与 legacy 都不使用该输出投影，因此两条路径共用此过滤规则。

执行命令（工作目录为 ImageWAM_MLX；输出目录必须不存在）：

```sh
.venv/bin/python scripts/validate_cases.py --output records/redundancy_validation_20261103
.venv/bin/python scripts/audit_redundant_weights.py
```

51 个离线用例、文本长度切换及三相机/状态/文本扰动输出均保存。与 `records/validation_v1` 比较 162 个数组；结果保存在 `records/redundancy_audit_20261103/audit.json`。逐位一致是相对既有 MLX 基线，不能解释成与 PyTorch 完全一致或闭环任务成功率不变。

修改前 model.py 保存在 `records/redundancy_audit_20261103/model_before.py`，日志位于同目录 validation.log。该轮逐用例时间包含形状首次编译，不作为稳态端到端性能数据。
