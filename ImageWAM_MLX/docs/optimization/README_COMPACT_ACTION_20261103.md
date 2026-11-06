# 动作推理权重与计算裁剪：适配及精度验证

后续更新：VAE GN+SiLU 融合已接入默认，正式稳态 287.763 ms；完整精度、各候选结果与回退方式见 [本轮适配报告](README_NEXT_OPTIMIZATIONS_RESULTS_20261103.md)。

已部署到 M5 Pro `/Users/qklj/work/cx/ImageWAM_MLX`，默认入口自动使用 `weights/bf16/action_only`。原始权重及配置未覆盖。目录内只有视觉和 VAE 文件是新副本，其余分片通过相对符号链接复用原文件；请勿删除原始权重目录。

## 实际裁剪

- 视觉最后 single block：完整 linear1 [27648,3072] 改为 kv [6144,3072]，直接复制原权重行 3072:9216。删除 Q/MLP 输出、query_norm 权重以及未使用的 linear2；最后层只产生动作专家需要的 K/V。
- VAE quant_conv：输出 64 通道改为 32，保留原均值分支权重与 bias。输入仍为 64 通道，encoder.conv_out 不裁剪。原代码的 `[..., :32]` 在新布局上只是取全部输出。
- 保留的权重直接复制 safetensors 字节，无重算、重训、量化或类型转换。新文件实际不含被删除的权重，运行时不再加载这些部分。
- 相对原始完整转换权重，删除 207,622,464 字节有效张量数据（约 198.004 MiB）；其中 72 MiB 的 linear2 已在上一轮从运行时集合移除，本轮新增约 126.004 MiB 的裁剪。不是进程峰值内存下降测量。
- 动态文本、三相机处理、动作长度与去噪时间表不变。无跨输入特征缓存。

## 精度结果

`records/compact_action_20261103/precision.json` 的 `all_equal=true`。

1. 完成 51 个离线用例及文本长度切换、图像/状态/文本扰动验证。162 个保存数组与裁剪前 `records/redundancy_validation_20261103` **逐位一致，最大绝对差为 0**。覆盖已收集 RoboTwin 文本长度及超出样本范围的 193-token 切换测试。
2. 额外比较 horizon/steps：1/1、8/3、16/7、16/10、32/10、64/2。使用相同默认优化内核，唯一开关为 compact。图像/状态/文本中间表示、VAE latent、抽查的第 0/4/5/24 层 K、每一步动作、归一化动作、最终动作均逐位一致。
3. 原有 MLX 与 PyTorch 的差异没有因本次裁剪改变。这是所测离线输入的数值等价证据，不是全部可能输入的形式证明或机器人闭环成功率测试。

原始输出数组、输入扰动结果及日志均保留在 `records/compact_action_validation_20261103` 和 `records/compact_action_20261103`。修改前代码在后者的 `before` 目录。

## 当前性能记录

同一 M5 Pro 环境：15 核 CPU、16 核 GPU、24 GiB 统一内存（GPU 共享，无独立显存）；macOS 27.2，Python 3.12.13，MLX 0.32.4。

8 轮预热、50 轮稳态，三相机在线编码、文本在线编码、10 步动作推理；输入已解码数组到 CPU 最终动作，不含文件读取和结果写盘。

| 指标 | 实测 |
|---|---:|
| 模型加载 | 2.571 s |
| 首次推理（含首次编译） | 1402.466 ms |
| 稳态平均端到端 | 304.580 ms |
| 稳态中位数 | 297.309 ms |
| 稳态 P95 | 342.878 ms |

原始记录及逐模块 profile 在 `records/compact_action_benchmark_20261103/benchmark.json`。本轮没有同期配对速度对照，不据此宣称稳定加速。

## 复现与回退

以下命令在项目根目录执行。生成权重脚本拒绝覆盖已存在目标目录；验证/性能输出目录也必须是新目录。精度对照脚本使用文中固定记录路径。

```sh
# 已执行，无须重复生成；原始 BF16 文件须保留。
.venv/bin/python scripts/compact_action_weights.py
.venv/bin/python scripts/validate_cases.py --output records/compact_action_validation_20261103
.venv/bin/python scripts/validate_compact_action.py
.venv/bin/python scripts/benchmark.py --reference ../ImageWAM/evaluate_results/m5pro_robotwin/20261103_dynamic_profile --output records/compact_action_benchmark_20261103 --warmup 8 --iterations 50
```

默认 `ImageWAM('weights/bf16','assets')` 检查裁剪副本 manifest 与原 manifest 的 SHA256 关联后选用新布局；没有完整副本则保持原路径。转换脚本最后才发布 manifest，未完成的转换不会被选中。

保持当前优化内核而关闭裁剪：

```python
from imagewam_mlx import ImageWAM
model = ImageWAM('weights/bf16', 'assets', compact=False)
```

`optimization='legacy'` 也使用原始权重路径。历史 selective-K/V 实验已显式设置 compact=False，以便继续复现完整投影对照。
