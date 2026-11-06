# VAE 深入诊断与精确幂运算优化（2026-11-03）

本轮不再盲目扩大图融合范围，而是定位上一轮 GN 编译精度变化的首个算子，并实现保留原始数值行为的新路径。默认推理配置暂不改变。

## 定位结果

M5 Pro / 24 GiB / MLX 0.32.4，当前默认 BF16 VAE，RoboTwin 288×256 拼图。三轮逐算子同步诊断中，卷积合计平均 30.860 ms，20 次 GN+SiLU 合计 24.752 ms，另外 GN 约 0.790 ms。高分辨率 `encoder.down.0` 的四次 GN+SiLU 合计约 15.796 ms，是优先研究点。

**这些是插入同步后的诊断值，不是正常调度下的 GPU kernel 时间。** 它们不能直接与约 42 ms 的完整图像编码相加、计算占比或推断内存带宽；卷积、归一化之外仍有 attention、残差加法等操作。

对每个 GN 逐步检查：20/20 层的 mean 与 centered 输入在 eager/compiled 之间一致，20/20 层的平方结果不一致，随后影响方差与 BF16 输出。检查程序返回了中间张量，这也可能影响编译，因此它说明了实际复现的数值差异起点，不是对所有编译图的机器码证明。

MLX v0.32.4 的 `Power` 浮点实现调用 `metal::pow`；不能默认把 `x ** 2` 改成 `x*x` 后仍逐位一致。依据：[官方版本源码](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.4/mlx/backend/metal/kernels/binary_ops.h)。本次 GPU 实测是主要依据，未把其他平台的 issue 当成当前设备的测量结论。

## 两个新实现

`imagewam_mlx/pow_vae.py`：

1. **fused**：合并 BF16→FP32 读取、减当前组均值、原始平方幂运算。保持原有 mean/variance 归约形状和顺序。普通 `metal::pow` 自定义 kernel 仍产生差异；显式 `metal::precise::pow` 恢复初筛 latent 一致。指数使用运行时单元素数组，避免常量 2 的编译改写。
2. **lookup**：对大的 BF16 张量，按当前输入的每个组均值，计算全部 65,536 种 BF16 位模式的 centered-power 值，再逐元素查表；小张量走 fused 路径。无需近似、量化或降低分辨率。表仅依赖当前次 mean，**每次重新生成，没有跨输入/帧缓存**。batch=1 时临时表约 8 MiB；包含所有位模式，未使用的 NaN/Inf 表项不作为正常激活读取。

查表只在 `x.size >= 8,000,000` 且 dtype=BF16 时启用，这是性能分流阈值，不是固定图像输入尺寸。shape、batch、组索引来自当前张量，GN 组数仍是模型原有的 32。文本路径完全不变。

## 本轮实验

| 图像编码局部实验 | 原版 ms | 候选 ms | latent 一致 | 节省 95% CI，ms |
|---|---:|---:|---|---|
| 普通 pow 融合，80 对 | 45.344 | 43.111 | 否，max_abs=0.15625 | [0.278, 4.203] |
| precise pow 融合，40 对 | 45.509 | 45.203 | 是 | [-3.036, 3.597] |
| precise pow + 残差编译，40 对 | 44.646 | 44.888 | 是 | [-3.162, 2.605] |
| precise pow 查表，80 对 | 45.245 | 44.308 | 是 | [-1.078, 3.076] |

普通 pow 结果被拒绝，当前保存的实现已经使用 precise pow。早期两次 Metal 编译错误的日志保留：标量索引和不存在的 size 元数据引用均已修复，没有把失败运行计入性能。

首轮 100 对 AB/BA 端到端：原版 293.054 ms，lookup 289.388 ms，平均节省 3.666 ms；bootstrap 95% CI 为 **[-1.793, 9.395] ms**。所有动作逐字节一致，但区间跨零，不能认定稳定加速。没有删除慢样本。设备上同时观察到 mediaanalysisd、视频编解码服务等活动；这不证明它们造成了 GPU 抖动，也未停止这些系统或用户进程。

独立进程 **200 对复测**：原版 **289.952 ms**，lookup **290.380 ms**，平均节省 **-0.427 ms**，95% CI **[-4.388, 3.570] ms**，全部动作一致。未复现稳定收益，因此两种新 pow 模式默认关闭。不能挑选首轮均值或最小样本宣称加速。

正式 `--vae-pow-mode lookup` 接口另完成 8 次预热、30 次稳态、10 次模块分析：端到端均值 **285.049 ms**、P95 **312.354 ms**，图像编码同步分析均值 **45.133 ms**。最终动作与原默认基准逐字节一致。这是集成验证，不能替代上面的同期配对结论。

## 精度

lookup 完整验证：51 个离线案例及输入扰动/长度切换产生的 **162 个 NPY 输出**，与上一轮默认 MLX 逐字节一致。4 个真实观测的 **40 个残差块中间输出、4 个完整 latent** 全部一致。覆盖不同真实图像及文本长度，不只对单张固定图做比较。

这是对现有 MLX 行为的数值回归，不代表与 PyTorch 逐位相同或已证明闭环成功率。跨框架既有 BF16 误差未增加。fused-only 单独完成初筛，完整回归主对象是包含 fused 回退的 lookup 路径。

## 可选运行接口

```python
model = ImageWAM('weights/bf16', 'assets', vae_pow_mode='lookup')
# 'original'（默认）、'fused'、'lookup'
```

`legacy`、关闭 fused_vae、启用另一实验 vae_group_inv 时有效模式保持 original；启用 pow 模式时不叠加 vae_tail_compile。benchmark JSON 保存实际模式。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
.venv/bin/python scripts/benchmark.py --vae-pow-mode lookup \
  --reference ../ImageWAM/evaluate_results/m5pro_robotwin/20261103_dynamic_profile \
  --output records/new_vae_lookup --warmup 8 --iterations 30 --profile-iterations 10
.venv/bin/python scripts/test_vae_lookup_e2e.py \
  --pairs 200 --output records/new_vae_lookup_pairs
.venv/bin/python scripts/run_vae_lookup_validation.py \
  --output records/new_vae_lookup_validation
```

所有输出目录应为新目录。正式默认不加 `--vae-pow-mode`，继续使用已确认的原始路径。

## 证据位置

项目根目录 `/Users/qklj/work/cx/ImageWAM_MLX`：

- `records/vae_deep_diagnostic_20261103/results.json`：逐算子时间、20 层 GN 中间步骤差异。
- `records/vae_precise_pow_20261103`、`records/vae_precise_pow_res_20261103`、`records/vae_pow_lookup_fixed_20261103`：局部配对和 latent。
- `records/vae_lookup_e2e_20261103`、`records/vae_lookup_e2e_repeat_20261103`：完整推理配对时间和动作。
- `records/vae_lookup_validation_20261103`、`records/vae_lookup_checks_20261103`：全部输入精度、最终输出与 latent。
- `records/vae_deep_source_20261103`：修改前默认入口备份。
- `records/vae_lookup_native_20261103`：正式可选入口逐次时间、输出、源码快照和 trace。

没有覆盖模型、配置或旧性能记录。当前更大的收益仍需结合卷积/归约实现研究，不能把查表或更多融合本身当作已实现的性能收益。
