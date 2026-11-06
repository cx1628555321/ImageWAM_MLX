# CPU/GPU 重叠、组内复用与 Qwen RoPE 适配结果

已部署到 `/Users/qklj/work/cx/ImageWAM_MLX`。本轮三项均完成实现和验证；**默认只启用 CPU/GPU 重叠**，其余两项提供原生可选开关，默认关闭。继续保留上一轮 compact 权重和 VAE GN+SiLU 融合。原始权重与配置未覆盖，无文本或图像特征缓存。

## 实现

- `runtime.py`：先按实际文本分词，提交 Qwen 的 `mx.async_eval(hidden)`，在 CPU 完成三图 PIL 缩放/拼图，然后按真实依赖执行 VAE、视觉前缀和动作。仍用默认 GPU stream，没有宣称两个 GPU 编码器并行。CPU 图像预处理仍在端到端计时内。
- `fused_vae_inv.py`：保留原 mean/variance，仅对当前输入的每个 group 计算一次 FP32 inverse std，再由融合后处理读取；不跨帧缓存。默认关闭。
- `fused_qwen_rope.py`：Qwen 前半/后半配对旋转，保持两次 BF16 乘法和一次 BF16 加法的舍入边界，支持实际 token 长度及 strided Q/K。原始 RMS、QKV GEMM、注意力和输出层均不变。默认关闭。
- `qwen.py` 增加独立 rope 方法以支持原生替换，避免运行时依赖实验脚本。

## 同期配对结果

每项独立对照“compact + fused VAE”的上一轮默认版。首轮 40 组，复测 80 组 AB/BA；每次前后同步 GPU，在线三图/文本编码、10 步去噪，包含 CPU 最终动作，不含模型加载和文件 IO。

|候选/轮次|基线 ms|候选 ms|节省 ms|节省的 95% 区间|决定|
|---|---:|---:|---:|---|---|
|CPU/GPU 重叠，首轮|289.129|286.906|2.223|[1.457,3.042]|通过|
|CPU/GPU 重叠，80 组复测|290.078|287.355|2.723|[2.192,3.258]|默认启用，约 0.94%|
|VAE 组内复用，首轮|286.654|285.605|1.050|[0.247,2.231]|初筛有收益|
|VAE 组内复用，80 组复测|303.705|306.880|-3.175|[-11.300,5.138]|收益未复现，默认关闭|
|Qwen RoPE，首轮|290.130|284.647|5.484|[4.970,5.990]|初筛有收益|
|Qwen RoPE，80 组复测|301.537|297.100|4.437|[-2.085,10.971]|区间跨零|
|重叠+Qwen 组合，80 组|297.229|289.819|7.410|[2.243,12.617]|组合有收益，但不足以证明 Qwen 增量|
|已启用重叠后再加 Qwen，120 组|289.746|290.550|-0.804|[-6.222,4.609]|增量收益未确认，Qwen 默认关闭|

不把各候选首轮收益相加。区间采用 10000 次配对 bootstrap，seed=42，未校正时间相关性或多重比较。系统波动明显，保留所有结果，没有剔除慢样本或只选择最快轮次。

## 精度

- 三个单项候选、组合候选、正式原生默认入口：各自 **162 个数组与上一轮默认基线逐值及逐字节一致**，包含 51 个离线用例、长短文本切换、193-token 文本和三相机/状态/文本扰动。
- 组合诊断：219 项检查一致，涵盖文本特征、图像 latent、全部 25 层 K/V、逐步动作和最终输出；其中另包含 VAE 组内复用的 latent 比较。
- 原生默认及各可选开关：492 项比较逐字节一致；包含 31/79/193-token 文本、动作长度 1/8/16/32/64、不同步数、trace 中间量及最终动作。
- 相对原 PyTorch 的既有 BF16 差异未变。结论是所测离线数值等价，不是所有可能输入的形式证明或闭环任务成功率测试。

记录根目录：`records/remaining_20261103`。精度汇总在 `summary.json`；中间量在 `intermediates`，原生开关结果在 `native_options`，正式默认输出在 `native_validation`。所有候选及基线输出均保留。

## 正式默认版独立性能

M5 Pro 15 核 CPU / 16 核 GPU，24 GiB 统一内存，无独立显存；macOS 27.2、Python 3.12.13、MLX 0.32.4。8 轮预热、50 轮稳态。

|指标|实测|
|---|---:|
|模型加载|2.638 s|
|首次推理（含本进程首次编译）|1144.741 ms|
|稳态端到端平均|294.692 ms|
|中位数|288.345 ms|
|P95|341.548 ms|

本轮独立均值受到尾延迟影响；不能与上一轮独立 287.763 ms 直接相减判断版本退化或收益。优化收益依据上面的同期配对测试。未清理 OS/驱动缓存，首次不是严格冷启动。

profile 模式主动同步文本编码，会抑制这项 CPU/GPU 重叠，因此不能用同步模块求和判断它的收益。完整性能、源码快照和 `overlap/qwen_rope/vae_group_inv` 标记在 `native_benchmark/benchmark.json`。

## 启动命令与回退

所有命令在项目根目录执行，输出目录必须不存在。

```sh
# 正式默认：本轮独立平均 294.692 ms；同期配对平均节省 2.723 ms。
.venv/bin/python scripts/benchmark.py \
  --reference ../ImageWAM/evaluate_results/m5pro_robotwin/20261103_dynamic_profile \
  --output records/remaining_default_repeat --warmup 8 --iterations 50

# 回退本轮调度，保留上一轮 compact + fused VAE。
.venv/bin/python scripts/benchmark.py --no-overlap \
  --reference ../ImageWAM/evaluate_results/m5pro_robotwin/20261103_dynamic_profile \
  --output records/remaining_baseline_repeat --warmup 8 --iterations 50

# 原生可选项：精度已测通过，但增量性能不稳定，非默认推荐。
# 在 benchmark 命令追加 --qwen-rope 或 --vae-group-inv。

# 全样本验证；不将该命令的首次形状耗时当作稳态性能。
.venv/bin/python scripts/validate_cases.py --output records/remaining_validation_repeat

# 固定上一轮基线的单项配对实验。
.venv/bin/python scripts/test_remaining.py --mode overlap --pairs 80 \
  --output records/overlap_paired_repeat
# mode 可选 group_inv / qwen_rope / combined。
# --mode combined --baseline-mode overlap 用于测 Qwen 的增量收益。
```

Python API：

```python
from imagewam_mlx import ImageWAM
model = ImageWAM('weights/bf16', 'assets')  # overlap=True，其余两个新开关=False
baseline = ImageWAM('weights/bf16', 'assets', overlap=False)
# 两项可选实验：qwen_rope=True、vae_group_inv=True。
# legacy 模式不启用本轮优化；关闭 fused_vae 时组内复用也不启用。
```

上面实例展示不同配置，24 GiB 机器进行实际测试时应顺序创建/释放实例，避免同时加载两套完整模型影响内存与速度。

修改前代码快照在 `records/remaining_20261103/before`。本轮未实施系统 wired limit 调整、文本特征缓存或 shapeless 编译；这些属于后续独立议题。
