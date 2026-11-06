# 最后一层 K/V 裁减复测（2026-11-03）

继续完成 100 组端到端配对和 100 组独立视觉前缀配对。正式默认代码未变化，选择性 K/V 投影仍保留为实验模式。

## 结构依据

最后一层视觉 linear1 权重实际为 [27648,3072]。动作推理只使用其中 K/V 的 6144 行（3072:9216）；候选减少该投影约 77.8% 的输出计算。该比例不是全模型或端到端提速比例。所有模型文件保持不变。

## 结果

|测量范围|默认均值 ms|候选均值 ms|平均降低|配对节省 ms 的 95% 区间|
|---|---:|---:|---:|---|
|完整端到端，100 组|311.746|308.402|1.073%|[-3.239, 9.859]|
|仅视觉前缀，100 组|129.818|127.113|2.083%|[-0.9391112501907628, 6.373208680197422]|

AB/BA 交替，每次计时前后同步 GPU，保留全部样本。区间采用 10000 次逐对 bootstrap，seed=42，未校正时序相关性。两项平均值均下降，但收益区间跨零，不能认定稳定提速，因此不升级默认路径。

端到端包括三图/状态预处理、完整动态文本在线编码、VAE、视觉前缀、十步去噪和 CPU 动作返回，不含模型加载和文件 I/O。

独立前缀测量使用预先计算并驻留的 hidden/latent/state；只测 prefix 返回全部 25 层 K/V 的求值时间。它不是完整推理，不能与在线端到端混用，也不能把两者相减作为其他模块精确耗时。

本轮每组前缀测试均在计时外检查全部 25 层 K/V，输出逐值一致。端到端动态初筛覆盖 31→79→48→31 tokens、horizon/steps=8/3、16/7、32/10，全部一致。此前完整 51 样本及扰动共 162 文件也逐值一致，见 [完整验证](../../records/selective_kv_validation_20261103/selective_vs_original.json)。没有闭环任务成功率验证。

## 环境和波动

Apple M5 Pro（15 CPU / 16 GPU 核）、24 GiB 统一内存，macOS 27.2、Python 3.12.13、MLX 0.32.4。端到端基线范围 278.70–372.54 ms，候选 278.12–376.46 ms，波动较大。

期间只读检查 pmset -g therm：没有记录 thermal/performance warning；memory_pressure -Q 报告系统内存 free percentage 45%。这些是单次观测，不能据此排除 GPU 频率变化、并发负载或其他系统影响，未更改系统电源设置或终止其他进程。

## 复现

在 /Users/qklj/work/cx/ImageWAM_MLX 下运行，输出目录必须不存在。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
# 100 组端到端：本轮候选平均 308.402 ms
.venv/bin/python scripts/selective_kv_experiments.py --output records/selective_kv_pair_repeat2 --iterations 100
# 100 组前缀：本轮候选 127.113 ms，不是端到端
.venv/bin/python scripts/profile_selective_prefill.py --output records/selective_prefill_repeat
```

## 记录

- [端到端原始结果及逐轮动作](../../records/selective_kv_repeat_20261103/results.json)。
- [端到端区间统计](../../records/selective_kv_repeat_summary_20261103.json)。
- [独立前缀原始结果及末层 K/V](../../records/selective_prefill_20261103/results.json)。
- [独立前缀区间统计](../../records/selective_prefill_summary_20261103.json)。
- [当前正式默认路径](README_CURRENT_RUNTIME_20261103.md)。

结论：停止继续堆叠同条件复测，不把小于当前噪声的均值改善当作确定收益。该候选保留，后续优先考虑具有更大理论覆盖范围的融合，例如多层 RMSNorm 与 RoPE 联合 kernel；这项本轮未实现，也未计入收益。
