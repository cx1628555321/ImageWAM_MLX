# 图调度与 KV 优化实验（2026-11-03）

完成三种图调度适配及 KV 微基准。可选推荐 `lazy_prefix`：保留 both_compile 算术，仅移除预测主路径中视觉前缀后的显式 `mx.eval(cache)`，让后续依赖调度计算。原入口、模型权重和配置不变。

## 环境与测量

Apple M5 Pro，15 CPU 核 / 16 GPU 核、24 GiB 统一内存、macOS 27.2、Python 3.12.13、MLX 0.32.4。环境细节见 [environment.json](../../records/environment.json)。三图、完整动态文本、状态、相同初始噪声；BF16，10 步去噪，无特征缓存。端到端：已解码 RGB、状态及文本到 CPU 动作，不含模型加载、文件读取及结果落盘。

## 图调度配对测试

每个候选 30 组 AB/BA 交替，计时前后同步 GPU。基线为 both_compile。

|方案|同期基线 ms|候选 ms|延迟降低|节省 ms 的 95% 区间|
|---|---:|---:|---:|---|
|lazy_prefix|293.144|290.586|0.87%|[2.064, 2.993]|
|chunk2|295.316|295.738|-0.14%|[-0.694, -0.132]|
|chunk5|295.984|297.578|-0.54%|[-1.923, -1.258]|

置信区间是逐组时间差的 10000 次 bootstrap（seed=42），未校正时序相关性和多重比较。不能把不同轮次的系统波动算作优化收益。两步、五步联合编译均未提速；保留为实验模式。这里只联合编译去噪步骤，未宣称整个模型融合为单个 kernel。

初筛：31→79→48→31 tokens；另测试 horizon/steps 为 8/3、16/7、32/10，验证非整除尾块。三种候选均与同期基线逐值一致。

## lazy_prefix 完整验证和独立性能

51 个代表样本覆盖 48 种已观测文本长度及 3 个额外真实观测；另包含长度切换、超长文本、摄像头/状态/文本扰动。与原 MLX 对比的 162 个保存文件全部逐值一致（51 组动作、归一化动作、token IDs，共 153 个，另 9 个扰动/切换输出）。输入敏感性检查通过。不是全数据集或闭环成功率测试。

相对 MPS dynamic 最大动作差异 0.0158013701，平均逐样本 MAE 0.0008851273，与原 MLX 验证结果一致。

独立运行：8 次预热，50 次稳态，另 5 次同步模块分析。
- 加载：2.531 s；首次推理：1586.286 ms。
- 稳态平均：294.137 ms；P95：305.094 ms。
- 首次包含本进程初始化/编译影响，未清空 OS/驱动缓存，不是严格冷启动。

|模块（同步分析）|平均 ms|
|---|---:|
|preprocess|2.920|
|tokenization|0.273|
|text_encoder|39.514|
|image_encoder|53.504|
|visual_prefill|122.946|
|action_denoise|84.354|
|postprocess|0.164|

模块 profile 会主动求值和同步，因此不能用 profile 模块求和评估取消前缀求值的收益；收益以无模块同步的配对端到端为准。

## KV 微基准：保留负结果

测试 BF16、24 heads、head_dim=128、16 action tokens；前缀长度 320/328/368 对应三种动态文本形状。合成驻留输入，预热 10 次，100 次交替同步计时。比较拼接本身、拼接后官方 SDPA、分段 logits 加统一 softmax 的普通算子实现。

|前缀 tokens|K/V 拼接 ms|拼接＋SDPA ms|分段普通算子 ms|Attention 输出最大差异|
|---|---:|---:|---:|---:|
|320|0.2377|0.3412|0.3542|0.00244141|
|328|0.1956|0.2312|0.2461|0.00292969|
|368|0.1971|0.2283|0.2448|0.00244141|

分段实现没有获得收益，且不同 GEMM/softmax 路径引入误差，未接入模型推理。它使用一个联合 softmax，不是把两次分别归一化的 Attention 相加。

这些时间包含每次同步和调度开销，不能把拼接时间乘 250 推算端到端收益，不能把两列之差当作精确 kernel 时间。尚未取得 Metal trace，也未实现分段 KV 的自定义 Metal kernel；本实验不证明底层融合分段 Attention 没有潜力。

## 复现命令

在项目根目录执行；输出目录必须不存在，避免覆盖记录。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
# 推荐可选模式：本轮独立稳态 294.137 ms，配对提速 0.87%
.venv/bin/python scripts/run_graph_optimized.py --mode lazy_prefix --entry benchmark \
  --reference ../ImageWAM/evaluate_results/m5pro_robotwin/20261103_dynamic_profile \
  --output records/lazy_prefix_benchmark_repeat --warmup 8 --iterations 50

# 完整精度和输入变化验证，不用该脚本统计稳态性能
.venv/bin/python scripts/run_graph_optimized.py --mode lazy_prefix --entry validate_cases \
  --output records/lazy_prefix_validation_repeat

# 三种调度配对测试；端到端时间见上表
.venv/bin/python scripts/graph_experiments.py --output records/graph_repeat --iterations 30

# KV 微基准，结果不是端到端耗时
.venv/bin/python scripts/kv_microbench.py --output records/kv_micro_repeat
```

`run_graph_optimized.py` 也支持 `--entry infer`，其输入参数与现有 infer.py 相同。chunk2/chunk5 仅供实验；它们在 trace=True 时回退到原单步编译路径以导出每步中间量，元数据明确记录这一点。lazy_prefix 的 trace 路径没有这个回退。

## 实现与原始记录

- [图调度实现](../../scripts/graph_experiments.py)、[运行入口](../../scripts/run_graph_optimized.py)、[KV 探针](../../scripts/kv_microbench.py)。
- [配对原始数据及每轮动作](../../records/graph_20261103/results.json)。
- [配对置信区间](../../records/graph_summary_20261103.json)。
- [KV 原始数据和输出](../../records/kv_micro_20261103/results.json)。
- [51 样本验证](../../records/lazy_prefix_validation_20261103/validation.json)、[逐文件一致性](../../records/lazy_prefix_validation_20261103/graph_vs_original_mlx.json)。
- [独立基准及源码快照](../../records/lazy_prefix_benchmark_20261103/benchmark.json)。

下一阶段应先用 Metal trace 确认真实的 KV 拼接与小算子成本，再决定是否开发分段 KV kernel、融合 RMSNorm/RoPE，或缓存时间调制等输入无关内容。这些方向本轮未实现，不包含在已测收益中。
