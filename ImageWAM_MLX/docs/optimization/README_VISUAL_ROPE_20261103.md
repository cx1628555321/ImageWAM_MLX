# 视觉＋动作 RoPE Metal 融合（2026-11-03）

新增可选 `visual_rope` 模式。在已验证的动作 RoPE 模式上，将相同 strided Metal kernel 扩展到视觉前缀的 Q/K 旋转。保留 lazy_prefix、原始 BF16 权重、在线文本/图像编码、状态、KV 和十步去噪，不缓存观测特征，不固定或截断文本。

## 实现与边界

[适配实现](../../scripts/visual_rope_experiments.py) 为视觉前缀建立独立的编译函数，绑定单独的视觉权重包装对象，避免基线和候选切换时意外捕获另一模式。权重数组共享原值，不生成新权重副本；视觉原始 RoPE 被替换，其余计算保持原样。

[运行入口](../../scripts/run_visual_rope_optimized.py) 显式选择模式，默认 infer.py 和核心 model.py 未更改。回退可继续使用 run_rope_optimized.py 的 action-only rope。

kernel 复用 [动作 RoPE 实现](../../scripts/rope_experiments.py)，显式读取输入 strides，合并 FP32 转换、旋转及 BF16 输出，保留分步乘积。未替换 RMSNorm、Attention、KV 拼接或时间调制。

## 配对性能

环境：Apple M5 Pro，15 CPU / 16 GPU 核、24 GiB 统一内存，macOS 27.2、Python 3.12.13、MLX 0.32.4。[完整环境](../../records/environment.json)。

40 组 AB/BA 交替、计时前后同步。基线是上一轮已融合动作 RoPE 的版本，不是原始 MLX。端到端包含已解码三图和状态预处理、完整文本在线编码、VAE、视觉前缀、十步去噪、CPU 动作返回；不含文件读取、模型加载和结果保存。

|模式|平均 ms|P95 ms|
|---|---:|---:|
|仅动作 RoPE 基线|292.088|294.886|
|视觉＋动作 RoPE|286.357|287.846|

平均缩短 5.732 ms，提速 1.962%。逐组节省时间 bootstrap 95% 区间为 [5.073, 6.345] ms（10000 次、seed=42；未校正时序相关性）。不能将历史独立测试之间的差值当作累计优化收益。

## 验证范围

配对初筛覆盖 31→79→48→31 tokens，horizon/steps 为 8/3、16/7、32/10，全部逐值一致。

完整 51 样本验证覆盖 48 种已观测文本长度和额外真实观测，另做非单调长度切换及逐路摄像头、状态、文本扰动。kernel 独立检查覆盖长度 31、80、288、320、328、368，各自测试连续与非连续视图，以及非连续 cos/sin。离线数值验证不代表全数据集或闭环任务成功率。

## 复现命令

在项目根目录执行，输出目录必须不存在。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
# 配对端到端：本轮 292.088 -> 286.357 ms
.venv/bin/python scripts/visual_rope_experiments.py \
  --output records/visual_rope_pair_repeat --iterations 40

# 独立加载、首次推理、50 轮稳态和模块耗时
.venv/bin/python scripts/run_visual_rope_optimized.py \
  --mode visual_rope --entry benchmark \
  --reference ../ImageWAM/evaluate_results/m5pro_robotwin/20261103_dynamic_profile \
  --output records/visual_rope_benchmark_repeat --warmup 8 --iterations 50

# 完整精度/动态输入检查；含首次形状编译，不用于稳态统计
.venv/bin/python scripts/run_visual_rope_optimized.py \
  --mode visual_rope --entry validate_cases \
  --output records/visual_rope_validation_repeat
.venv/bin/python scripts/validate_visual_rope_kernel.py \
  --output records/visual_rope_kernel_repeat

# 单次真实三图推理
.venv/bin/python scripts/run_visual_rope_optimized.py \
  --mode visual_rope --entry infer \
  --inputs examples/robotwin_ep0_frame0/inputs.npz \
  --prompt-file examples/robotwin_ep0_frame0/prompt.txt \
  --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy \
  --output records/visual_rope_infer_repeat
```

## 原始记录

- [40 组配对及每轮输出](../../records/visual_rope_20261103/results.json)、[置信区间](../../records/visual_rope_summary_20261103.json)。
- [完整验证](../../records/visual_rope_validation_20261103/validation.json)、[逐文件差异](../../records/visual_rope_validation_20261103/visual_rope_vs_original_mlx.json)。
- [独立基准、每轮动作及源码快照](../../records/visual_rope_benchmark_20261103/benchmark.json)。
- [长序列 kernel 验证](../../records/visual_rope_kernel_20261103/results.json)。

本轮没有解决 Xcode 缺失问题，也未声称取得 kernel 时间线；性能结论来自端到端配对测试。原有 Metal capture 保留。RMSNorm+RoPE 联合归约及分段 KV Attention 尚未实施。

## 完整验证与独立基准结果

51 个代表样本及额外输入扰动、长度切换，共 162 个对比文件全部与原 MLX 逐值一致。12 组视觉长度 kernel 测试全部逐值一致。

独立进程：加载 2.529 s；首次推理 1266.467 ms；8 次预热、50 次稳态平均 289.962 ms，P95 291.794 ms。上述 benchmark 启动命令对应这些数值。OS/驱动缓存未清理，不代表严格冷启动。

|模块（另 5 次同步分析）|平均 ms|
|---|---:|
|preprocess|3.277|
|tokenization|0.273|
|text_encoder|39.959|
|image_encoder|53.150|
|visual_prefill|117.261|
|action_denoise|81.122|
|postprocess|0.184|

模块分析增加求值和同步，不可把各模块求和直接当作无同步稳态端到端。跨轮独立均值受系统状态影响，本轮收益以同期配对为依据。
