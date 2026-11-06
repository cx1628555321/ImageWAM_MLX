# Metal 捕获与时间调制复用（2026-11-03）

本轮成功保存 Metal capture，并实现、测试时间调制参数复用。没有获得新的可确认提速；继续推荐上一轮 `lazy_prefix`，不替换默认入口或模型权重。

## Metal 捕获

服务器仅安装 Command Line Tools，`xcrun --find xctrace` 返回找不到工具，`/Applications` 未发现 Xcode。按 [MLX 官方 Metal Debugger 文档](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html) 设置 `MTL_CAPTURE_ENABLED=1`，通过 `mx.metal.start_capture/stop_capture` 完成捕获。

- 文件：[action_step.gputrace](../../records/metal_action_20261103/action_step.gputrace)。
- 大小：18,328,807,461 字节，约 17.07 GiB。捕获文件包含关联资源，不能将文件大小视为该步骤实际内存流量。
- 范围：已计算的真实观测 KV 上执行 `steps=1` 的动作去噪；预热三次。该调度不是原十步轨迹中的某个步骤，不可拿输出直接与十步动作对比。
- [状态记录](../../records/metal_action_20261103/capture.json)、[脚本](../../scripts/capture_action_metal.py)。
- 缺少 Xcode，尚未重放和读取 kernel 时间线。因此尚不能给出真实 KV 拷贝耗时、kernel 热点排行，或断言带宽瓶颈。未用普通 CPU 墙钟微基准冒充 Metal profiler 数据。

## 固定时间调制复用

实现：[conditioning_experiments.py](../../scripts/conditioning_experiments.py)、[运行入口](../../scripts/run_conditioning_optimized.py)。

每个冻结模型实例，根据 `steps` 生成并缓存 action expert 的时间 embedding 后续调制：double/single modulation、head shift/scale 以及调度 delta。LRU 最多保存四种步数。保留原始 BF16 算术和形状输入，图像、文本、状态、动作和 KV 均不缓存；RoPE 仍根据当前 horizon 构建。

仅适用于固定权重和当前固定调度公式的推理。修改权重或调度后应重建适配器，不能复用旧缓存。首次构建开销计入首次候选调用。trace 与普通路径使用同一个条件化去噪实现。

实测环境：Apple M5 Pro（15 CPU / 16 GPU 核），24 GiB 统一内存，macOS 27.2、Python 3.12.13、MLX 0.32.4；[完整环境](../../records/environment.json)。

40 组 AB/BA 交替配对，基线为上一轮 `lazy_prefix`，每次计时前后同步 GPU，包含原始 RGB 预处理、在线文本编码、图像编码、前缀、十步动作去噪和 CPU 动作返回，不含文件读取与结果落盘。

|模式|端到端平均 ms|P95 ms|
|---|---:|---:|
|lazy_prefix 基线|297.704|298.807|
|conditioning 候选|298.048|303.935|

候选平均慢 0.344 ms（0.116%），没有证据证明提速。逐组节省时间的 95% bootstrap 区间为 [-1.139, 0.375] ms（10000 次，seed=42，未校正时序相关性）。首次候选调用 303.273 ms，是同进程预热基线后的调用，不代表独立冷启动或模型加载时间。

初筛：31→79→48→31 tokens，以及 horizon/steps 为 8/3、16/7、32/10，全部与同期基线逐值一致。完整 51 样本验证与扰动结果见下方记录。本轮不进行额外加载/独立稳态速度宣传，因为候选未胜过同期基线。

[配对原始结果及各轮动作](../../records/conditioning_20261103/results.json)，[区间统计](../../records/conditioning_summary_20261103.json)，[完整验证](../../records/conditioning_validation_20261103/validation.json)，[相对原 MLX 逐文件差异](../../records/conditioning_validation_20261103/conditioning_vs_original_mlx.json)。验证是离线代表样本，不是整个数据集或闭环成功率。

## 复现命令

输出目录必须不存在；保留已测记录。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
# Metal 捕获：不用于计时，可能产生约 17 GiB 文件
MTL_CAPTURE_ENABLED=1 .venv/bin/python scripts/capture_action_metal.py \
  --output records/metal_action_repeat

# 端到端配对：本轮基线 297.704 ms，候选 298.048 ms
.venv/bin/python scripts/conditioning_experiments.py \
  --output records/conditioning_repeat --iterations 40

# 完整精度/动态输入验证，不用于稳态计时
.venv/bin/python scripts/run_conditioning_optimized.py \
  --mode conditioning --entry validate_cases \
  --output records/conditioning_validation_repeat
```

后续首先使用 Xcode 打开现有 capture 分析依赖与 kernel；无需重新下载模型。分段 KV 自定义 Metal、RMSNorm/RoPE 融合本轮未实现，不能将这些方向描述为已测提速。

完整验证结果：51 组动作、归一化动作、token IDs 和额外扰动/长度切换输出，共 162 个对比文件，全部与原 MLX 逐值一致。
