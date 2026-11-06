# 当前最快配置与启动命令

> 2026-11-05后续更新：10步无缓存speed已默认启用INT8权重直读＋K循环展开。全量8进程对照201.544→198.652 ms（约1.43%），候选复核均值194.176 ms。详见[最新报告与启动命令](README_DEVICE_NAX_20261105.md)。下文缓存/4步数字是旧版本历史记录，本轮未重测，不能视为新内核性能。

更新：2026-11-05。依据已归档的真实测试，本次只核对和整理记录，未重新计时。

## 结论

**已测且可从正式入口启动的最快配置：平均 136.390 ms，P95 137.938 ms。** 参数为 `--profile speed --text-cache-size 8 --steps 4`，要求重复指令的文本缓存命中，并接受 4 步去噪带来的输出变化。这是稳态端到端均值，不是首次推理，也不是原始无缓存、10 步的速度。

|配置|平均端到端 ms|P95 ms|条件|
|---|---:|---:|---|
|缓存命中＋4步：最快可选|136.390|137.938|输出改变|
|无缓存＋4步|159.888|160.653|输出改变|
|缓存命中＋10步：保持当前 speed 输出一致的最快已测配置|172.664|173.014|重复指令，与当前10步speed逐值一致|
|默认无缓存＋10步|198.665|203.567|前后两个基线进程合并|

硬件：M5 Pro，15核CPU、16核GPU、24 GiB统一内存；macOS 27.2、Python 3.12.13、MLX 0.32.4。输入：RoboTwin真实三图＋状态＋39-token文本，H16。计时包含在线图像编码、动作计算和CPU动作输出，不含模型加载、磁盘读取与视频解码。最快配置预热8次后计60次，模型加载3.026秒，首次前向778.353毫秒；独立正反顺序复核同配置均值138.643/140.969毫秒，存在运行波动。

4步相对当前10步量化speed，在51组真实输入上的平均逐样本动作RMSE为0.382030，最大绝对误差1.392112。未验证RoboTwin闭环成功率。“逐值一致”仅指缓存10步相对当前speed，不代表相对原始PyTorch/accuracy无损。

## 最快配置：复现性能测试

在 m5pro 执行；时间戳创建新的输出目录，避免覆盖记录：

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
.venv/bin/python scripts/benchmark.py \
  --profile speed --text-cache-size 8 --steps 4 \
  --reference examples/robotwin_ep0_frame0 \
  --output "records/fastest_$(date +%Y%m%d_%H%M%S)" \
  --warmup 8 --iterations 60 --profile-iterations 3
```

对应历史实测：**136.390 ms**。新运行以其 `benchmark.json` 为准，不保证每次相同。

## 最快配置：实际输入推理

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
.venv/bin/python scripts/infer.py \
  --profile speed --text-cache-size 8 --steps 4 \
  --inputs examples/robotwin_ep0_frame0/inputs.npz \
  --prompt-file examples/robotwin_ep0_frame0/prompt.txt \
  --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy \
  --output "records/fastest_infer_$(date +%Y%m%d_%H%M%S)" \
  --warmup 1
```

该命令先执行一次预热（文本缓存 miss），再测一次命中请求；136.390 ms来自上面的多次benchmark，不是此单次命令的承诺。输出包括 `action.npy`、`normalized.npy`、`action.csv`、`run.json`。更换输入文件即可使用新三图/状态/完整指令；文本不固定长度。

缓存仅在同一模型进程内有效，退出程序后消失。实际服务需要复用模型实例；首次或新指令未命中时不会省去文本编码，图像与状态仍每次重新处理。默认最多8项、特征64 MiB；更换权重/策略后调用 `model.clear_text_cache()` 或重建模型。

## 保持当前10步输出一致

将上述命令的 `--steps 4` 改成 `--steps 10`，保留 `--text-cache-size 8`，对应缓存命中的实测均值 **172.664 ms**。如果不使用重复文本缓存，删除 `--text-cache-size 8`，并使用10步，恢复默认基线。

## 可追溯记录

* 最快正式测试：`records/remaining_20261105/cli_cache_steps4/benchmark.json`，包含逐次时间、首次/加载、模块诊断、源码快照及输出。
* 独立复核：`records/remaining_20261105/independent_summary.json`。
* 精度汇总：`records/remaining_20261105/precision_summary.json`。
* 完整说明：[全部方向汇总](README_ALL_DIRECTIONS_20261105.md)。

当前正式默认仍为无缓存、10步；本文件未将少步或缓存自动设为默认。

## 无缓存10步后续复测

2026-11-05专项复测：原路径197.313 ms，padding候选204.703 ms，未采纳为默认；历史198.665 ms基线保持有效。详见[无缓存10步报告](README_ONLINE10_20261105.md)。

## SIMD布局后续复核

当前默认本轮正式194.468 ms（200次，P95 196.212 ms）；与历史数字差异不算代码收益。双SIMD候选局部prefill更快，但正式端到端未优于基线，未默认启用。[完整报告](README_DEVICE_LAYOUT_20261105.md)。

## 版本启动与汇总索引（2026-11-06）

各版本实际推理命令、测试速度、优化点和输入说明见 [推理版本索引](../../README_INFERENCE_VERSIONS.md)。
