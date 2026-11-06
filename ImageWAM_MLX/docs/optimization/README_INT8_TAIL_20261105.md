# 视觉 INT8 尾块优化：正式适配与真实性能（2026-11-05）

已适配到 m5pro `/Users/qklj/work/cx/ImageWAM_MLX` 的 `--profile speed` 默认路径。accuracy 路径保持原实现。本轮只合入有实测收益的视觉尾块方案；软件流水、KV 物理复制审计及独立进程慢态根因仍待后续工作，未宣称已解决。

## 实现

新增 `imagewam_mlx/tail_visual.py`，在原有 INT8 矩阵乘的一次 dispatch 中，通过线程组一致分支将不超过 32 行的最后一个行块使用 BM32 计算，其余块使用原 BM64。保持 128 线程、原 64 行激活打包布局、K 累加顺序及输出缩放；尾块寻址/步进仍按原布局，不新增激活重排或单独尾块 dispatch。不改变权重、不增加完整权重反量化、不缓存输入特征。

按实际行数选择分支，文本不截断；余数为 0 或大于 32 时回退原内核。沿用既有 applegpu_g17s / MLX 0.32.4 环境保护，只有完整 GateQuantVisual 路径进一步启用 TailVisual；关闭上游优化时不会强行启用。量化信息记录 `tail_rows=32, tail_layout_rows=64`。

## 环境与计时边界

Apple M5 Pro，15 核 CPU、16 核 GPU，24 GiB 统一内存；MLX 0.32.4、Python 3.12.13，macOS 27.2。没有独立显存。沿用 `MLX_MAX_OPS_PER_BUFFER=256`。未终止其他用户/系统服务。

真实 RoboTwin 三图、状态、39-token 文本、H=16、10 步。端到端从已解码 RGB/状态/文本到 CPU 动作输出，包含在线文本及图像编码，不包含磁盘读取与模型加载。所有慢样本保留。首次推理包含本进程首次前向，但不代表清空系统缓存后的冷启动。

## 三层性能验证

1. 同进程随机配对 100 轮：旧版 198.188 ms，完整尾块 196.959 ms，节省 1.229 ms，配对 bootstrap 95% 区间 [1.170, 1.292] ms。仅 linear2 使用尾块反而慢 0.392 ms，未采用。
2. 六个独立进程，每个预热 10 次、稳态 100 次，顺序旧/完整/局部/局部/完整/旧：旧版 198.353 ms，完整候选 197.010 ms，节省 1.343 ms（0.68%）。局部候选 199.065 ms，未采用。
3. 集成后调用正式 `scripts/benchmark.py`，旧/新/新/旧四个独立进程，分别预热 10 次、稳态 100 次、模块诊断 5 次。逐块均值为 197.813 / 196.695 / 198.281 / 200.541 ms。存在时间漂移，但前后对照均支持完整尾块提速；没有声称消除背景负载影响。

正式 CLI 合并结果（每种版本 200 次）：

|项目|回退旧版|新默认|
|---|---:|---:|
|稳态端到端均值 ms|199.177|197.488|
|稳态端到端 P95 ms|201.838|200.176|
|模型加载平均 s|3.121|3.096|
|首次推理平均 ms|827.795|833.828|
|文本编码诊断 ms|27.837|27.639|
|图像预处理诊断 ms|2.404|2.492|
|图像编码诊断 ms|27.905|27.900|
|视觉 prefill 诊断 ms|83.951|82.637|
|动作十步诊断 ms|62.578|62.796|

本轮正式端到端节省 **1.689 ms（0.85%）**。首次推理没有改善；加载差异不作为优化收益。模块诊断带同步，会改变重叠关系，不可直接相加当作端到端。没有证据支持几十毫秒级收益。

## 精度与产物

实验完整/局部候选各通过 51 组真实输入逐值比较；另有 4 组动态边界测试：39 tokens/H1/1步、39/H8/3步、189/H16/10步、39/H64/11步。集成后的默认入口另执行 55 组回归，结果见 `records/int8_tail_20261105/default_validation/results.json`。正式 CLI 400 次稳态动作及 normalized 输出全部逐值一致。

比较对象是上一版量化 speed，不是 PyTorch/accuracy；不能据此声称原模型精度无损或 RoboTwin 闭环成功率不变。

记录目录 `records/int8_tail_20261105/` 包含 paired、abba、dynamic、default_validation、四组 cli_* 原始采样、每次输出、trace、源码快照和加载/首次耗时。实验脚本位于 `experiments/int8_tail_20261105/`，正式入口不依赖实验脚本。不要使用旧实验 worker 当作集成后的旧版基线，应使用正式回退开关。

## 启动命令

在 `/Users/qklj/work/cx/ImageWAM_MLX` 运行，输出目录需使用新的唯一名称：

```bash
# 新默认：本轮两进程合并 197.488 ms
.venv/bin/python scripts/benchmark.py --profile speed \
  --reference examples/robotwin_ep0_frame0 \
  --output records/tail_default_retest_01 \
  --warmup 10 --iterations 100 --profile-iterations 5

# 同轮旧版对照：199.177 ms
.venv/bin/python scripts/benchmark.py --profile speed --no-tail-visual \
  --reference examples/robotwin_ep0_frame0 \
  --output records/tail_disabled_retest_01 \
  --warmup 10 --iterations 100 --profile-iterations 5

# 真实输入推理；耗时以该次 run.json 为准
.venv/bin/python scripts/infer.py --profile speed \
  --inputs examples/robotwin_ep0_frame0/inputs.npz \
  --prompt-file examples/robotwin_ep0_frame0/prompt.txt \
  --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy \
  --output records/tail_infer_01 --warmup 1
```
