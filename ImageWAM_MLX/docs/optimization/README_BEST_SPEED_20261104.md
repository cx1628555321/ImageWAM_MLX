# 当前最佳速度版本：固化与回归（2026-11-04）

## 正式使用

`scripts/infer.py --profile speed` 在已验证的 M5 Pro / MLX 0.32.4 / action W1 环境自动启用全部已获益路径，不依赖任何 experiments 目录。`--profile accuracy` 仍是默认精度优先路径。`--no-action-epilogue` 可复现固化前的 speed。新自定义内核只在已验证环境启用，其他环境/动作量化策略自动回退，run.json记录实际状态及原因。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
RUN_TAG=$(date +%Y%m%d_%H%M%S)
.venv/bin/python scripts/infer.py --profile speed \
  --inputs examples/robotwin_ep0_frame0/inputs.npz \
  --prompt-file examples/robotwin_ep0_frame0/prompt.txt \
  --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy \
  --warmup 3 --output records/$RUN_TAG-best-speed
```

仅改为 `--profile accuracy` 即运行精度路径；speed命令追加 `--no-action-epilogue` 即回退旧speed。输出目录拒绝覆盖。示例固定噪声用于复现对比，正常使用可不传noise并指定seed/horizon；文本长度保持动态，无截断或固定文本长度。

正式性能入口：

```bash
RUN_TAG=$(date +%Y%m%d_%H%M%S)
.venv/bin/python scripts/benchmark.py --profile speed \
  --reference examples/robotwin_ep0_frame0 \
  --warmup 4 --iterations 30 --profile-iterations 5 \
  --output records/$RUN_TAG-best-benchmark
```

## 已固化的有效优化

- 保留已有 action-only 权重/冗余层裁剪、动态文本、RoPE与VAE融合、文本/图像编码调度重叠、编译前缀与单步动作图。
- speed保留 fast BF16 VAE、动作65个合格Linear的W1/group64打包权重、BF16激活、请求内KV工作区。
- 新增正式动作W1 Split-K＋门控/残差输出融合；FP32部分和归约后保留BF16舍入，不执行完整权重反量化。
- 自适应矩阵块：合格动作Linear在长度14–16使用BM16/BN64/WM1/WN2；长动作保留BM32/BN64/WM2/WN2；短动作/非合格层回退原运算。目标线程组512，实际分块按M/N/K及量化组对齐确定。
- 未引入无收益候选：双段KV、门控拼接融合、向量归约、过高Split-K并行度、此次视觉行padding。

正式实现：imagewam_mlx/action_epilogue.py、action_kernel.py及metal/epilogue.h。上游MIT许可证和哈希同目录保留。原实验代码和模型权重未覆盖。

## 环境与协议

Apple M5 Pro，15CPU核/16GPU核，24GiB统一内存（GPU共享，不另有独立显存），推荐GPU工作集19069665280字节；macOS27.2、Python3.12.13、MLX0.32.4，架构applegpu_g17s。

100对AB/BA交替，双方各首次+4次预热后计时；同一权重，旧动作实现与新正式动作实现分别编译。真实三图、39token、16×14动作、10步，所有编码在线重新计算，无跨请求特征缓存。端到端含CPU动作输出，不含模型加载、文件读取和保存。

|路径|稳态端到端均值 ms|
|---|---:|
|旧speed|240.778|
|新正式speed|226.693|

降低 **14.085 ms（5.85%）**；配对bootstrap节省95%区间 [13.990, 14.181] ms。区间仅反映本轮样本，不覆盖所有系统负载。

正式CLI单次检查：加载 2.768 s、首次前向 1002.373 ms、预热后请求 226.463 ms。加载不是冷磁盘保证；首次前向包含该进程初始化/编译。上述配对的first值共享编译缓存，不能用于冷启动对比。

## 同步模块计时（各5次均值）

|模块|旧speed ms|新speed ms|
|---|---:|---:|
|tokenization|0.172|0.150|
|text_encoder|27.388|27.315|
|preprocess|2.375|2.464|
|image_encoder|27.814|27.829|
|visual_prefill|112.329|112.264|
|action_denoise|75.360|61.286|
|postprocess|0.171|0.161|

同步profile会改变重叠调度，不能将各模块直接求和当成端到端。

## 精度与回退

51组真实样本/文本长度案例，正式新speed与上一版最佳实验全部逐值一致；动作长度1/8/14/15/16/32/64同样一致。accuracy对原保存参考输出逐值一致。正式CLI输出也与新speed回归输出逐值一致。

相对固化前旧speed，本样本最终动作 max abs=0.01600122、RMSE=0.00255085。此前51组与旧speed的最大差为0.02261293。speed是速度优先分支，不宣称与accuracy/PyTorch精度等同，尚无RoboTwin闭环任务成功率验证。

## 继续优化：视觉前缀

当前最大同步模块为视觉前缀约112ms，其次动作约61ms，文本/VAE各约27–28ms。因此转向视觉大矩阵，避免重复微调低占比归约。

本轮已试验：仅在视觉Linear内部，把实际行数>=128的矩阵临时补到32/64倍数，计算后立即裁回原长度；不在attention中增加token，不改变输入图片或文本内容。每配置4次预热、40轮随机顺序完整请求：

|视觉行对齐|端到端 ms|相对同轮基线节省 ms|最终动作max abs|
|---|---:|---:|---:|
|0|226.981|0.000|0.00000000|
|32|228.804|-1.822|0.00000000|
|64|234.675|-7.694|0.00000000|

0表示不额外对齐。该候选没有稳定收益，未合入正式版；额外行计算与padding开销没有被更规则的矩阵块抵消。这里只能确认净效果，尚无GPU时间线分解成本。

后续优先级：
1. 捕获视觉重层真实M/N/K及耗时，针对实际GEMM布局/线程块调优；不再笼统地将所有视觉权重量化，已有实验表明更慢。
2. 针对视觉输出投影研究矩阵乘epilogue融合，保留高吞吐矩阵内核，避免为了融合切换成低并行路径。
3. 用31–79token实际长度及193token扩展输入验证任何候选，保持请求内复用边界和数值记录。

以上是下一阶段方向，不计入已取得收益。代码、依赖哈希、性能逐次采样与最终动作保存于records/best_native_20261104；修改前入口备份在before/。

## 独立正式benchmark核验

加载 2.752 s，首次 996.827 ms；预热4次，稳态30次均值 226.641 ms、P95 227.153 ms。正式入口 action_epilogue=true；旧速度回退入口 action_epilogue=false 且输出与旧speed回归逐值一致。

相对 examples/robotwin_ep0_frame0/reference_action.npy：max abs=1.80814493，RMSE=0.51819539，cosine=0.61043662。这是激进W1速度分支的显著精度差异，不能用与既有实验版逐值一致来替代该精度比较。精度优先时继续使用accuracy。
