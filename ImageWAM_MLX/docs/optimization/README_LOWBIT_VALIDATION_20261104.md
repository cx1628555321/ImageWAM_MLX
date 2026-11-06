# M5 Pro 真实低比特算子与全模型验证（2026-11-04）

本轮将原生 INT8 GEMM、激活量化边界融合、INT8 KV/attention、量化卷积做成可运行实验，并完成单算子及完整模型验证。默认 accuracy 和现有 speed 代码保持原样：实验需要用真实端到端收益决定是否采用，不能仅凭位宽或单算子局部结果升级默认配置。
复测决策：现有 speed 平均 244.458 ms，相对 accuracy 的 288.153 ms 延迟降低 15.16%；加入 INT8 视觉后的 253.215 ms 比现有 speed 慢 3.58%。本轮不将新量化候选合入默认配置。这里的 accuracy→speed 收益来自既有优化，并非本轮新增收益。


## 环境与测试边界

Apple M5 Pro，15 核 CPU、16 核 GPU、24 GiB 统一内存，无独立显存；macOS 27.2、Python 3.12.13、MLX 0.32.4。原生 INT8 内核采用 Metal Performance Primitives TensorOps。全部文件保存在 `/Users/qklj/work/cx/ImageWAM_MLX` 下。

正式输入为 `examples/robotwin_ep0_frame0/` 三张真实相机图像、完整任务文本、状态及固定 FP32 初始噪声。端到端包括在线分词、三图预处理、文本编码、VAE 编码、视觉前缀、10 步动作去噪及 CPU 反归一化输出，默认 horizon=16、动作维度=14；不含加载、图片文件解码和输出保存。没有跨请求缓存文本/图像特征。

## 建议落实情况

|方向|实现与验证|结论|
|---|---|---|
|细粒度剖析|`profile_inventory.py`，真实各层输入捕获、同步 eager 算子归因、Metal action-step trace|trace 已保存，但缺少完整 Xcode/Instruments，尚未解析 GPU kernel 时间线；eager 时间不是编译图生产耗时|
|真实低比特 GEMM|`native_int8.py`，INT8×INT8→INT32，融合 FP32 scale/bias→BF16；5 组整数逐值验证|整数 GEMM 对 NumPy 整数参考最大误差为 0；与 MLX BF16、W1/W4/W8A16 比较完整在线激活量化成本|
|量化边界融合|`fused_quant.py`，LN/RMS/SiLU/gated/concat-gated→INT8+scale，视觉/动作可选接入|已测试融合路径；未证明整体提速。Qwen 只替换线性层，未宣称融合其全部 RMS；没有把所有残差边界都改成整数|
|量化 KV/attention|`quant_attention.py`，前缀 KV 量化一次，动作 KV 在线量化，INT8 QK/PV，保留浮点 softmax|支持当前完整动作路径；前缀仅在同一次推理内复用，不缓存旧输入|
|量化卷积|`quant_conv.py`，NHWC im2col+INT8 GEMM，1×1/3×3，计入展开和激活量化成本|真实 VAE 全卷积替换完成；不称为直接 INT8 convolution，完整模型验证不支持默认采用|
|只保留整体有效方案|12 组筛选，4 种配置正反顺序各 50 次复测|accuracy 仍默认；实验与稳定入口分开，原权重/配置未覆盖|

## 端到端复测

顺序：accuracy → current_speed → fused_visual → speed_native_visual → speed_native_visual → fused_visual → current_speed → accuracy。每轮独立进程，首次前向单独记录、再预热 3 次、计时 50 次；每配置共 100 次。模块剖析另测每进程 3 次。单机串行，不能据此保证所有负载和样本均有同样收益。

|配置|平均 ms|两轮均值 ms|对 accuracy 最大绝对误差|
|---|---:|---|---:|
|accuracy|288.153|289.019, 287.287|0.000000|
|current_speed|244.458|241.507, 247.410|1.808145|
|fused_visual|257.683|257.702, 257.664|0.012305|
|speed_native_visual|253.215|253.373, 253.058|1.808145|

current_speed 是原有动作 W1A16 + fast BF16 VAE 路径；fused_visual 是 BF16 动作 + fast VAE + INT8 视觉融合；speed_native_visual 是原有 speed 再加 INT8 视觉融合。不能把减少内存造成的整机行为变化误解为每个 INT8 内核都更快。

### 模块复测（独立同步剖析，ms）

|配置|文本编码|图像编码|视觉前缀|动作去噪|
|---|---:|---:|---:|---:|
|accuracy|48.156|42.845|118.797|81.759|
|current_speed|27.549|27.755|116.037|76.768|
|fused_visual|27.552|27.650|124.068|80.657|
|speed_native_visual|27.475|27.624|123.793|76.408|

模块同步会改变执行重叠及调度，不将模块求和代替端到端结果。

## 全部候选筛选记录

每组首次前向+3 次预热+12 次稳态；筛选数值不能替代上面的交替顺序复测。加载包括构建及一次性量化打包；首次耗时受系统/驱动缓存影响，非严格冷启动。

|配置|加载 s|首次 ms|稳态 ms|最大绝对误差|RMSE|
|---|---:|---:|---:|---:|---:|
|accuracy|5.157|22646.136|293.648|0.000000|0.000000|
|current_speed|2.844|1408.599|241.872|1.808145|0.518311|
|fast_vae|2.578|1455.965|266.526|0.008001|0.002168|
|native_action|3.041|3472.996|326.017|0.016001|0.004834|
|fused_action|2.781|1263.318|324.224|0.018860|0.004808|
|native_visual|3.769|7405.393|258.418|0.009573|0.002865|
|fused_visual|3.554|5082.865|257.744|0.012305|0.003218|
|native_all|4.906|1461.257|343.939|0.016001|0.004827|
|fused_all|5.526|467.452|345.358|0.016001|0.005021|
|int8_kv|2.570|1997.332|277.153|0.024002|0.005705|
|int8_conv|2.820|4163.251|303.932|0.008001|0.001527|
|all_lowbit|6.841|511.421|370.687|0.032003|0.006231|

## 输入可变性与精度边界

每种配置另跑 39-token/horizon8/3步、193-token/horizon16/10步、39-token/horizon32/3步，输出均有限且形状正确，输出和噪声独立保存。没有截断/固定文本长度；本轮实验没有重复完整 31–79 token 的 51 样本精度套件，因此不宣称新量化路径全数据精度已验证。默认 accuracy 的基准样本与已存 accuracy 参考逐值一致。量化误差仅为对保存的 MLX 浮点动作参考的数值偏差，不是机器人任务成功率，也不是 PyTorch 真值精度结论。

## 为什么全量低比特仍然较慢

1. 当前动作序列只有 16，真实输入的矩阵形状、padding、分块和调度成本决定结果；M5 TensorOps 支持 INT8 不等于本实验内核在这些形状下胜过 MLX 已优化内核。
2. 动态激活量化需要逐行 absmax/scale，融合内核的归约和同步仍有成本。LN、SiLU、gated 融合并未普遍获益；RMS 的微小单次收益不足以证明端到端收益。
3. INT8 attention 增加 Q/P 量化、尺度处理和两次 GEMM；此处较短 attention 下，复用 INT8 KV 仍未胜过原浮点 attention。
4. im2col 卷积产生 patch 展开和量化成本；局部 1×1 微基准略快不代表 VAE 整体快。
5. 本实现权重不在每次推理展开成完整 BF16；但归一化、softmax、调度、残差及模块接口仍有 BF16/FP32。它是低比特 GEMM 混合精度推理，不是全图 INT8，也没有声称当前 MLX 能把所有算子换成整数后直接加速。

真实算子逐项时延、矩阵尺寸及输出误差见 `gemm.json`、`boundaries.json`；微基准计入 dispatch/eval/synchronize，不能当作 GPU kernel-only 时间或直接累加。W1/W4/W8 使用 MLX packed-weight qmm，激活仍 BF16；本轮新增真正 W8A8，未新增原生 W4A8 TensorOps 内核。

## 启动与结果位置

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
# 现有精度优先路径（默认也是 accuracy）
RUN_TAG=$(date +%Y%m%d_%H%M%S)
.venv/bin/python scripts/infer.py --profile accuracy \
  --inputs examples/robotwin_ep0_frame0/inputs.npz \
  --prompt-file examples/robotwin_ep0_frame0/prompt.txt \
  --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy \
  --output records/$RUN_TAG-accuracy
# speed 使用同一命令，将 --profile accuracy 改为 --profile speed。

# 实验：CASE 可取上表配置名或 speed_native_visual
CASE=int8_kv
.venv/bin/python experiments_lowbit_20261104/e2e_case.py \
  --case "$CASE" --iterations 50 --output "records/$RUN_TAG-$CASE"
```

上述单次 infer 命令本身不是稳态基准；对应配置的稳态端到端耗时见复测表。实验程序拒绝覆盖已存在输出目录。`run_e2e.py`/`run_confirm.py` 固定使用本轮记录路径，重跑应改成新的记录目录，不能删除旧数据来复用目录。

- 实验源码：`experiments_lowbit_20261104/`；归档源码及 SHA256：`records/lowbit_20261104/source/`、`summary.json`。
- 单算子：`records/lowbit_20261104/{native_probe,gemm,boundaries}.json`。
- 完整筛选：`records/lowbit_20261104/e2e/<case>/results.json`、`output.npz`、`dynamic_*.npz`。
- 复测：`records/lowbit_20261104/confirm/<round_case>/`，保存逐轮时间、模块、输出、加载和首次耗时。
- GPU 捕获：`records/lowbit_20261104/profile/action_step.gputrace`，约 17 GB；捕获包含驻留 buffer，不等于 trace 计算量大。保留在服务器，未复制到本地。该项仍需完整 Xcode 才能解析 timeline，本报告不伪造 kernel 级耗时。

## 来源与许可

原生 GEMM 参考 [Cider](https://github.com/Mininglamp-AI/cider)，commit `4d91fcee9439f7aea17ae6e965271d9536c604a0`，MIT 许可证已保留在 `kernels/LICENSE.cider`。适配 BF16 epilogue、当前 MPP 编译类型、动态形状 padding、在线行量化及 head-batched attention。

技术依据：[MLX Metal debugger](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html)、[Apple Metal Performance Primitives programming guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)、[MLX 0.32.4 quantized backend](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/quantized.cpp)。
