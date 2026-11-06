# KV 工作区正式适配与结构优化结果（2026-11-04）

已将 KV 工作区适配到正式 speed 运行时，accuracy 默认不启用。原始模型权重/配置未修改。实现位于 `imagewam_mlx/workspace.py`；每次请求重新创建工作区，仅在本请求的去噪步骤内传递 KV，下一请求不复用旧图像/文本特征。`slice_update` 是函数式操作，未取得底层原位复用证据，不宣称零拷贝或已消除全部前缀搬运。

## 正式入口复测

同一模型实例、共享同一份权重，切换 kv_workspace=False/True，100 组 AB/BA，双方各100次正式稳态，预热后计时。GPU 在每次端到端计时边界同步；模型内部正常使用 async_eval。不计加载/文件解码/保存；包含在线三图预处理、分词、编码、视觉前缀、16×14动作的10步去噪、CPU反归一化输出。

|模式|平均 ms|
|---|---:|
|原 speed，关闭工作区|242.950|
|新 speed，启用工作区|242.085|

平均节省 **0.865 ms（0.356%）**，逐对 bootstrap 95% 区间 [0.760, 0.971] ms。Bootstrap 10000次、seed42，未校正时序相关性和多重比较；只证明本轮小幅收益，不保证所有场景。先前实验入口60组配对节省0.731ms，方向一致。

## 验证

- 原生 speed 新旧路径51个真实样本，覆盖全部已观测31–79 token长度（48种，缺78）和3个额外真实观察，动作与归一化数组全部逐值一致。
- 筛选额外验证193-token文本、horizon8/16/32；原生验证另覆盖horizon1/64、3步及trace中间量，逐值一致。
- accuracy 默认 kv_workspace=False；基准样本最终动作与已保存 accuracy 参考逐值一致。
- 不是全数据集或闭环机器人成功率验证；speed 原有量化误差保持，不能把“相对旧speed一致”称为“相对BF16无损”。

## 启动和回退

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
RUN_TAG=$(date +%Y%m%d_%H%M%S)
.venv/bin/python scripts/infer.py --profile speed \
  --inputs examples/robotwin_ep0_frame0/inputs.npz \
  --prompt-file examples/robotwin_ep0_frame0/prompt.txt \
  --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy \
  --output records/$RUN_TAG-speed
```

该配置本轮稳态242.085ms；单次CLI请求包含自身预热流程，不能直接视为稳态平均。run.json及benchmark报告新增kv_workspace字段。`--profile accuracy`仍走原精度路径。

Python显式回退：`ImageWAM('weights/bf16','assets',profile='speed',kv_workspace=False)`。原生构造参数默认为None，按profile选择：speed启用、accuracy关闭；legacy关闭。两条路径仍支持动态文本和动作长度。

## 未采用的候选与后续范围

- Qwen编译：基准配对有小幅收益，但动态输入有数值差异，未默认启用。
- 视觉addmm：筛选未显示收益；不是完整GEMM+gate自定义epilogue。
- VAE NHWC统计：显著变慢且有数值差异，未采用。
- 原生双段KV attention、视觉GEMM+gate epilogue和W1内核解包复用仍是后续工作，本轮没有实现或宣称完成。

环境为M5 Pro（15CPU/16GPU，24GiB统一内存），macOS27.2，MLX0.32.4。当前没有kernel时间线，不能把收益精确归因于实际减少的DRAM字节数。

记录全部位于 `records/next_20261104/`：screen是候选筛选；paired是实验配对；workspace_validation是实验精度；native_paired是正式入口100组配对；native_validation是原生精度及trace；before是修改前代码；final_source是最终源码；source_hashes.json是哈希。下方保留原筛选报告，其中“默认未修改”只描述筛选当时状态。

---

# 结构性优化适配与筛选（2026-11-04）

在 M5 Pro 的现有 speed 路径基础上适配四项独立候选，全部通过真实串行端到端运行测试；未修改原模型权重、配置与默认推理入口。

## 实现与范围

|候选|实现|尚未证明的部分|
|---|---|---|
|workspace|每次请求分配带动作尾部的 25 层 KV；单步用 slice_update 更新动作区，并将更新后的 cache 返回下一步|函数式更新不保证底层原位写入或无完整复制，不能称为已实现零拷贝|
|qwen_compile|对现有 Qwen 整体调用增加 mx.compile，保持原始投影及全部 27 层|不是合并 QKV、不是减少层数，也不是 shapeless 编译|
|visual_addmm|视觉线性层改为 mx.addmm(bias,x,weight.T)，给后端选择 GEMM+bias 路径|不是已经实现 GEMM+SiLU/gate 的自定义 epilogue；当前权重无 bias 的层实际仍走原 matmul|
|vae_nhwc|直接在 NHWC 分组视图归约 mean/variance，避免 GroupNorm 前后的 transpose/reshape 重排|增加 FP32 统计/中间张量，未承诺减少实际搬运；允许与 speed 原归一化有舍入差异|

视觉自定义 GEMM+gate epilogue、原生双段 KV attention、动作 W1 内核解包复用仍未实现；本轮先验证已有 MLX 算子可完成的结构调整，没有把这些后续项目标记为完成。

## 环境与协议

M5 Pro，15 CPU/16 GPU，24 GiB 统一内存；macOS 27.2，Python 3.12.13，MLX 0.32.4。每配置独立进程、首次前向、3 次预热、20 次稳态及3次独立同步模块剖析；前后各一组 speed 基线。每请求重新处理真实三图、状态和完整文本，无旧输入特征缓存；16×14 动作，10步。端到端包含 CPU 输出，不含模型加载、文件解码和保存。模块同步改变调度，不能直接求和。

|配置|端到端均值 ms|P95 ms|加载 s|首次 ms|文本 ms|VAE ms|视觉 ms|动作 ms|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|00_baseline|241.553|242.163|2.834|1049.291|27.406|27.700|112.504|76.430|
|01_workspace|240.543|241.191|2.670|941.039|27.357|27.779|112.445|75.106|
|02_qwen_compile|240.089|240.335|2.701|1398.850|25.971|27.732|112.671|76.160|
|03_visual_addmm|241.567|241.984|2.685|864.693|27.325|27.767|112.612|76.310|
|04_vae_nhwc|278.713|285.188|2.685|3048.898|46.561|41.776|115.317|76.734|
|05_baseline|242.100|242.785|2.691|851.769|27.733|27.781|113.529|76.641|

## 数值验证

与本轮 speed 基线比较，不是与 accuracy 或 PyTorch 真值比较。基准输入外，还验证 39-token/horizon8/3步、193-token/horizon16/10步、39-token/horizon32/3步；所有输出 shape 正确且有限。

|配置|四组 action/normalized 是否全部逐值一致|最大动作绝对差|
|---|---|---:|
|00_baseline|True|0.00000000|
|01_workspace|True|0.00000000|
|02_qwen_compile|False|0.01600134|
|03_visual_addmm|True|0.00000000|
|04_vae_nhwc|False|0.01600134|
|05_baseline|True|0.00000000|

## 启动

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
RUN_TAG=$(date +%Y%m%d_%H%M%S)
.venv/bin/python experiments_next_20261104/bench.py \
  --case workspace --iterations 50 --output records/$RUN_TAG-workspace
```

--case 可替换 baseline、qwen_compile、visual_addmm、vae_nhwc；端到端参考值见上表。输出目录必须不存在。记录目录 `records/next_20261104/screen/` 包含逐次时间、模块、首次/加载、输出及动态输入 noise；`summary.json` 保存数值对比和源码哈希。

筛选不代表可靠的因果收益；只有具有足够幅度且复测通过的方案才升级默认。默认 accuracy 与现有 speed 均保持原样。
