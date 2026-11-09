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
