# 当前裁剪版 MLX 的下一步优化分析

后续更新：VAE GN+SiLU 融合已接入默认，正式稳态 287.763 ms；完整精度、各候选结果与回退方式见 [本轮适配报告](README_NEXT_OPTIMIZATIONS_RESULTS_20261103.md)。

分析依据：远端当前 runtime.py/model.py/vae.py 与本地 SHA256 一致；默认启用 action_only 权重。性能采用 records/compact_action_benchmark_20261103/benchmark.json，不使用不同轮次历史均值计算加速比。本轮未改动推理实现或执行新的性能实验。

## 热点定位

50 次稳态端到端平均 304.580 ms，中位数 297.309 ms，P95 342.878 ms。另 5 次同步模块 profile 平均：

|模块|ms|占模块时间总和|
|---|---:|---:|
|视觉前缀|115.504|36.8%|
|10 步动作去噪|88.288|28.2%|
|VAE 编码|55.092|17.6%|
|Qwen 编码|47.594|15.2%|
|预处理|6.397|2.0%|
|分词|0.440|0.1%|
|输出处理|0.205|0.1%|

这些 profile 主动同步，不能当作正常流水的精确端到端分摊。文本编码单轮从 28.08 到 66.89 ms，系统波动明显。必须采用同期 AB/BA 配对，不能把过去约 286 ms 与本轮约 305 ms 直接比较。

## 建议实施顺序

### 1. 在当前裁剪版复测精确 RMSNorm + RoPE 融合，再扩展到视觉前缀

当前 runtime.RopeWeights.qkv 仍调用独立 rms 与 fused_rotate。已存在四元素归约分组 + precise rsqrt 的实验内核，仅动作分支已有完整 162 数组一致性证据；历史配对节省约 0.950 ms，尚未作为默认，也没有验证与本轮裁剪的组合。

先复测组合，再为视觉双流/单流 QK 以及末层 rotate_key 增加对应融合入口。视觉 token 比动作 token 多，可能有更大的中间数组读写收益，但没有实测速度保证。不得直接套动作实验后宣称视觉也已覆盖。重点保持 FP32 归约顺序、rsqrt、BF16 归一化与缩放舍入边界、非连续张量 strides；检查所有层 K/V，而不仅最终动作。

这是最接近可落地的方向，预计是小幅改进，不能承诺大幅加速。

### 2. VAE GroupNorm 后处理、激活与布局

vae.py::gn 显式执行 FP32 转换、mean、差平方、第二次 mean、rsqrt、仿射、BF16 回写；res 随后又执行 BF16 SiLU 再卷积。共有 10 个残差块中的 20 次 GN，加中间 attention 的 GN 和最终 GN，共 22 次。

优先保留原生 mean/variance 归约，只融合归一化后处理与 SiLU，显式保留原有 BF16 舍入位置。可先做单 residual block 的分解计时，再与完整在线三图推理对照。如果 trace 证明归约才是主要开销，再尝试符合原归约顺序的 GN 内核。

已有“整个编码器编译”“强制 contiguous”“fast RMSNorm”未给出兼顾稳定速度和逐位一致的结果，不重复直接切换这些开关。GN 不能简单换成 RMSNorm，方差也不能随意改成 E[x²]-E[x]²。

### 3. 分段 K/V 的专用 attention 内核

runtime._step 每层执行两次 concatenate，25 层 × 10 步 = 500 次逻辑 K/V 拼接。前缀缓存已经跨去噪步共享，仍重复的是将前缀与本步动作 K/V 拼成连续输入；不是缺少前缀缓存。

典型前缀 328 tokens、24 heads、128 head_dim、BF16。每层前缀 K/V 大小约 3.844 MiB；250 次 attention 对应拼接输出中约 960.94 MiB 重复前缀数据。此数值是形状推算，不是测得的 DRAM 流量，不能直接换算耗时或加速比。

目标内核直接读取 prefix K/V 与 action K/V 两段，在同一个 softmax 归一化中完成 attention。两次分别归一化的 attention 相加不等价。也不能跨层共享 K/V：每层投影及输入不同。预分配缓冲区后 Python 切片赋值不保证原地更新/零拷贝，需验证实际图与内存行为。

已有分段普通算子版没有提速且有数值差异，因此真正的新工作是专用 Metal/SDPA 实现，不是再次拆成两次普通 matmul。难度和精度风险高于前两项；注意保持 attention 中间精度及归约语义。先 kernel 级检查再跑动作逐步轨迹，若不能满足精度约束，保留实验模式。

### 4. 用 GPU 时间线选择视觉 GEMM 前后的融合边界

视觉前缀最大，但不能仅由模块耗时断言是带宽受限。应区分 GEMM、attention、RMSNorm/RoPE、调制/激活、layout copy 和 CPU 调度。根据结果再考虑 norm + modulation、门控激活 + 拼接、输出 residual 的专用融合；先检查 mx.compile 已融合哪些点操作，避免重复造轮子。

服务器当前仍无 xctrace/Xcode。之前保存的约 17 GiB action capture 不是实际内存流量；只有捕获而没有读取 kernel 时间线。先重放已有 capture 或配置可用的分析环境，再为当前真实十步中的代表步捕获；旧 steps=1 capture 不代表十步内对应噪声时刻。

## 降低优先级的方案

- 两步/五步去噪联合编译：历史实测略慢，单步编译 + async_eval 已启用。
- 时间调制缓存：已经测试，约慢 0.344 ms，收益区间跨零。
- 更大范围 QKV 合并：历史 14/51 动作有差异，并增加约 810 MiB 投影副本；不满足当前精度目标。
- 跨请求文本/图像特征缓存：改变测量口径且可能污染变化输入，不作为通用优化。
- 减少去噪步数、缩短文本、降图像分辨率、量化：改变当前算法或精度，不纳入本轮等价优化。
- shapeless 编译/按形状预热：主要针对首次形状延迟，不能直接当作稳态加速；不通过截断或固定 token 长度实现。

## 接入默认代码的门槛

每个候选单独比较当前 compact 默认基线：先模块输出，再 51 用例/162 文件，以及动作长度 1/8/16/32/64、不同步数、31→79→短→193 tokens 的输入变化。保存中间层 K/V、latent、逐步动作与最终输出。使用同期 AB/BA 多轮性能测量，单独报告首次编译、稳态、模块同步 profile。只有输出符合精度要求且收益可复现，才默认启用。

官方依据：[MLX 自定义 Metal 内核](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)说明 strides 支持和默认连续化行为；现有 RoPE 已设置 ensure_row_contiguous=False。新融合应正确处理 strides，避免引入隐式复制。[MLX Metal Debugger](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html)说明捕获与 Xcode 重放流程。实现细节应以服务器安装的 MLX 0.32.4 为准。
