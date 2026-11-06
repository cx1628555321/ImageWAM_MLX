# 默认 VAE 融合版之后的优化机会

最新适配：默认已启用 CPU/GPU 重叠；组内复用与 Qwen RoPE 提供可选开关。结果、精度与命令见 [本轮适配记录](README_REMAINING_ADAPTATION_20261103.md)。

基线为 compact + fused VAE，独立稳态 287.763 ms，P95 291.353 ms。本轮仅进行代码与环境分析，未修改推理路径，也没有新增加速实测。

## 保持现有在线输入与数值要求的优先顺序

1. **CPU 图像预处理与 GPU 文本编码重叠。** runtime.predict 目前先预处理三图，再分词和构建编码图，直到动作循环才提交异步求值。可以先分词并提交 Qwen async_eval，同时在 CPU 做 PIL resize/拼图，然后依赖正确的文本/图像结果进入前缀。先试单 GPU stream，仅隐藏 CPU 工作，不必立即引入两个 GPU stream。不能把预处理移出计时区间或复用上一帧图像。

2. **VAE 每组逆标准差只计算一次。** fused_vae.py 当前 kernel 每个元素执行 precise::rsqrt(var[group]+eps)，同组数值相同。候选是保留原 mean/variance，预先求 32 个 group 的 inverse std，再由融合 kernel 读取；或在 threadgroup 内复用。前者增加一次小 kernel，后者受 NHWC 组布局影响，必须实测净收益。不要把 inverse std 乘 scale 后提前量化，也不要改变 FP32 乘加顺序。

3. **Qwen 专属 RoPE 点算子融合。** qwen.py 中 Q/K norm 和非交错 RoPE 仍分开执行，每层都有切分、拼接、逐点乘加。先只融合 RoPE、保持原 RMS 归约和 Q/K/V GEMM 形状，显式保留 BF16 乘法与加法舍入；再考虑融合 Q/K RMS。不能照搬 FLUX 的相邻偶奇旋转，Qwen 使用前半/后半配对；Q/K 分别为 32/8 heads。也不能用先前 QKV 合并有差异的结果声称这个新方案已验证。

4. **内存驻留与尾延迟。** 当前代码没有设置 wired limit。只读查询得到总内存 25,769,803,776 字节（24 GiB），建议 GPU 工作集上限 19,069,665,280 字节（约 17.76 GiB）。系统当前 swap used 1734.69 MiB，是全系统存量，不是推理期间换页证据。下一步在相同负载下采集 active/peak memory、压缩/换页计数增量与 P50/P95/P99；只有证实相关性才测试不超过系统上限的进程级 set_wired_limit，退出恢复原值。不先修改系统 sysctl。该项主要可能改善抖动，未必降低无内存压力时的平均延迟。

5. **精确形状编译缓存/预热。** 保持原 token 长度不截断，对已观测形状预热并记录首次形状延迟，限制缓存占用。当前 prefill/Qwen 的 Python shape 和 NumPy arange 参与图构建，不能直接打开 shapeless=True 就假定支持任意文本长度。此项针对首次/切换延迟，不等同稳态加速。

## 业务模式可选：精确文本缓存

机器人一条指令常连续对应不同观测。仅当实际 token IDs、模型版本、tokenizer/模板及 dtype 等均相同，允许复用确定性的 Qwen 输出。三图、状态、视觉前缀和动作仍每次计算；文本一变就失效。用完整 token 序列校验，限制缓存字节数，不用长度作为缓存键。

这可以减少重复指令场景的在线文本编码，但改变了“每次均在线编码”的基准口径。必须独立报告命中/未命中/切换及命中率，不能拿命中延迟冒充通用端到端成绩。若维持当前完全无特征缓存协议，此项不启用。

## 暂不重复推进

- 当前简单分段 attention 原型比原生 SDPA 明显慢且数值不符；只有采用更好的 tiling/向量化、并取得 GPU 数据后才值得重做。
- 大范围 RMS/RoPE 融合、视觉门控拼接、时间调制缓存和多步联合编译，已有收益不稳定或负收益证据。
- 降步数、量化、低秩、减小输入不符合本轮严格等价目标。
- VAE 最终 norm_out 的 GN+SiLU 尚可扩展融合，只有一处，优先级低于上述重复计算；中间 attention 的 GN 后没有 SiLU，不能直接套同一融合。

## 验证要求

每项单独对照当前默认版：162 个数组、真实观测中间量、动作长度与步数变化、长短文本切换；性能使用同期 AB/BA，至少两轮稳定收益。异步调度与缓存还需专门检查连续不同输入，避免错帧和旧文本复用。未取得真实 GPU 时间线前，不把墙钟分解视为 kernel 或带宽瓶颈证明。

官方依据：[MLX streams](https://ml-explore.github.io/mlx/build/html/usage/using_streams.html)、[维护者关于多 stream 的说明](https://github.com/ml-explore/mlx/discussions/2083)、[wired limit](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html)、[编译与 shapeless 限制](https://ml-explore.github.io/mlx/build/html/usage/compile.html)。多 GPU stream 不保证算力受限工作负载提速；编译并不保证浮点逐位一致。
