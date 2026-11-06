# 四个优化方向的适配、精度与性能结果

本轮在 M5 Pro `/Users/qklj/work/cx/ImageWAM_MLX` 按顺序实现并测试候选。默认代码只接入通过精度和重复性能验证的 **VAE GroupNorm 后处理 + SiLU 融合**。RMSNorm/RoPE、分段 attention、视觉门控融合保留实验，不默认启用。真实 GPU kernel 时间线仍未取得，详见工具限制。

## 同期性能比较

基线为上一轮 compact 默认实现，BF16、三图在线编码、动态文本、10 步动作去噪，不缓存图像或文本特征。每个候选单独比较，40 组 AB/BA；VAE 另做 80 组独立复测。计时前后 GPU 同步；不含模型加载、读文件和写结果。

|候选|基线 ms|候选 ms|平均节省 ms|节省的 95% 区间|结果|
|---|---:|---:|---:|---|---|
|动作 RMSNorm/RoPE|313.313|309.626|3.686|[-9.083,15.590]|精度通过，收益不确定，不默认启用|
|视觉+动作 RMSNorm/RoPE|299.452|325.009|-25.557|[-59.443,-2.333]|精度通过，变慢且尾延迟明显，不默认启用|
|VAE GN+SiLU，修正版|293.643|286.803|6.840|[6.181,7.490]|通过|
|VAE 独立 80 组复测|308.248|302.806|5.441|[4.334,6.436]|通过，约降低 1.77%，接入默认|
|分段 K/V 专用 Metal attention|306.194|453.371|-147.176|[-158.715,-135.894]|变慢且精度不符，拒绝|
|视觉门控+拼接，修正版|297.698|301.484|-3.786|[-5.178,-2.423]|精度通过但变慢，拒绝|

置信区间为配对差值的 10000 次 bootstrap，seed=42；未校正时间相关性或多重比较。不同轮次环境波动较大，因此只用同期比较判断收益，不将历史独立均值之差当作加速。

初版 VAE/视觉门控使用普通 sigmoid 表达式，均引入动作差异。检查本机 MLX unary_ops.h 后改为其原始稳定分支与 precise::exp，再通过完整验证。初版代码保留在 scripts/next_optimization_candidates_v1.py。

## 默认代码改动

新增 `imagewam_mlx/fused_vae.py`，在 10 个 VAE residual block 的 20 个 GN+SiLU 位置使用融合后处理。保留原生 FP32 mean/variance 归约，融合归一化、仿射、BF16 回写、稳定 sigmoid 和乘法，保持原舍入边界。其余 GN、attention、卷积与当前裁剪权重保持原逻辑。

`runtime.ImageWAM` 默认启用，原始 VAE 保留，可用 `fused_vae=False` 单独关闭；legacy 模式也不启用。复用原先已转换为 NHWC 的权重，不再次转置、不新增权重副本。

新增 `Weights.norm_rotate_key` 作为末层 K 融合实验接口，默认仍执行原 rms + rotate_key；它本身不改变算术。

## 精度证据

- 正式默认入口、VAE 候选、动作 RMS 候选、视觉+动作 RMS 候选、视觉门控修正版：各自 162 个保存数组与上一轮 compact 基线逐位一致，最大绝对差为 0。
- 包含 51 个离线用例、非单调文本长度切换、193-token 输入，以及三相机/状态/文本扰动。动态初筛还覆盖动作长度 1/8/16/32/64 及不同去噪步数。
- `vae_intermediates_verified` 明确使用 fused_vae=False 基线：4 个真实观测的 80 个 GN+SiLU 输出和 4 个 VAE latent 全部逐位一致。
- 视觉 RMS/RoPE 隔离诊断：单个真实观测的全部 25 层 K/V 与原实现逐位一致；完整样本最终输出另行验证。
- 分段 attention 初筛已出现差异，不进行默认部署，也没有将其宣称为无损。

这里的“精度通过”是相对现有 MLX 的所测离线数值一致性；不是与 PyTorch 绝对无差异、所有可能浮点输入的形式证明或闭环任务成功率测试。

核心记录在 `records/next_optimizations_20261103/precision_summary.json`，正式输出在 `default_validation`；中间结果在 `vae_intermediates_verified`。

## 正式默认版独立性能

M5 Pro：15 核 CPU、16 核 GPU、24 GiB 统一内存，GPU 共享内存而非独立显存；macOS 27.2、Python 3.12.13、MLX 0.32.4。

8 轮预热，50 轮稳态；完整三图/文本输入到 CPU 最终动作。

|指标|结果|
|---|---:|
|加载|2.607 s|
|首次推理（含编译）|1532.007 ms|
|稳态端到端平均|287.763 ms|
|中位数|287.752 ms|
|P95|291.353 ms|

额外 5 轮同步模块 profile：预处理 5.357 ms，分词 0.425 ms，文本 51.691 ms，VAE 42.978 ms，视觉前缀 116.104 ms，动作去噪 80.874 ms，输出 0.234 ms。同步 profile 的求和不等于正常端到端耗时。

本轮首次未清除 OS/驱动缓存，不是严格冷启动。实际 source_snapshot 与权重路径、compact/fused_vae 标记均保存在 `default_benchmark/benchmark.json`。

## 第四方向：视觉算子诊断与工具限制

完成视觉门控激活+拼接融合的 Metal 实现、修复、配对测试与完整精度验证，结果不值得默认启用。

另外执行串行 eager 算子诊断，缓存与默认编译路径一致。每次前缀有 86 次 linear、40 次 norm、59 次 RMS、59 次 RoPE、24 次 attention、29 次门控激活；其中 linear 的同步墙钟汇总约 140.13 ms，其他类别见 `visual_operator_profile/results.json`。此诊断强制每算子同步，包含调度、可能的依赖开销，**不是生产图 kernel 时间线，也不能拿其总和与 116 ms 的默认前缀直接比较**。

远端没有 Xcode/xctrace，但发现系统自带 `/usr/bin/gpudebug`。查阅系统 man 后，实际尝试 `profile run` 重放旧 capture 和查询会话：两者均无响应、无可见重放后端，随后终止本次启动的客户端。未取得有效 GPU counter/时间线。本地有 Xcode 16.2，也未将异机重放计时当作 M5 Pro 性能。因此“依据真实 GPU 时间线进一步选择 GEMM 融合边界”仍受工具限制，不能标为已完成。官方参考：[gpudebug 调试](https://developer.apple.com/documentation/xcode/debugging-with-interactive-command-line-tools)、[MLX Metal Debugger](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html)。

## 异常记录

最初 rms_action 与 rms_both 短暂重叠，已停止并排除；只采纳后缀 serial、precise、repeat、direct 的正式结果。rms_both_serial 曾长时间无结果并被终止，不纳入性能统计。随后 eager/直接编译隔离诊断通过，改用直接绑定 Base.prefill 编译入口得到 rms_both_direct 有效结果。最初阻塞根因未最终确定，不将其笼统归因于 MLX 内核。

一次 SCP 连接中断后已重传。为了消除脚本版本不确定性，中间量最终采用独立 `vae_intermediates_verified` 目录和显式关闭融合的基线复核。

## 启动与复现

工作目录为 `/Users/qklj/work/cx/ImageWAM_MLX`。以下复测输出目录必须不存在。

```sh
# 正式默认：本轮稳态 287.763 ms；compact + fused VAE。
.venv/bin/python scripts/benchmark.py \
  --reference ../ImageWAM/evaluate_results/m5pro_robotwin/20261103_dynamic_profile \
  --output records/next_default_repeat --warmup 8 --iterations 50
.venv/bin/python scripts/validate_cases.py --output records/next_default_validation_repeat

# 当前实验脚本显式关闭默认 VAE 融合，以保持原 compact 比较基线。
.venv/bin/python scripts/test_next_optimizations.py \
  --mode vae_gn_silu --pairs 80 --output records/vae_fusion_repeat
# --mode 还可选 rms_action / rms_both / segmented_attention / visual_gate_concat。
# 各候选实测端到端耗时见上表；非默认候选不代表建议生产使用。
.venv/bin/python scripts/run_next_candidate.py --mode rms_both \
  --entry validate_cases --output records/rms_both_validation_repeat
.venv/bin/python scripts/validate_vae_intermediates.py \
  --output records/vae_intermediate_repeat
```

保持 compact 与其余当前优化，仅关闭新 VAE 融合：

```python
from imagewam_mlx import ImageWAM
model = ImageWAM('weights/bf16', 'assets', fused_vae=False)
```

修改前默认代码快照：`records/next_optimizations_20261103/before`。实验逻辑独立保存在 scripts，正式运行不导入实验脚本。未覆盖任何已有模型或配置。
