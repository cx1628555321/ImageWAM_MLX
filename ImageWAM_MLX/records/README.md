# MLX 测试记录索引

正式结果与可复现命令见 [项目 README](../README.md)。

|目录|状态/用途|
|---|---|
|`bf16_final`|最终正式 MLX BF16 基准，6 预热/20 稳态/5 profile；具有源码快照、原始输入、每轮动作和中间特征|
|`mps_dynamic_retest`|同机 MPS dynamic 重新测试，6/20/5，原模型仅只读使用|
|`validation_v1`|48 种已观察文本长度 + 3 组额外真实观察，另含长短切换和输入扰动|
|`mps_reference_trace`|独立 MPS 中间结果，用于逐模块数值对比|
|`cli_verified`|独立 infer.py 命令测试，动作与 bf16_final 逐值一致|
|`bf16_v1`|修正图像标量精度后的前一轮有效基准，结果保留；最终报告优先 bf16_final|
|`smoke_v1`|失败：最初环境缺少 Jinja2，加载后尚未执行模型推理；已补齐依赖|
|`smoke_v2`|调试试跑：图像预处理发现与 MPS 最大差 0.0078125，后续已修正；不可用作输入等价的精度或正式性能结论|

`environment.json` 为不包含设备序列号的硬件/软件快照。`assets_manifest.json` 保存 tokenizer、配置、统计及样例文件的 SHA256。每个正式记录目录包含独立 JSON 及输出，未覆盖旧记录。

初次 smoke_v1 的 benchmark.json 在早期脚本异常退出时停留在 running；以本索引中的失败说明为准。当前 benchmark.py 已具备异常记录逻辑。各进程的首次耗时会受未受控系统缓存/内存状态影响，不删除异常慢的首次或稳态样本。

本次详细测试新增 `detailed_20261103`、`accuracy_detailed_20261103`、`async_validation_20261103`；完整结论和命令见 [详细报告](../README_PERFORMANCE_20261103.md)。默认推理代码未修改。

第二轮有效实验：`compile_20261103_v2`；独立运行：`compiled_benchmark_20261103`；完整验证：`compile_validation_20261103`、`compile_variability_20261103`；独立 CLI：`compiled_cli_20261103`。`compile_20261103` 为初始化阶段源码缩进提取失败，非有效性能记录。详见 [编译优化报告](../README_COMPILE_OPTIMIZATION_20261103.md)。

[编码器与 QKV 优化实验](../README_ENCODER_QKV_20261103.md)：encoders_20261103、qkv_20261103、qkv_validation_20261103、qkv_benchmark_20261103 均已完成。

[图调度与 KV 实验](../README_GRAPH_KV_20261103.md)：graph_20261103、kv_micro_20261103、lazy_prefix_validation_20261103、lazy_prefix_benchmark_20261103 已完成。

[Metal 捕获与时间调制复用](../README_METAL_CONDITIONING_20261103.md)：metal_action_20261103、conditioning_20261103、conditioning_validation_20261103。

[RoPE Metal 实验](../README_ROPE_METAL_20261103.md)：rope_metal_20261103、rope_validation_20261103、rope_benchmark_20261103、rope_kernel_validation_20261103。

[视觉 RoPE 实验](../README_VISUAL_ROPE_20261103.md)：visual_rope_20261103、visual_rope_validation_20261103、visual_rope_benchmark_20261103、visual_rope_kernel_20261103。

[正式运行时整合及选择性 K/V 实验](../README_CURRENT_RUNTIME_20261103.md)：integrated_*、selective_kv_*，整合前代码快照 pre_runtime_integration_20261103。

[选择性 K/V 复测](../README_SELECTIVE_KV_RETEST_20261103.md)：selective_kv_repeat_20261103、selective_prefill_20261103。

[RMSNorm＋RoPE 实验](../README_RMS_ROPE_20261103.md)：rms_rope_kernel_20261103、rms_rope_20261103、rms_rope_aligned_20261103、rms_rope_validation_20261103。

[真实 Q/K 诊断与组合修复](../README_RMS_ROPE_DIAGNOSTIC_20261103.md)：rms_rope_diagnostic_20261103、rms_rope_native_reduce_20261103、rms_rope_precise_*、rms_rope_aligned_precise_*。
