# 后台优化验证进度与汇总

状态：finished_with_failures
更新时间：2026-11-06 12:03:12

| 任务 | 状态 | 端到端 ms |
|---|---|---:|
| cli_stock | completed | — |
| cli_runtime | completed | — |
| cli_native-w1 | completed | — |
| cli_native-w4 | completed | — |
| baseline | completed | 148.2597916573286 |
| fuse_bias | completed | 153.53323166142218 |
| fuse_norm | completed | 149.8668717232067 |
| fuse_both | completed | 145.00233494327404 |
| slim_quant | interrupted_contention | None |
| slim_quant | failed | None |
| confirm_0_baseline | completed | 145.7952157803811 |
| confirm_1_fuse_both | completed | 149.0054390579462 |
| confirm_2_fuse_both | completed | 148.18080993602052 |
| confirm_3_baseline | completed | 148.0797374167014 |

## 数值与模块统计

| 任务 | 前缀 ms | 动作 ms | 标准参考 RMSE | 51 案例最大差（相对 stock） |
|---|---:|---:|---:|---:|
| baseline | 67.462 | 47.584 | 0.5180615791487811 | 0 |
| fuse_bias | 67.451 | 47.382 | 0.5180615791487811 | 0.0 |
| fuse_norm | 67.394 | 47.503 | 0.5179937293254462 | 0.016001343727111816 |
| fuse_both | 67.402 | 47.488 | 0.5179937293254462 | 0.016001343727111816 |
| slim_quant | 0 | 0 | — | 0 |
| slim_quant | 0 | 0 | — | 0 |
| confirm_0_baseline | 67.518 | 47.596 | 0.5180615791487811 | 0 |
| confirm_1_fuse_both | 67.42 | 47.108 | 0.5179937293254462 | 0.016001343727111816 |
| confirm_2_fuse_both | 67.399 | 47.662 | 0.5179937293254462 | 0.016001343727111816 |
| confirm_3_baseline | 67.576 | 47.706 | 0.5180615791487811 | 0 |

默认推理代码未自动切换。完整结果、误差与环境见各任务 results.json / 日志。
profile/toolchain.json 说明逐内核时间线是否具备条件；命令缓冲时间不等于逐内核时间。