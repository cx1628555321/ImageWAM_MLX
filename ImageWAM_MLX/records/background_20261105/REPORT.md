# 后台优化验证进度与汇总

状态：finished_with_failures
更新时间：2026-11-05 18:24:27

| 任务 | 状态 | 端到端 ms |
|---|---|---:|
| profile | completed | — |
| stock_baseline | completed | 157.74654336099047 |
| stock_fuse_bias | completed | 157.53275748051237 |
| stock_fuse_norm | completed | 157.1407415997237 |
| stock_fuse_both | completed | 157.37488328071777 |
| stock_math_relaxed | completed | 157.65562243992463 |
| stock_math_fast | completed | 165.38554080121685 |
| stock_native_w1 | completed | 171.93596084136516 |
| stock_native_w4 | completed | 172.00622836127877 |
| stock_qq_mxfp4 | completed | 207.6246949803317 |
| stock_qq_mxfp8 | completed | 237.27640408207662 |
| stock_qq_nvfp4 | completed | 192.2279524768237 |
| build_route4796 | failed | — |
| build_affine4802 | failed | — |

## 数值与模块统计

| 任务 | 前缀 ms | 动作 ms | 标准参考 RMSE | 51 案例最大差（相对 stock） |
|---|---:|---:|---:|---:|
| stock_baseline | 66.233 | 62.546 | 0.5178162607085236 | 0 |
| stock_fuse_bias | 66.136 | 62.467 | 0.5178162607085236 | 0.0 |
| stock_fuse_norm | 65.988 | 62.137 | 0.51782398636891 | 0.016959667205810547 |
| stock_fuse_both | 65.944 | 62.305 | 0.51782398636891 | 0.016959667205810547 |
| stock_math_relaxed | 66.083 | 62.501 | 0.5179086172649155 | 0.016959667205810547 |
| stock_math_fast | 66.089 | 62.628 | 0.5179086172649155 | 0.016959667205810547 |
| stock_native_w1 | 66.136 | 76.885 | 0.5177743987540178 | 0.016959667205810547 |
| stock_native_w4 | 66.071 | 76.659 | 0.0076068771648549775 | 2.280182719230652 |
| stock_qq_mxfp4 | 66.347 | 112.146 | 0.02008746828956041 | 2.3481881618499756 |
| stock_qq_mxfp8 | 66.775 | 141.744 | 0.006292678802468457 | 2.288183331489563 |
| stock_qq_nvfp4 | 66.221 | 96.714 | 0.013773616371947451 | 2.2721821069717407 |

默认推理代码未自动切换。完整结果、误差与环境见各任务 results.json / 日志。
profile/toolchain.json 说明逐内核时间线是否具备条件；命令缓冲时间不等于逐内核时间。