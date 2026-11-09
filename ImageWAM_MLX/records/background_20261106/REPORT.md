# 后台优化验证进度与汇总

状态：finished_with_failures
更新时间：2026-11-06 09:30:46

| 任务 | 状态 | 端到端 ms |
|---|---|---:|
| fusion_confirm_0 | completed | 157.94537168112583 |
| fusion_confirm_1 | completed | 157.290623402223 |
| fusion_confirm_2 | completed | 157.7808249357622 |
| fusion_confirm_3 | completed | 157.7141999779269 |
| build_route4796 | failed | — |
| build_affine4802 | failed | — |

## 数值与模块统计

| 任务 | 前缀 ms | 动作 ms | 标准参考 RMSE | 51 案例最大差（相对 stock） |
|---|---:|---:|---:|---:|
| fusion_confirm_0 | 66.243 | 62.552 | 0.5178162607085236 | 0 |
| fusion_confirm_1 | 65.977 | 62.232 | 0.51782398636891 | 0.016959667205810547 |
| fusion_confirm_2 | 65.998 | 62.33 | 0.51782398636891 | 0.016959667205810547 |
| fusion_confirm_3 | 66.159 | 62.543 | 0.5178162607085236 | 0 |

默认推理代码未自动切换。完整结果、误差与环境见各任务 results.json / 日志。
profile/toolchain.json 说明逐内核时间线是否具备条件；命令缓冲时间不等于逐内核时间。