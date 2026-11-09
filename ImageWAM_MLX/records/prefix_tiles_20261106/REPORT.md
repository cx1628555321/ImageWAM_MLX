# 后台优化验证进度与汇总

状态：completed
更新时间：2026-11-06 14:33:55

| 任务 | 状态 | 端到端 ms |
|---|---|---:|
| diagnostic | completed | — |
| baseline | completed | 149.6816400019452 |
| selected_tiles | completed | 145.14799585798755 |
| confirm_0_baseline | completed | 144.52945257653482 |
| confirm_1_selected_tiles | completed | 145.08693504147232 |
| confirm_2_selected_tiles | completed | 147.98377250204794 |
| confirm_3_baseline | completed | 145.01604501972906 |

## 数值与模块统计

| 任务 | 前缀 ms | 动作 ms | 标准参考 RMSE | 51 案例最大差（候选相对本轮 native-w1；baseline 行为占位0） |
|---|---:|---:|---:|---:|
| baseline | 67.453 | 47.175 | 0.5180615791487811 | 0 |
| selected_tiles | 67.339 | 47.436 | 0.5180615791487811 | 0.0 |
| confirm_0_baseline | 67.435 | 47.133 | 0.5180615791487811 | 0 |
| confirm_1_selected_tiles | 67.425 | 47.301 | 0.5180615791487811 | 0.0 |
| confirm_2_selected_tiles | 67.424 | 47.371 | 0.5180615791487811 | 0.0 |
| confirm_3_baseline | 67.538 | 47.533 | 0.5180615791487811 | 0 |

默认推理代码未自动切换。完整结果、误差与环境见各任务 results.json / 日志。
结论与 ABBA 对照见 ../../README_PREFIX_TILES_20261106.md。