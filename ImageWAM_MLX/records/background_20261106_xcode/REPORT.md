# 后台优化验证进度与汇总

状态：completed
更新时间：2026-11-06 11:33:40

| 任务 | 状态 | 端到端 ms |
|---|---|---:|
| stock_current_baseline | completed | 166.65128995198756 |
| build_route4796 | completed | — |
| route4796_baseline | interrupted_contention | 146.92702590138651 |
| route4796_baseline | completed | 146.9692650833167 |
| route4796_native_w1 | completed | 144.8436092841439 |
| route4796_native_w4 | completed | 145.4300324805081 |
| route4796_qq_mxfp4 | completed | 156.58493742113933 |
| route4796_qq_mxfp8 | completed | 172.91817920398898 |
| route4796_qq_nvfp4 | completed | 156.7960908752866 |
| build_affine4802 | completed | — |
| affine4802_baseline | interrupted_contention | 147.81207171967253 |
| affine4802_baseline | completed | 150.29052095720544 |
| affine4802_affine_w4 | completed | 150.6519008462783 |
| affine4802_affine_w8 | completed | 164.84197580371983 |
| affine4802_prefix_affine_w4 | completed | 162.98279002308846 |
| affine4802_prefix_affine_w8 | completed | 166.6826158843469 |
| confirm_route4796_native_w1_0 | completed | 152.1150884393137 |
| confirm_route4796_native_w1_1 | completed | 145.54442582302727 |
| confirm_route4796_native_w1_2 | completed | 148.26192163978703 |
| confirm_route4796_native_w1_3 | completed | 152.27449662983418 |
| confirm_route4796_native_w4_0 | completed | 149.20626742299646 |
| confirm_route4796_native_w4_1 | completed | 145.15653417794965 |
| confirm_route4796_native_w4_2 | completed | 150.6134716013912 |
| confirm_route4796_native_w4_3 | completed | 149.7001750243362 |

## 数值与模块统计

| 任务 | 前缀 ms | 动作 ms | 标准参考 RMSE | 51 案例最大差（相对 stock） |
|---|---:|---:|---:|---:|
| stock_current_baseline | 67.653 | 63.765 | 0.5178162607085236 | 0 |
| route4796_baseline | 0 | 0 | 0.5181807834132783 | 0 |
| route4796_baseline | 67.404 | 49.668 | 0.5181807834132783 | 0 |
| route4796_native_w1 | 67.351 | 47.537 | 0.5180615791487811 | 0.016001343727111816 |
| route4796_native_w4 | 67.563 | 47.744 | 0.0075469038811879045 | 2.280182719230652 |
| route4796_qq_mxfp4 | 67.494 | 53.03 | 0.02008746828956041 | 2.3481881618499756 |
| route4796_qq_mxfp8 | 68.031 | 66.706 | 0.006268111864140207 | 2.288183331489563 |
| route4796_qq_nvfp4 | 67.892 | 52.849 | 0.013773616371947451 | 2.2721821069717407 |
| affine4802_baseline | 0 | 0 | 0.5181807834132783 | 0 |
| affine4802_baseline | 67.531 | 49.138 | 0.5181807834132783 | 0 |
| affine4802_affine_w4 | 67.629 | 49.078 | 0.007273979394393444 | 2.2761824131011963 |
| affine4802_affine_w8 | 67.915 | 62.637 | 0.002427172992134226 | 2.2841830253601074 |
| affine4802_prefix_affine_w4 | 84.338 | 49.202 | 0.517594942817413 | 0.022612929344177246 |
| affine4802_prefix_affine_w8 | 89.625 | 49.305 | 0.5181045267221412 | 0.022612929344177246 |
| confirm_route4796_native_w1_0 | 67.431 | 49.453 | 0.5181807834132783 | 0 |
| confirm_route4796_native_w1_1 | 67.413 | 47.78 | 0.5180615791487811 | 0.016001343727111816 |
| confirm_route4796_native_w1_2 | 67.54 | 47.337 | 0.5180615791487811 | 0.016001343727111816 |
| confirm_route4796_native_w1_3 | 67.533 | 49.367 | 0.5181807834132783 | 0 |
| confirm_route4796_native_w4_0 | 67.521 | 49.184 | 0.5181807834132783 | 0 |
| confirm_route4796_native_w4_1 | 67.539 | 47.663 | 0.0075469038811879045 | 2.280182719230652 |
| confirm_route4796_native_w4_2 | 67.456 | 47.84 | 0.0075469038811879045 | 2.280182719230652 |
| confirm_route4796_native_w4_3 | 67.519 | 49.455 | 0.5181807834132783 | 0 |

默认推理代码未自动切换。完整结果、误差与环境见各任务 results.json / 日志。
profile/toolchain.json 说明逐内核时间线是否具备条件；命令缓冲时间不等于逐内核时间。