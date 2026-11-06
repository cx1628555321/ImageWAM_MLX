# 后台优化验证续跑结果：2026-11-06

结论：原环境11项（含基线）完整模型测试已完成，追加4进程ABBA也完成。没有确认足以替换现有推荐路径的稳定收益。两个上游版本的构建均已推进至实际编译，但因缺少Metal编译器受阻；这些版本的推理测试未执行。逐内核时间线也因缺少xctrace受阻。不是所有方向都已完成。

## 已完成的完整模型测试

保持3张图片在线编码、动态文本、H16、10步、文本缓存命中；每进程10预热/50正式/50模块统计，51案例与3组动态输入。以下为同一轮筛选，不拿最低单次均值取代ABBA。

| 方案 | E2E ms | 前缀 ms | 动作 ms | 51例相对speed最大差 | 标准参考RMSE |
|---|---:|---:|---:|---:|---:|
| stock_baseline | 157.747 | 66.233 | 62.546 | 0.000000 | 0.517816 |
| stock_fuse_bias | 157.533 | 66.136 | 62.467 | 0.000000 | 0.517816 |
| stock_fuse_norm | 157.141 | 65.988 | 62.137 | 0.016960 | 0.517824 |
| stock_fuse_both | 157.375 | 65.944 | 62.305 | 0.016960 | 0.517824 |
| stock_math_relaxed | 157.656 | 66.083 | 62.501 | 0.016960 | 0.517909 |
| stock_math_fast | 165.386 | 66.089 | 62.628 | 0.016960 | 0.517909 |
| stock_native_w1 | 171.936 | 66.136 | 76.885 | 0.016960 | 0.517774 |
| stock_native_w4 | 172.006 | 66.071 | 76.659 | 2.280183 | 0.007607 |
| stock_qq_mxfp4 | 207.625 | 66.347 | 112.146 | 2.348188 | 0.020087 |
| stock_qq_mxfp8 | 237.276 | 66.775 | 141.744 | 2.288183 | 0.006293 |
| stock_qq_nvfp4 | 192.228 | 66.221 | 96.714 | 2.272182 | 0.013774 |

## 追加独立复测

顺序：baseline → fuse_norm → fuse_norm → baseline。

| 进程 | mean ms | P95 ms |
|---|---:|---:|
| 1: baseline | 157.945 | 158.362 |
| 2: fuse_norm | 157.291 | 157.780 |
| 3: fuse_norm | 157.781 | 159.953 |
| 4: baseline | 157.714 | 158.262 |

基线平均 157.830 ms，融合平均 157.536 ms，节省 0.294 ms（0.186%）。反向配对中候选并未胜出，且有数值变化，因此不切换默认，也不宣传为新的最佳性能。

## 命令缓冲诊断

30个诊断请求：平均178个命令缓冲；执行区间并集154.332 ms，首个GPU开始到最后结束154.664 ms，区间间隙0.332 ms；诊断请求wall平均163.131 ms（该轮未插桩对照163.529 ms）。

这是命令缓冲时间，不能当作逐kernel active cycles，也不能据此判定算力/带宽占比。CPU profile中的137ms落在workspace_denoise内包含GPU提交/等待，不代表CPU纯计算。此前157ms与此诊断轮次不应直接相减归因于某项优化。

## 已排除的构建问题与最终阻塞

已尝试并完成：源码归档替代partial clone、转送官方依赖wheel、固定版本fmt/nanobind离线依赖、复用json和metal-cpp缓存、关闭不使用的GGUF、DEV_RELEASE归档构建模式、MLX_METAL_JIT=ON。没有覆盖原.venv或模型权重。

最终两个版本均实际进入C++/Metal构建，日志包含：

```text
xcrun: error: unable to find utility "metal", not a developer tool or in PATH
NAX kernels require Metal 4 ... Building without NAX kernels.
Building arg_reduce.air
make[2]: *** [mlx/backend/metal/kernels/arg_reduce.air] Error 72
```

因此JIT开关不等于完全不需要离线Metal工具链。即使强行关闭Metal/NAX或只构建CPU，也不满足本次M5 GPU性能验证目标，未这样替代测试。

未完成项：#4796/#4682相关环境的native W1/W4和FP4/FP8测试、#4802 affine动作与前缀W4/W8测试、这些候选的ABBA，以及逐内核时间线。需要先提供支持Metal4的有效metal工具链；逐内核分析还需要完整Xcode/xctrace。未安装或切换系统Xcode，也未接受软件许可。

## 记录与恢复

- 已完成筛选：records/background_20261105/
- ABBA：records/background_20261106/fusion_confirm_*
- 最终编译日志：records/background_20261106_final/build_*_attempt1.log
- 离线源码/构建依赖/配置：records/background_20261106_offline/builds/
- 完整原始动作、轨迹保留在m5pro对应目录；本地保存轻量结果。
- 当前无测试队列继续运行；缺失工具链前不进行无意义重试。
- 补齐工具链后，先核验xcrun -f metal和xcrun xctrace list templates，再使用新的记录目录运行未完成方向；不要覆盖原结果。
- 原详细方案：README_BACKGROUND_OPTIMIZATION_PROMPT_20261105.md。
