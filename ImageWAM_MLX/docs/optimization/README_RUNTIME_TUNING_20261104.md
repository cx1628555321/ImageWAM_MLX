# M5 Pro：视觉 INT8 与运行时调优（2026-11-04）

正式 speed 已启用本轮两项优化。ABBA 四个独立进程、每进程 4 次预热与 40 次稳态，同期旧版 260.167 ms，新版 230.060 ms，减少 30.106 ms（11.57%），吞吐倍率 1.131×。

设备为 Apple M5 Pro（15 CPU / 16 GPU）、24 GiB 统一内存，无独立显存；MLX 报告建议工作集约 17.76 GiB。macOS 27.2、Python 3.12.13、MLX 0.32.4，架构 applegpu_g17s。测试存在后台系统负载和时间漂移，不能用历史最低值与本轮平均值直接归因。

计时范围：已解码的三张 RGB、state 和文本到 CPU 上反归一化动作；在线编码，无跨请求特征缓存；不含视频解码、模型加载和结果写盘。固定噪声、10 步、动作 16×14。

| 指标 | 同期旧 speed | 新 speed |
|---|---:|---:|
| 稳态平均 ms（各 80 次） | 260.167 | 230.060 |
| 合并 P50 / P95 ms | 254.153 / 297.605 | 229.844 / 264.428 |
| 加载平均 s（各 2 次） | 2.745 | 3.095 |
| 首次推理平均 ms（含首次编译等） | 1139.942 | 956.837 |

模块单独同步诊断各 6 次，样本较少且同步会改变调度，不可将模块之和视为无插桩端到端耗时。

| 模块 | 旧 ms | 新 ms |
|---|---:|---:|
| tokenization | 0.332 | 0.175 |
| preprocess | 4.855 | 2.213 |
| text_encoder | 38.424 | 31.589 |
| image_encoder | 28.231 | 38.507 |
| visual_prefill | 121.072 | 96.265 |
| action_denoise | 68.919 | 67.736 |
| postprocess | 0.202 | 0.164 |

## 实现与精度

1. 38 个视觉 single_blocks.linear1/linear2 使用 W4 打包存储（group64），运行时局部转换为 INT8 权重块；激活在线 INT8，INT32 累加，输出 BF16。冻结权重的行尺度只计算一次。不在每次前向展开整个 BF16 权重；仍有局部解包、缩放与激活量化，不能称为没有转换开销。其他视觉层、文本、VAE 保持原有精度，动作层仍用原 W1 混合路径，整个模型不是全 INT8。
2. speed CLI 在导入 MLX 前默认设置 MLX_MAX_OPS_PER_BUFFER=256，保留用户显式环境值。四个交错进程块均更快，平均节省 14.229 ms；测试样本输出完全一致。单独 INT8 的独立 120 组配对测试从 260.374 降至 229.694 ms，节省 30.680 ms（bootstrap 区间 23.856–37.467 ms）。两个独立收益不可相加，正式组合收益以本报告 ABBA 为准。
3. 当前 MLX 未内置该 PR 接口。适配来自 [MLX PR #4722](https://github.com/ml-explore/mlx/pull/4722)，固定提交 9cea817470afa7d3a4f1500a57687de8acb28895 的 mode-2 Metal 内核，通过 mx.fast.metal_kernel JIT 使用；未替换 MLX wheel，也未声称已编译原 PR。来源、哈希、MIT 许可见 imagewam_mlx/metal/staged_int8_upstream.json 与 LICENSE.mlx。三个整数矩阵案例与独立 NumPy INT32 参考逐值一致。

51 个代表输入相对旧 speed 的新增动作最大绝对误差为 0.02261293；标准样本新增 RMSE 为 0.00307573。正式入口在全部 51 个案例中与实验候选逐值一致；accuracy 标准样本与历史精度参考逐值一致；关闭 staged 后与旧 speed 逐值一致。infer CLI 与 benchmark 输出也逐值一致。未固定或截断文本长度，测试包含真实文本长度范围及额外长文本，候选额外覆盖 horizon 1/16/64。
新版标准样本相对提供的 PyTorch 参考 max_abs=1.808145、RMSE=0.518440。speed 原先已有激进 W1 误差，本次小幅新增误差不代表整体精度无损；没有进行机器人任务成功率验证。

## 未采用的候选与后续方向

关闭 Winograd 的卷积路径在正反测试均变慢，保留默认。fast sync、BFS 宽度、buffer MB 候选未确认稳定收益，不固化。环境项参考 [MLX 官方文档](https://ml-explore.github.io/mlx/build/html/usage/environment_variables.html)。视觉 prefill 与动作去噪仍是主要成本；下一步应针对剩余视觉 BF16 投影和 INT8 激活量化/布局开销做分段诊断与融合，同时测量动作阶段的实际 GPU 时间。已有双 GPU stream 测试未获益，不将增加 stream 数量视为必然提速。

## 启动命令

在 /Users/qklj/work/cx/ImageWAM_MLX 下运行。新 speed 组合稳态平均约 230.060 ms：

```bash
.venv/bin/python scripts/benchmark.py --profile speed --reference examples/robotwin_ep0_frame0 --output records/retest_speed_new --warmup 4 --iterations 40 --profile-iterations 3
```

旧 speed 对照稳态平均约 260.167 ms；需清除相关环境变量，以免继承调优：

```bash
env -u MLX_MAX_OPS_PER_BUFFER -u MLX_MAX_MB_PER_BUFFER -u MLX_BFS_MAX_WIDTH -u MLX_METAL_FAST_SYNCH -u MLX_CONV_WINOGRAD .venv/bin/python scripts/benchmark.py --profile speed --no-staged-visual --no-runtime-tuning --reference examples/robotwin_ep0_frame0 --output records/retest_speed_old --warmup 4 --iterations 40 --profile-iterations 3
```

实际三图入口（上面的稳态统计不等同于单次调用延迟）：

```bash
.venv/bin/python scripts/infer.py --profile speed --inputs examples/robotwin_ep0_frame0/inputs.npz --prompt-file examples/robotwin_ep0_frame0/prompt.txt --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy --output records/infer_speed_new --warmup 1
```

默认不传 profile 仍是 accuracy；--profile accuracy 明确选择精度路径。--no-staged-visual 关闭本轮视觉 INT8；--no-runtime-tuning 仅禁止设置环境默认值，不清除外部环境。直接 Python API 会自动选择 staged speed，但 buffer 阈值需调用者在初始化 MLX 前设置。自动 staged 仅对已验证的 M5 架构、MLX 0.32.4、原 BF16 visual + W1 action 配置启用。输出目录需使用新名称。

## 记录位置

records/runtime_tuning_20261104 包含 before 原代码、screen/reverse 候选、confirm 四块复测、int8_probe、int8_paired、native_confirm、final_cli ABBA 完整输出/输入/计时/源码快照、final_validation 51 例及 accuracy/rollback 结果和 cli_infer。实验脚本在 experiments_runtime_tuning_20261104；模型权重与原配置未覆盖。源码来源记录位于 upstream_int8。归档省略大型 trace.npz，但远端原文件保留并记录 SHA256。
