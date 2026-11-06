# W1 Linear 门控残差融合测试（2026-11-04）

## 结论

已完成单内核 epilogue 与 Split-K 归约融合两个候选。生产 accuracy/speed 默认路径均未修改；Split-K 的第二轮端到端复测取得正向收益，但存在数值差异与较大运行波动，因此先提供独立可选入口。

## 实现与范围

保持动作 W1/group64 权重、BF16 激活，视觉/Qwen/三图预处理不变。不调用完整权重 dequantize；沿用块内解码与矩阵乘。融合 bias、gate 乘法及 residual 加法，保留 BF16 中间舍入。仅动作序列长度 >13 且 K>4096 的打包 Linear 使用候选，其他情况回退官方实现。文字长度没有固定或截断。

单核版：NAX 矩阵乘输出直接计算门控残差。Split-K 版：NAX 输出 FP32 部分和，单独内核归约并融合 bias/gate/residual，保留一个 FP32 中间缓冲，并非零拷贝。修改累加顺序后不保证逐值一致。

官方调度与 Split-K 参考：[MLX 0.32.4 quantized.cpp](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/quantized.cpp)。代码结构解释不是 GPU 时间线证明。生成器、上游哈希和 MIT 许可证随源码归档。

## 环境与计时

M5 Pro，15 CPU 核 / 16 GPU 核，24 GiB 统一内存，macOS 27.2，Python 3.12.13，MLX 0.32.4。真实 RoboTwin 三图、39-token 文本、状态、16×14 动作、10 步。每候选与当前 speed（KV 工作区开启）共用权重；双方首次运行及各 3 次预热后 60 对 AB/BA 交替，端到端含 CPU 动作输出，不含模型加载、文件读取及保存。

本轮基线 268–275 ms，明显高于前轮约 241 ms，且单次波动较大；不可跨轮直接比较。主机存在 mediaanalysisd、VideoToolbox、虚拟机等进程，未终止它们；没有 GPU 时间线，不能归因到某个进程。首次运行先执行基线、后执行候选，共享部分编译缓存，不能用于比较冷启动收益。

|候选|同轮基线 ms|候选 ms|节省 ms|节省 bootstrap 95% 区间 ms|最终动作 max abs|
|---|---:|---:|---:|---|---:|
|paired|274.822|281.321|-6.499|[-15.453, 2.287]|0.01600122|
|split_fixed_paired|268.537|262.683|5.854|[-1.453, 13.301]|0.01600122|

以上两个端到端区间均跨零，不认为已证明提速。区间仅反映本轮配对样本，不覆盖不同系统状态。

## 模块同步计时

每配置 3 次，逐模块同步会改变调度，不能直接相加作为端到端。

|轮次/模块|基线 ms|候选 ms|
|---|---:|---:|
|paired/tokenization|0.353|0.218|
|paired/text_encoder|42.476|45.678|
|paired/preprocess|2.980|2.832|
|paired/image_encoder|32.791|32.598|
|paired/visual_prefill|124.214|141.316|
|paired/action_denoise|76.592|73.325|
|paired/postprocess|0.209|0.166|
|split_fixed_paired/tokenization|0.268|0.265|
|split_fixed_paired/text_encoder|37.532|40.805|
|split_fixed_paired/preprocess|2.759|4.216|
|split_fixed_paired/image_encoder|33.555|28.295|
|split_fixed_paired/visual_prefill|118.248|117.618|
|split_fixed_paired/action_denoise|84.408|69.708|
|split_fixed_paired/postprocess|0.177|0.169|

## 隔离动作模块复核

120 对交替、双方各 5 次预热；同一前缀 KV，仅用于隔离计时，不能当成完整请求耗时。每次仍重新执行 10 步 denoise。基线 82.419 ms，Split-K 候选 68.580 ms，节省 13.838 ms，95% 区间 [11.264, 16.507] ms。归一化动作 max abs=0.01562500、RMSE=0.00360057。

## 动态输入与精度

短动作长度 1、8 回退路径输出逐值一致。下表为修正版与当前 speed 的差异；不是相对 PyTorch 的精度，也未验证闭环任务成功率。

|文本 token|动作长度|步数|max abs|RMSE|
|---:|---:|---:|---:|---:|
|39|1|3|0.00000000|0.00000000|
|39|8|3|0.00000000|0.00000000|
|39|32|3|0.01230478|0.00294469|
|39|64|3|0.01600134|0.00303567|
|193|16|3|0.01600134|0.00358702|

## 排除的失败实现

初版 Split-K 使用了不受 helper 支持的 TM=TN=1 fragment 组合，矩阵乘漏算，出现 max abs>2 的动作差异。该轮数据在 split_paired/INVALID.md 明确标为无效，不计收益。已改为 BN64，并新增静态断言、随机矩阵相对 RMSE 检查；修正后的分块部分和与完整矩阵乘吻合至浮点舍入差异。保留失败日志是为了可追溯。

## 启动命令

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
.venv/bin/python experiments_epilogue_20261104/build.py
.venv/bin/python experiments_epilogue_20261104/build_split.py
.venv/bin/python experiments_epilogue_20261104/split_probe.py
.venv/bin/python experiments_epilogue_20261104/paired.py
.venv/bin/python experiments_epilogue_20261104/split_paired.py
.venv/bin/python experiments_epilogue_20261104/action_paired.py
```

paired.py / split_paired.py 对应上表两种端到端耗时；action_paired.py 仅为动作模块计时。脚本拒绝覆盖已存在的测试目录；重复执行应先在脚本副本中将 root 指向新的 records 子目录。所有输入输出 npz、原始采样、首次耗时和加载耗时已归档。

## 后续方向

已完成 Split-K 第二轮复测及真实样本覆盖，后续应在负载更稳定时复测并评估任务成功率，再决定默认配置。若继续做内核优化，应先扫描 BN/分块数、测量实际调度与中间缓冲成本，而不是假定算子数越少越快。本轮取得可复核的局部及第二轮端到端收益，尚未验证闭环任务精度，不直接替换默认路径。

## 第二轮端到端复测

120 对交替：基线 272.581 ms，Split-K 候选 266.295 ms，节省 6.285 ms，95% 区间 [1.200, 11.465] ms；最终动作 max abs=0.01600122。复现脚本为 experiments_epilogue_20261104/confirm.py。

## 真实样本覆盖

51 组真实样本/长度案例，最终动作均有限。最大绝对差 0.02261293，逐值一致 0 组。全部输入对应的动作与归一化动作已保存。此检查不能证明任务成功率不变。

## 实验推理入口

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
RUN_TAG=$(date +%Y%m%d_%H%M%S)
.venv/bin/python experiments_epilogue_20261104/infer.py --epilogue split \
  --inputs examples/robotwin_ep0_frame0/inputs.npz \
  --prompt-file examples/robotwin_ep0_frame0/prompt.txt \
  --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy \
  --warmup 3 --output records/$RUN_TAG-epilogue-infer
```

通过 --epilogue baseline 可在同一入口回到当前 speed，--epilogue nax 对应非分块候选。生产 scripts/infer.py 默认仍保持 accuracy，原 speed 未变。
