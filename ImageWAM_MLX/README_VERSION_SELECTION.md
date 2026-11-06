# ImageWAM 推理版本选型与启动总表

更新：2026-11-06。**默认选型建议：native-w4 兼顾速度与数值精度；action-bf16 用于更低平均数值偏差；accuracy 用于全模型浮点对照。native-w1 仅建议作为速度极限实验。** 本文不改变代码的默认参数。

适用设备：M5Pro 开发服务器，Apple M5 Pro（15核CPU、16核GPU）、24GiB统一内存；macOS27.2、Xcode27.2。项目：`/Users/qklj/work/cx/ImageWAM_MLX`。旧环境MLX0.32.4；新环境为已验证源码构建0.32.5.dev20261106。以下命令在该服务器执行，不要用其他Python环境替换后直接沿用表中结论。

## 先设置公共输入

以下适用于 bash/zsh。同一终端执行一次，再复制总表中所需版本的启动命令：

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
NEWPY="$PWD/records/background_20261106_xcode/builds/route4796/venv/bin/python"
SAMPLE="$PWD/examples/robotwin_ep0_frame0"
RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
WAM_ARGS=(
  --inputs "$SAMPLE/inputs.npz"
  --prompt-file "$SAMPLE/prompt.txt"
  --noise "$SAMPLE/initial_noise_fp32.npy"
  --steps 10
  --warmup 10
)
```

同一版本重复执行前重新设置RUN_TAG；输出目录必须不存在。固定noise用于数值复现，不是缓存模型输出。

## 一张表选择版本

精度列全部来自同一批RoboTwin真实样本：12个episode、36帧、2份噪声，每版本72次。RMSE与最大差均越低越好；“参考”是同批全模型accuracy。性能列是各版本此前正式测量，**不是本轮六版本同批性能复测**。

| 版本与建议用途 | 稳态端到端 ms | 相对accuracy RMSE | 最大绝对差 | 相对示范动作 RMSE | 主要优化与精度配置 | 启动命令（先执行上方公共设置） |
|---|---:|---:|---:|---:|---|---|
| **native-w4：推荐，速度/精度折中** | **147.885** | **0.007701** | **0.043815** | **0.031396** | 新MLX运行时；INT8多模态前缀；原生W4动作线性层、BF16激活；文本特征缓存 | `"$NEWPY" scripts/infer_version.py --variant native-w4 "${WAM_ARGS[@]}" --output "records/${RUN_TAG}_native_w4"` |
| **action-bf16：更低平均数值偏差** | **179.828** | **0.003633** | **0.045226** | **0.029726** | 新运行时＋INT8前缀；恢复原始BF16动作权重与浮点GEMM；文本缓存；**不是全模型BF16** | `"$NEWPY" scripts/infer_version.py --variant action-bf16 "${WAM_ARGS[@]}" --output "records/${RUN_TAG}_action_bf16"` |
| **accuracy：全模型浮点参照** | **281.650**（历史、无文本缓存） | 0（参照定义） | 0 | 0.030532 | BF16主路径，统计/位置等保留必要FP32；冗余裁剪、图编译、RoPE融合、VAE布局/归一化融合、CPU/GPU重叠 | `.venv/bin/python scripts/infer.py --profile accuracy "${WAM_ARGS[@]}" --output "records/${RUN_TAG}_accuracy"` |
| native-w1：仅速度极限实验 | 146.903；另一批144.773 | 0.571768 | 3.008241 | 0.574711 | 新运行时＋INT8前缀；原生W1动作线性层、BF16激活；文本缓存；**量化偏差明显** | `"$NEWPY" scripts/infer_version.py --variant native-w1 "${WAM_ARGS[@]}" --output "records/${RUN_TAG}_native_w1"` |
| runtime：新运行时对照 | 149.187 | 0.571789 | 3.008241 | 0.574733 | 新MLX源码构建、Metal JIT及NAX可用；保留既有W1动作计算路径、INT8前缀及文本缓存 | `"$NEWPY" scripts/infer_version.py --variant runtime "${WAM_ARGS[@]}" --output "records/${RUN_TAG}_runtime"` |
| stock：旧运行时对照 | 160.747 | 0.571783 | 3.008241 | 0.574726 | MLX0.32.4；既有W1动作Split-K/输出融合、INT8前缀；文本缓存 | `.venv/bin/python scripts/infer_version.py --variant stock "${WAM_ARGS[@]}" --output "records/${RUN_TAG}_stock"` |

新版各分支还共用：按实际文本长度计算、裁剪无用加载/计算、图编译、请求内KV工作区、fast BF16图像编码，以及已验证的前缀INT8权重预处理、激活量化融合、尾块布局、权重直接读取与K循环展开。不能把这些组合收益单独归因于某一种位宽。W1/W4指动作权重量化位宽，不表示整网激活为INT1/INT4；图像编码与attention等仍保留浮点。

## 如何理解性能和精度

- 所有行均为三图输入、H16、14维动作、10步。除accuracy外，性能是文本缓存命中路径，图片和状态每次重算。端到端不含加载、磁盘读取、保存或仿真。新运行时版本正式测量为每进程10次预热、50次计时并交叉复测；accuracy的281.650ms是历史8次预热、100次测量。
- 不同批次受系统负载/调度波动影响，不能用两行差值声称某项优化净收益。W1后续窗口约145–150ms；action-bf16的同期W1为150.171ms。CLI的run.json.request_ms是单次请求，不是表中多次均值。
- 多样本精度统计覆盖真实36–51 token指令，输入与噪声完全一致。此前31–79 token长度压力测试是另一套；不要把它当成本表的独立真实场景。
- accuracy误差为0仅因为它是参照，不代表没有模型误差。action-bf16平均误差比W4低，但最大单元素差略高，不能保证每个样本都优于W4。
- 示范动作误差按当前帧起连续16步计算，包含12个关节与2个夹爪；量纲混合，只用于相同定义下的对照。分维统计见完整报告。
- 数据元信息标记train，本次未核验训练留出划分。**这是离线数值/示范误差验证，不是独立测试集泛化结论，也没有完成RoboTwin闭环成功率验证。** 不把RMSE换算成“准确率百分比”。

## 换自己的输入与取回输出

`inputs.npz`需要三路uint8、HWC、RGB图像：`cam_high`、`cam_left_wrist`、`cam_right_wrist`，以及float32的14维`state`。当前样本图像为480×640。`prompt.txt`为完整模型提示词；也可用`--instruction '任务描述'`替换`--prompt-file`，入口会添加训练提示前缀，不要同时传两者。文本没有固定为本表某个长度，仍受模型上下文上限约束。

同一模型对象内，相同文本复用特征，文本改变重新计算；每次独立启动CLI不共享上一个进程的缓存。连续开发或服务应复用模型实例。

每个输出目录含`action.npy`、`normalized.npy`、`action.csv`、`initial_noise.npy`、`run.json`。版本包装入口还在run.json的`tested_version`中记录实际分支、MLX路径与适配信息。action-bf16入口已与正式实验的action及normalized输出逐位比对一致。

## 原始证据与维护

- [RoboTwin多样本精度报告](README_ROBOTWIN_ACCURACY_20261106.md)：本表精度来源；远端原始输入、432份输出和逐案例/逐维结果在`records/robotwin_accuracy_20261106/`。
- [版本与启动索引](README_INFERENCE_VERSIONS.md)、[Xcode后性能验证](docs/optimization/README_XCODE_MLX_VALIDATION_20261106.md)：新运行时/W1/W4性能依据。
- [Action浮点对照](docs/optimization/README_ACTION_FLOAT_COMPARE_20261106.md)：BF16性能、FP16及展开W1的排除依据。
- [全模型accuracy历史报告](docs/optimization/README_DAILY_FINAL_20261103.md)：281.650ms来源。

FP16 action、W1展开BF16及无收益分块/融合候选仅保留实验记录，不列为推荐使用版本。4步去噪改变计算量与输出，不混入本表10步选择。更新模型、运行时或计算路径后，应重新跑精度与性能对照再更新本表。
