# M5 Pro ImageWAM 测试命令与性能

项目根目录：/Users/qklj/work/cx/ImageWAM。以下对应2026-11-03实测，并非新运行的速度保证。

## 硬件与软件

|项目|实际环境|
|---|---|
|设备|Mac mini / Mac17,16|
|芯片 / CPU|Apple M5 Pro / 15 核|
|GPU|16 核 Apple GPU / Metal 4|
|统一内存|24 GB；sysctl=25769803776 bytes（24 GiB）|
|独立显存|无，GPU共享统一内存|
|当前MPS建议工作集|17.760 GiB|
|macOS|27.2 / 26B123|
|Python / PyTorch|3.12.13 / 2.15.0|
|torchvision / Transformers|0.29.2 / 4.56.1|
|NumPy / MPS可用|2.5.4 / True|
|执行配置|BF16；PYTORCH_ENABLE_MPS_FALLBACK=0|

Mac没有独立显存，CPU/GPU共享统一内存，不能理解为“24 GB内存 + 24 GB显存”。
recommended_max_memory是MPS当前建议工作集预算，不是独立显存或本次占用。
以下取第一个稳态迭代结束采样，**不是峰值**；allocator与driver两种口径不能相加。

|运行记录|MPS tensor allocated GiB|MPS driver allocated GiB|
|---|---:|---:|
|20261103_100525|14.336|15.813|
|20261103_102606_verified|14.336|15.813|
|20261103_dynamic_profile|14.336|15.820|


[本次提取的环境JSON](development_records/m5pro_robotwin/20261103_test_commands/environment.json)。
信息来自system_profiler、sysctl、sw_vers和.venv-mps；不保存设备序列号。
历史软件版本以各结果benchmark.json/requirements-lock.txt为准，本次采集不替代历史记录。

## 性能口径和总表

- MPS、BF16、CPU fallback关闭，模型常驻；每次在线编码文本和图像，无特征缓存。
- RoboTwin episode0/frame0，三图拼接H×W=288×256，14维状态，seed42，16×14动作，10步去噪。
- 性能样本实际39 token；原版padding到512，dynamic按实际长度执行、不截断。
- 端到端：已解码三图、状态、文本 → CPU反归一化动作，包含预处理、编码、prefill、去噪和回传；不含读视频、仿真、加载、写文件。
- 正式独立基准：首次前向单列，6次预热、20次无模块钩子稳态、5次独立模块profiling。
- 模块为同步inclusive wall时间，父子项不能相加，也不能当作无钩子稳态。
- 未控制OS页缓存、系统Metal编译缓存；加载和首次推理不是受控冷系统启动。

|测试|模型加载 s|首次推理 s|稳态平均 ms|稳态p95 ms|稳态样本数|
|---|---:|---:|---:|---:|---:|
|Original baseline|14.676|5.758|711.456|713.805|20|
|Verified baseline|11.140|4.829|739.498|770.935|20|
|Dynamic standalone|9.354|2.477|351.494|352.850|20|
|Interleaved baseline|共用一次加载|未单列|722.527|729.515|20|
|Interleaved dynamic|共用一次加载|未单列|392.177|409.672|20|


交错测量722.527→392.177 ms：耗时减少45.72%，加速1.84倍。不要选择不同轮次的最快值计算加速比。
以下使用当前正式入口复跑同样测试语义；历史原始源码另存于各source_snapshot.tar.gz。

## 公共设置

所有后续命令先执行：

    cd /Users/qklj/work/cx/ImageWAM
    export HF_HUB_OFFLINE=1
    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_ENABLE_MPS_FALLBACK=0
    RUN_TAG=$(date +%Y%m%d_%H%M%S)

使用已有.venv-mps。输出目录必须不存在，脚本拒绝覆盖；再次运行前更新RUN_TAG。

## A. 原版三图基准：711.456 ms

    .venv-mps/bin/python -u scripts/m5pro/benchmark_robotwin.py \
      --optimization baseline \
      --episode 0 --frame 0 --warmup 6 --iterations 20 --profile-iterations 5 \
      --output-dir "evaluate_results/m5pro_robotwin/$RUN_TAG-baseline"

对应首次实测：加载14.676 s、首轮5.758 s、稳态711.456 ms、p95 713.805 ms。
[原始结果](evaluate_results/m5pro_robotwin/20261103_100525/benchmark.json)。

## B. 原版独立复核和历史动作比较：739.498 ms

    .venv-mps/bin/python -u scripts/m5pro/benchmark_robotwin.py \
      --optimization baseline \
      --episode 0 --frame 0 --warmup 6 --iterations 20 --profile-iterations 5 \
      --audit --reference-dir evaluate_results/m5pro_robotwin/20261103_100525 \
      --output-dir "evaluate_results/m5pro_robotwin/$RUN_TAG-baseline-verified"

对应复核：加载11.140 s、首轮4.829 s、稳态739.498 ms、p95 770.935 ms。
设备审计及输入扰动在性能测量后执行，不计入稳态。
[原始结果](evaluate_results/m5pro_robotwin/20261103_102606_verified/benchmark.json)。

## C. 可变文本优化：351.494 ms

    .venv-mps/bin/python -u scripts/m5pro/benchmark_robotwin.py \
      --optimization dynamic \
      --episode 0 --frame 0 --warmup 6 --iterations 20 --profile-iterations 5 \
      --audit \
      --output-dir "evaluate_results/m5pro_robotwin/$RUN_TAG-dynamic"

对应独立实测：加载9.354 s、首轮2.477 s、稳态351.494 ms、p95 352.850 ms。
[原始结果](evaluate_results/m5pro_robotwin/20261103_dynamic_profile/benchmark.json)。

dynamic不固定实际token数、不截断、不缓存文本特征；图像仍使用原RoboTwin预处理。
不要传历史--reference-dir：该选项要求逐位一致，而动态形状BF16输出有数值差异。数值比较用D节。

## D. 全文本长度、多观测与交错测试：722.527→392.177 ms

### D1. 全量文本扫描（无模型推理耗时）

    .venv-mps/bin/python -u scripts/m5pro/scan_robotwin_text.py

扫描921,032条任务，实际提示模板后31～79 token，共48种长度；刷新开发记录中的text_lengths.json。
这不是模型推理性能测试，无端到端推理耗时；不要与计时同时运行。

### D2. 数值对比与交错计时

    .venv-mps/bin/python -u scripts/m5pro/test_dynamic_optimization.py --full

自动新建evaluate_results/m5pro_robotwin/<时间戳>_dynamic_full/。
比较48种长度各一个真实文本及额外3组真实观测，共51组；之后每种模式6次预热、交替顺序各测20次。

- baseline平均722.527 ms，p95 729.515 ms。
- dynamic平均392.177 ms，p95 409.672 ms。
- [原始结果](evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full/comparison.json)。

全量文本只做长度扫描，没有对921,032条全部执行模型。
51组最大反归一化动作差0.0125736217、归一化差0.015625，样本平均MAE约0.00045250。
原始输入相同不等于BF16输出逐位一致；尚未验证闭环任务成功率。

## E. 四模式消融（小规模试跑）

    .venv-mps/bin/python -u scripts/m5pro/test_dynamic_optimization.py

每模式顺序执行3次，无独立标准预热。下表仅第2、3次原始端到端值，**不是正式稳态平均**。

|模式|内容|第2次 ms|第3次 ms|
|---|---|---:|---:|
|baseline|原版|722.465|709.389|
|tail|仅裁剪视觉末层未使用输出|707.270|707.147|
|trim|tail + 保留Qwen512计算，仅裁剪下游padding|487.491|486.653|
|dynamic|tail + Qwen和下游均按实际长度执行|350.947|349.570|

[原始结果](evaluate_results/m5pro_robotwin/20261103_104718_dynamic_smoke/comparison.json)。
tail在此样本逐位一致；trim/dynamic不是。试跑不足以证明tail的稳定收益。
单独正式测试可用以下命令，但尚无对应20次独立稳态结果，不能套用上表：

    .venv-mps/bin/python -u scripts/m5pro/benchmark_robotwin.py \
      --optimization tail --warmup 6 --iterations 20 --profile-iterations 5 \
      --output-dir "evaluate_results/m5pro_robotwin/$RUN_TAG-tail"

    .venv-mps/bin/python -u scripts/m5pro/benchmark_robotwin.py \
      --optimization trim --warmup 6 --iterations 20 --profile-iterations 5 \
      --output-dir "evaluate_results/m5pro_robotwin/$RUN_TAG-trim"

## F. 模块计时（已包含于A/B/C命令）

--profile-iterations 5在稳态之后执行独立模块测量，无需另一个启动脚本。

|运行|带同步profiling端到端均值 ms|无钩子稳态 ms|
|---|---:|---:|
|原版首次|726.475|711.456|
|原版复核|790.115|739.498|
|dynamic|445.313|351.494|

dynamic有一次701.334 ms异常样本，其中文本模块356.430 ms；profiling端到端中位数382.313 ms。
全部样本保留，未删点；这些profiling数字不等同于正式稳态。

## G. 原入口输入/后处理核对（无完整模型推理耗时）

    .venv-mps/bin/python scripts/m5pro/validate_io.py \
      evaluate_results/m5pro_robotwin/20261103_dynamic_profile

复测时将参数换成新目录。检查三图预处理、状态归一化和动作反归一化与原入口一致性。
不执行完整模型，不能代替原版/优化版动作数值比较。

## 保存的记录

每轮结果包含输入、任务文本、噪声、完整归一化/反归一化动作NPY/CSV、原始计时、环境版本和源码快照。
模型和历史基线不覆盖。性能测量时避免并行下载、扫描数据或运行其他推理。

- [优化覆盖与精度报告](development_records/m5pro_robotwin/20261103_dynamic_optimization/README.md)
- [正式代码入口](scripts/m5pro/README.md)
