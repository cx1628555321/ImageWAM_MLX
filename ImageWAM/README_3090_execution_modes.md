# 我们的 ImageWAM 3090 执行方式比较

本文记录 eager、`torch.inference_mode()`、`torch.compile`、TorchScript 和 CUDA Graph 的适配与实测。环境、模型下载地址和权重路径沿用 [README_3090_inference.md](README_3090_inference.md)。原仓库 README、模型实现及已有 CUDA/PyTorch 环境保持不变。

<a id="comparison-summary"></a>

## 五种方式统一对比与结论

本节是当前统一结果入口。测试日期 2026-10-27，RTX 3090 24 GiB，PyTorch 2.8.0+cu128，BF16。五种方式使用相同权重、输入图像 hash、task 0、seed 42、文本上下文 512、双路 256×256 原始图像和 10 步去噪，输出一个 16×7 动作块。每项一次加载、6 次预热、20 次稳态端到端采样和 10 次模块采样。

Eager 与 inference_mode 来自同日相邻的新一轮测试；compile / CUDA Graph 来自已验证的 v2，TorchScript 来自 v5。本次整理沿用这些实测数据，未重新运行模型。所有相对指标统一以本轮 Eager **400.409 ms** 为基线，不能与历史表中以 394.963 ms 为基线的百分比混用。

### 端到端耗时与显存

| 方式 | 平均 ms | 中位数 ms | p95 ms | 加速比 | 延迟下降 | allocated / reserved 峰值 GiB |
|---|---:|---:|---:|---:|---:|---:|
| Eager / no_grad | 400.409 | 397.124 | 421.109 | 1.000× | 0.00% | 17.360 / 17.438 |
| Eager + inference_mode | 383.178 | 382.927 | 385.927 | 1.045× | 4.30% | 17.360 / 17.438 |
| torch.compile | 260.685 | 260.385 | 262.579 | 1.536× | 34.90% | 17.281 / 17.346 |
| TorchScript | 282.798 | 282.390 | 284.730 | 1.416× | 29.37% | 18.399 / 18.557 |
| CUDA Graph | 316.240 | 315.816 | 318.640 | 1.266× | 21.02% | 18.027 / 18.518 |

端到端范围为原始观测与文本到 CPU 动作，包含在线文本编码、预处理、视觉 prefill、动作去噪和后处理；排除模型加载、准备、预热、仿真与文件写入。显存为当前进程 PyTorch allocator 在稳态区间的峰值，allocated 与 reserved 不能相加，也不能视为整卡占用或整个启动过程的峰值。

Eager 最后 3 个样本约为 413/422/421 ms，全部保留，因此均值与 p95 受到短时波动影响。inference_mode 的平均延迟下降 4.30%，按中位数比较下降 3.57%，相对上一轮 Eager 均值下降 2.98%；这体现本次测量的收益范围，不能保证其他任务得到相同提升。

### 模块耗时统一对照

单位 ms；10 次完整推理的 CUDA event 均值，动作单步为 100 次调用平均。CUDA event 包含 stream 等待和 CPU 提交间隙，不代表纯 kernel 活跃时间；父子项重叠，不能将全部行相加作为端到端耗时。

| 模块 | Eager | inference_mode | compile | TorchScript | CUDA Graph |
|---|---:|---:|---:|---:|---:|
| Qwen 文本前向 | 89.371 | 88.597 | 70.445 | 75.797 | 81.877 |
| AE 图像编码 | 14.095 | 14.098 | 10.618 | 12.963 | 13.286 |
| 视觉 Transformer / KV prefill | 143.700 | 143.425 | 120.352 | 124.859 | 135.557 |
| 动作 Transformer，10 步合计 | 137.246 | 122.938 | 50.460 | 58.081 | 75.233 |
| 动作 Transformer，单步 | 13.725 | 12.294 | 5.046 | 5.808 | 7.523 |
| 完整动作去噪循环（含上述 Transformer） | 143.218 | 127.989 | 51.817 | 59.427 | 76.566 |

全部 20 个模块的 CUDA 与 host wall 时间、逐次 span 保存在汇总 CSV 及单模式报告中。上述模块来自独立的带钩子测量，主 E2E 来自不带模块钩子的 20 次测量。

### 加载、准备和预热统一对照

单位秒。准备列包含编译/追踪/捕获及 A/B/C_text/A_again 四组输入验证，不含生成 eager 参考、加载与后续预热。Eager 和 inference_mode 的准备主要是验证；不能将此列全部称为编译时间。

| 方式 | 模型加载总计 | 模式准备与验证 | 6 次预热合计 |
|---|---:|---:|---:|
| Eager / no_grad | 63.024 | 1.594 | 2.392 |
| Eager + inference_mode | 61.461 | 1.536 | 2.290 |
| torch.compile | 68.500 | 86.032 | 1.568 |
| TorchScript | 67.760 | 24.340 | 1.719 |
| CUDA Graph | 87.878 | 2.890 | 1.903 |

各模型加载细项如下。Qwen 合并 pretrained、设备转换和 tokenizer；AE 合并初始化、读权重、赋值与设备转换。未列出的调度和组装开销仍包含在加载总计中。加载时长受磁盘、页缓存和内存压力影响，不能归因于模式本身加快模型加载。

| 加载细项（秒） | Eager | inference_mode | compile | TorchScript | CUDA Graph |
|---|---:|---:|---:|---:|---:|
| 视觉模型初始化与权重 | 15.056 | 13.476 | 16.944 | 20.752 | 21.261 |
| 动作模型初始化与设备转换 | 3.229 | 3.235 | 3.359 | 3.169 | 3.199 |
| Qwen 加载、设备转换和 tokenizer | 4.034 | 25.050 | 8.273 | 11.289 | 22.836 |
| AE 加载与设备转换 | 0.190 | 0.562 | 0.586 | 0.656 | 0.653 |
| ImageWAM 策略 checkpoint 读取与应用 | 39.941 | 18.584 | 38.785 | 31.321 | 39.280 |

compile 本轮四边界 FX graph cache miss，但未强制清空全部磁盘缓存，86.032 秒不代表所有缓存清空后的冷启动。TorchScript 的 24.340 秒包含首次追踪及后续延迟图优化，不能只累加 trace 调用计时。CUDA Graph 准备包括每边界的预热与捕获。

### 正确性与执行方式

| 方式 | 四组输入相对 eager 的最大绝对差 | gripper | 同模式重复输出最大差 | 实际生效检查 |
|---|---:|---|---:|---|
| Eager / no_grad | 0 | 一致 | 0 | 原生 eval/no_grad |
| Eager + inference_mode | 0 | 一致 | 0 | 四模块入口 inference_mode=True、grad_enabled=False |
| torch.compile | 0.00732421875 | 一致 | 0 | 四个完整边界图；稳态/模块测量无重编译 |
| TorchScript | 0.00732421875 | 一致 | 0 | 四个真实 ScriptModule；默认 JIT 优化开启，参数存储共享 |
| CUDA Graph | 0 | 一致 | 0 | 四边界各 capture 一次；稳态/模块测量只 replay |

compile/TorchScript 的 BF16 差异符合预先设定的逐元素容差 `abs(error) <= 0.005 + 0.02 * abs(reference)`；Eager、inference_mode、CUDA Graph 在本次四组输入上与参考相同。检查覆盖两种真实观测、变化的文本和返回原始输入；这些数值回归检查不等同于完整 LIBERO 成功率测试。

inference_mode 每次完整预测进入/退出 context，进出成本计入 E2E；加载、仿真、eager 参考生成在 context 外。首次验证临时检查 Qwen/AE/visual 各 1 次、action 10 次入口，随后移除审计；阶段计数为 prepare=4、warmup=6、steady=20、profile_prepare=2、profile=10。该模式仍使用 eager 算子，本次未与 compile、TorchScript 或 CUDA Graph 叠加。

### 比较结论

- **持续重复推理时，当前测得 compile 最快**：260.685 ms，比统一 Eager 基线减少 34.90%，稳态 allocated 也最低（17.281 GiB）；需承担本次约 86 秒的准备与验证成本。
- **TorchScript 次之**：282.798 ms，延迟下降 29.37%，准备与验证约 24 秒；allocated 为 18.399 GiB，高于 Eager。
- **CUDA Graph 在较短准备时间下获得中等收益**：316.240 ms，下降 21.02%，准备与验证约 2.89 秒；本次与参考输出相同，allocated 为 18.027 GiB。
- **inference_mode 无编译/捕获成本，收益较小**：383.178 ms，平均下降 4.30%，显存与 Eager 相同；主要改善动作阶段，Qwen、AE 和视觉 prefill 变化很小。
- **主要优化收益来自动作 Transformer**：10 步合计从 Eager 137.246 ms 降至 compile 50.460 ms；compile 下视觉 prefill 仍是最大的单项（120.352 ms）。

### 结果位置和复现入口

服务器仓库：`/home/ubuntu/data_1/chenxing/ImageWAM/ImageWAM`。默认运行全部五种方式：

```bash
cd /home/ubuntu/data_1/chenxing/ImageWAM/ImageWAM
bash scripts/flux2/compare_3090_execution_modes.sh
```

统一报告：[comparison.md](evaluate_results/3090_modes/modes_20261027_inference_verified/comparison.md)、[comparison.json](evaluate_results/3090_modes/modes_20261027_inference_verified/comparison.json)、[module_comparison.csv](evaluate_results/3090_modes/modes_20261027_inference_verified/module_comparison.csv)。各模式 `summary.md`、`benchmark.json`、`load_spans.json`、`iterations.csv` 与 `spans.csv` 保留详细记录。

该汇总目录内 eager / inference_mode 为最近一轮实跑，compile/cudagraph 链接至 v2，TorchScript 链接至 v5；[selection_manifest.json](evaluate_results/3090_modes/modes_20261027_inference_verified/selection_manifest.json) 记录真实来源和报告 hash。原四模式目录及失败实验均保留。

验证记录：`artifacts/3090_setup/inference_mode_verification.json`；环境快照：`artifacts/3090_setup/venv_after_inference_mode.json`。原 CUDA/PyTorch、项目 manifest、上游 README 保持不变。模型下载地址、下载命令与加载路径继续在 [README_3090_inference.md](README_3090_inference.md) 维护。

## 启动方式

```bash
ssh -p 9022 -o ClearAllForwardings=yes ubuntu@47.109.191.96
cd /home/ubuntu/data_1/chenxing/ImageWAM/ImageWAM
bash scripts/flux2/compare_3090_execution_modes.sh
```

默认顺序为 `eager inference_mode compile torchscript cudagraph`。每种模式使用一个独立进程，加载一次模型后多次推理。独立进程避免前一种模式的编译缓存对象、CUDA Graph 内存池或方法替换影响下一种模式。磁盘编译缓存保存在仓库 `.cache/torchinductor` 和 `.cache/triton`，不强制清除；准备时间必须结合缓存状态解释。

只比较指定模式，或增加采样次数：

```bash
EXECUTION_MODES="eager compile cudagraph" \
BENCH_ITERS=50 PROFILE_ITERS=20 WARMUP_MAX=30 \
COMPARISON_ID="modes_$(date +%Y%m%d_%H%M%S)_$$" \
  bash scripts/flux2/compare_3090_execution_modes.sh
```

只运行一种模式，也可以使用原性能启动入口：

```bash
RUN_ID="compile_$(date +%Y%m%d_%H%M%S)_$$" \
TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/torchinductor" \
TRITON_CACHE_DIR="$PWD/.cache/triton" TORCHINDUCTOR_COMPILE_THREADS=1 \
  bash scripts/flux2/profile_3090_inference.sh \
    --execution-mode compile --validate-execution
```

`--execution-mode` 可选 `eager`、`inference_mode`、`compile`、`torchscript`、`cudagraph`。编译/追踪/捕获模式的默认边界为 `qwen,ae,visual,action`；单模式可用 `--execution-boundaries ae,visual,action`，比较入口可用 `EXECUTION_BOUNDARIES`。省略的模块会保留 eager，因此报告必须同时说明生效范围，不能把局部加速称为整个模型全部编译。`inference_mode` 包裹整个预测调用，不使用此模块筛选参数。

单独运行 inference_mode，或与本次 Eager 基线相邻比较：

```bash
bash scripts/flux2/profile_3090_inference.sh \
  --execution-mode inference_mode --validate-execution

EXECUTION_MODES="eager inference_mode" \
  bash scripts/flux2/compare_3090_execution_modes.sh
```

默认预热 6～20 次、稳态端到端 20 次、模块计时 10 次。比较脚本每种模式默认最长运行 1800 秒（`MODE_TIMEOUT_SECONDS`），失败或超时仍保留该模式日志，继续其他模式；它不会把失败模式静默改成 eager。输出目录已存在时拒绝覆盖。

## 实际加速的范围

| 模式 | 实现 | 保留 eager 的部分 |
|---|---|---|
| eager | 原生 `eval()` + `no_grad()`，BF16 | 全部执行流程 |
| inference_mode | 每次完整 `predict` 外层启用 `torch.inference_mode()`，保留 `eval()` 与原有 `no_grad()` | 全部算子和 Python 调度仍为 eager；不启用编译、trace 或捕获 |
| compile | 四个 GPU 模块分别 `torch.compile(backend="inductor", fullgraph=True, dynamic=False)` | 文本模板/tokenizer、CPU 图像处理、调度控制、模块间连接、动作后处理 |
| torchscript | 四个 GPU 模块的 tensor-only adapter 使用 `torch.jit.trace`，真实参数共享原存储 | 同上；固定形状与 Python 静态参数 |
| cudagraph | 四个 GPU 模块分别捕获并 replay CUDA Graph | 同上；图外输入复制、掩码准备和输出复制也计入实际耗时 |

编译/追踪/捕获的四个边界对应：

| 名称 | 原生方法 |
|---|---|
| `qwen` | `model.text_encoder.model.forward`，保留原生 LM head 和 hidden-state 输出 |
| `ae` | `model.vae.encode` |
| `visual` | `model.mot.prefill_flux2_video_cache` |
| `action` | `model.mot.forward_flux2_action_with_video_cache`，每次动作块调用 10 次 |

`inference_mode` 是梯度与张量跟踪模式，与 eager/编译/捕获属于不同维度：它额外关闭 view tracking 和 version counter 更新，不会自动生成执行图，也不替代 `model.eval()`。本次只测原生 eager + inference_mode，没有把它叠加到 compile/TorchScript/CUDA Graph。参见 [PyTorch 2.8 inference_mode 文档](https://docs.pytorch.org/docs/2.8/generated/torch.autograd.grad_mode.inference_mode.html)。

实现位于 `benchmark_3090_inference.py` 的 `predict`：每次调用进入/退出 context，覆盖预处理、文本/视觉/动作模型及 CPU 动作后处理；context 进入退出也包含在 E2E 计时内。加载、仿真和生成 eager 参考保持在该 context 之外。prepare A 暂时审计 Qwen、AE、visual、action 的真实入口，要求 `torch.is_inference_mode_enabled()==True` 且 `torch.is_grad_enabled()==False`，随后恢复方法。该审计不进入 warmup/steady/profile；结果和各阶段启用次数记录在 `execution.inference_mode`。

这里的 `fullgraph=True` 要求每个指定模块完整编译，不代表把 tokenizer、仿真和整个 Python 推理循环编成一个图。编译模式显式关闭 Inductor 的 CUDA Graph，并设置 `emulate_precision_casts=True` 保留原生 BF16 中间舍入，单独测量编译效果；CUDA Graph 模式捕获原生 eager GPU 调用，单独测量 replay 效果。没有同时开启两者，也没有修改精度、量化、文本缓存、权重或去噪步数。

TorchScript 的 Qwen 使用图外原生 4D mask helper，避免原生 mask 的 vmap 在 trace 中出现 `unordered_map::at`。张量树的展平和重建必须在调用结束后释放临时引用；局部递归闭包产生的循环引用会延迟 GPU 张量释放，相关修复与失败记录见下文。

TorchScript 使用 trace，未调用 `torch.jit.freeze()` 或 `optimize_for_inference()`；参数保持共享原存储，避免复制大模型权重。输入结构、shape、stride、dtype、device 和 Python 静态值必须一致。trace 对数据相关 Python 分支有局限，当前测试不能证明任意形状或任意任务均兼容。[PyTorch trace 文档](https://docs.pytorch.org/docs/2.8/generated/torch.jit.trace.html)

CUDA Graph 每次调用都把当前输入复制进静态缓冲区，包括当前视觉 K/V；所有输出在捕获外复制到普通分配器持有的张量，保证后续 replay 不覆盖尚需使用的视觉缓存。四图共享内存池，输出复制和签名检查的开销均保留在模块与端到端测量内。Qwen 原生 2D mask 准备含 CPU 标量判断；适配在每次调用图外使用 Transformers 自带 `create_causal_mask` 生成等价 4D causal/padding mask，再复制入图。这部分准备仍计时，并针对变化的文本验证，未手写另一套注意力算法。图模式需要固定结构和生命周期，不能把它解释为复用上次推理结果。[PyTorch CUDA Graph 文档](https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-graphs)

## 正确性与计时口径

每个进程先生成原生 eager 参考，然后安装对应适配。验证使用 task 0 的初始状态 0 和 1，以及初始状态 0 配合变化的任务文本，最后回到原始输入再次执行。图捕获、trace 和编译均发生在准备阶段。

针对 BF16 编译/融合的数值差异，预先设定与 eager 比较的 `atol=0.005, rtol=0.02`，记录最大绝对差、平均绝对差、RMSE，并要求 gripper 输出一致。重复执行同一模式仍要求 `atol=rtol=1e-5`。正确性检查不计入推理延迟；这些检查是数值回归检查，不能替代完整 LIBERO 成功率评测。

准备完成后再进行自适应预热、20 次无模块钩子的端到端测量和 10 次模块测量。模块钩子安装后允许额外两次未计时准备，避免将钩子带来的首次状态变化混入样本。compile 禁止静默 fallback、禁止重新编译，并检查计时区间的 Dynamo 图计数；CUDA Graph 不允许在 steady/profile 首次捕获或重捕获。

端到端范围仍为原始双相机观测和任务文本到 CPU 上 `16×7` 动作块，含在线文本编码、图像/本体处理、视觉 prefill、10 步动作去噪和后处理，排除模型加载、准备、预热、仿真 step/render 和文件写入。各模式使用相同 BF16、seed 42、上下文 512、输入分辨率和 checkpoint。

模块 CUDA event 是 stream 经过时间，包含必要输入/输出复制和 CPU 提交间隙，不是纯 kernel 活跃时间。模块表为 inclusive 范围，父子项不能相加；重复 10 次的动作模块同时提供整次推理合计和单步平均。主要端到端指标取没有计时钩子的 `steady`。

## 输出与代码

比较输出：`evaluate_results/3090_modes/<COMPARISON_ID>/`。

| 文件 | 内容 |
|---|---|
| `comparison.md` | 各模式 E2E、准备、预热、显存及模块对照 |
| `comparison.json` / `comparison.csv` | 机器可读汇总；未完成或不稳定结果不计算有效加速比 |
| `module_comparison.csv` | 各模式模块 CUDA / host wall 对照 |
| `exit_codes.tsv` | 批量脚本逐模式退出码 |
| `<mode>/benchmark.json` | 实际模式、路径、配置、编译/捕获详情、正确性、原始加载/模块 span 与 E2E 统计；逐次 E2E 原始样本在 `iterations.csv` |
| `<mode>/execution_progress.json` | 准备阶段各输入的比较记录 |
| `<mode>/eager_*action.npy` | 同一进程生成的原生参考动作 |
| `<mode>/summary.md`、`iterations.csv`、`spans.csv`、`module_summary.csv` | 单模式性能统计 |
| `<mode>/run.log` | 原生加载、编译/trace/capture、失败堆栈与逐次耗时 |

主要代码：

- [compare_3090_execution_modes.sh](scripts/flux2/compare_3090_execution_modes.sh)：顺序运行各模式，限制编译线程数，隔离输出。
- [benchmark_3090_inference.py](scripts/flux2/benchmark_3090_inference.py)：加载、参考输出、执行适配、正确性检查、计时和报告。
- [profile_3090_compile.py](scripts/flux2/profile_3090_compile.py)：Inductor 与 TorchScript adapter，运行结束后恢复原方法。
- [profile_3090_cudagraph.py](scripts/flux2/profile_3090_cudagraph.py)：静态缓冲区、CUDA Graph 捕获/replay、共享 pool 与输出生命周期。
- [profile_3090_modules.py](scripts/flux2/profile_3090_modules.py)：统一的原生模块计时边界。
- [summarize_3090_modes.py](scripts/flux2/summarize_3090_modes.py)：从实际报告汇总，核对目录模式与报告模式。


## 历史四模式实验与适配过程

<details>
<summary>展开原始四模式结果、失败记录和修复过程（旧 Eager 基线 394.963 ms）</summary>


2026-10-27，在同一张 RTX 3090 24 GiB 上顺序执行。环境为 Python 3.12.13、PyTorch 2.8.0+cu128、CUDA runtime 12.8、Triton 3.4.0；本次未安装依赖，也未改变原 CUDA/PyTorch。每种模式一次加载、6 次预热、20 次稳态 E2E、10 次模块采样。输入为 task 0、seed 42、双路 256×256 原始图像，文本上下文 512，10 步去噪，输出一个 16×7 动作块。

| 模式 | E2E 平均 ms | p95 ms | 相对 eager 加速 | 延迟下降 | 稳态 allocated 峰值 GiB | 稳态 reserved 峰值 GiB |
|---|---:|---:|---:|---:|---:|---:|
| eager | 394.963 | 396.806 | 1.000× | 0.00% | 17.360 | 17.438 |
| compile | 260.685 | 262.579 | 1.515× | 34.00% | 17.281 | 17.346 |
| torchscript | 282.798 | 284.730 | 1.397× | 28.40% | 18.399 | 18.557 |
| cudagraph | 316.240 | 318.640 | 1.249× | 19.93% | 18.027 | 18.518 |

这些显存数字是当前进程 PyTorch allocator 在稳态测量期间的峰值，包含当时仍存活的模型、图和缓冲区；不是整卡 `nvidia-smi` 占用，也不是整个加载/编译过程的峰值。allocated 与 reserved 不能相加。

当前配置下 compile 的端到端耗时最低。主要收益来自动作 Transformer：10 步合计从约 137.97 ms 降到 50.46 ms。加速后视觉 prefill 仍是最大的单项。CUDA Graph 仍需执行原生计算，并额外保留静态缓冲区与输出复制；本次约减少 19.93% 端到端延迟，allocated 比 eager 增加约 0.67 GiB。

### 主要模块耗时

单位 ms；表内为 10 次完整推理的 CUDA event 均值，动作单步为 100 次调用平均。父子项有重叠，不能将全部行相加作为 E2E。

| 模块 | eager | compile | torchscript | cudagraph |
|---|---:|---:|---:|---:|
| Qwen 文本编码前向 | 89.191 | 70.445 | 75.797 | 81.877 |
| AE 图像编码 | 14.073 | 10.618 | 12.963 | 13.286 |
| 视觉 Transformer / KV prefill | 143.693 | 120.352 | 124.859 | 135.557 |
| 动作 Transformer，10 步合计 | 137.970 | 50.460 | 58.081 | 75.233 |
| 动作 Transformer，单步 | 13.797 | 5.046 | 5.808 | 7.523 |
| 完整动作去噪循环，含上述 Transformer | 143.969 | 51.817 | 59.427 | 76.566 |

Eager 模块测量的第 10 次出现约 423 ms 的 E2E 样本；本表保留全部 10 次数据，未删除该样本。独立的无钩子 E2E 20 次数据仍稳定，故模块均值的和与无钩子 E2E 不会严格一致。全部 20 个模块的 CUDA 与 host wall 时间、每次原始 span 见汇总 CSV 和各模式报告。

### 一次性加载、准备与预热

单位秒；各项分别报告，均不混入上面的稳态延迟。准备列含编译/trace/capture 及 A/B/C_text/A_again 四次正确性验证，不含之前生成 eager 参考、模型加载或后续预热。TorchScript 的延迟图优化发生在前几次执行，已计入准备阶段；不能仅用 trace 函数耗时代表全部准备成本。

| 模式 | 模型加载总计 | 模式准备与验证 | 后续 6 次预热合计 |
|---|---:|---:|---:|
| eager | 65.870 | 1.590 | 2.380 |
| compile | 68.500 | 86.032 | 1.568 |
| torchscript | 67.760 | 24.340 | 1.719 |
| cudagraph | 87.878 | 2.890 | 1.903 |

加载细项如下；Qwen 项合并 pretrained、设备转换和 tokenizer，AE 项合并初始化、读权重、赋值和设备转换。其他未列出的 factory 调度/组装开销仍包含在模型加载总计中。加载数据受文件系统页缓存、内存压力和磁盘读取影响，不能将这些差异解释为执行模式本身提高加载速度。

| 加载细项（秒） | eager | compile | torchscript | cudagraph |
|---|---:|---:|---:|---:|
| 视觉模型初始化与权重 | 8.224 | 16.944 | 20.752 | 21.261 |
| 动作模型初始化与设备转换 | 3.193 | 3.359 | 3.169 | 3.199 |
| Qwen 加载与设备转换、tokenizer | 14.641 | 8.273 | 11.289 | 22.836 |
| AE 加载与设备转换 | 0.070 | 0.586 | 0.656 | 0.653 |
| ImageWAM 策略 checkpoint 读取与应用 | 39.169 | 38.785 | 31.321 | 39.280 |

compile 四个边界首次准备约为 Qwen 27.565 s、AE 13.370 s、visual 23.998 s、action 20.212 s；本次各边界均记录 FX graph cache miss，未把磁盘编译缓存强制清空，所以此结果不代表全缓存清空时的冷启动。TorchScript 各边界首次追踪准备约为 1.331 / 0.192 / 1.142 / 0.955 s，包含参考前向、trace、首次执行及样例校验；仍保留 PyTorch 默认图优化（`optimized_execution=True`）。

### 正确性与实际生效检查

- 四种模式均通过 A/B/C_text/A_again 检查，改变观测、改变文本后均与对应 eager 参考比较，gripper 输出一致。
- CUDA Graph 四组输入相对 eager 的最大绝对差均为 0。compile / TorchScript 的最大绝对差均为 0.00732421875，满足预先设定的逐元素 `abs(error) <= 0.005 + 0.02 * abs(reference)`；这不是逐位一致，也不是放宽容差后的结果。
- 各模式稳态和模块重复测量相对本模式准备结束后的参考动作，最大绝对差均为 0。
- compile 共生成 4 个完整边界图，steady/profile 编译计数增量均为空；未发生测量期间重编译或静默 eager fallback。
- TorchScript 四边界均由真实 `torch.jit.ScriptModule` 执行，注册参数共享原始存储。已完成 6 次预热、20 次稳态和 10 次模块测量，未再次出现此前临时张量引用导致的 OOM；当前测试不等同于长时间服务压力测试。
- CUDA Graph 四边界各 capture 一次，steady/profile 捕获次数均为 0；steady 阶段 Qwen/AE/visual 各 replay 20 次、action replay 200 次。

### 适配中的失败与修复

| 原始记录目录（`evaluate_results/3090_modes/` 下） | 现象 | 处理与结果 |
|---|---|---|
| `modes_20261027_v1/compile` | 默认 Inductor 在观测 B 上未通过预设误差阈值 | 设置 `emulate_precision_casts=True` 保留 BF16 中间舍入，v2 通过；没有提高容差 |
| `modes_20261027_v2/torchscript` | Qwen mask 的 functorch/vmap 在 trace 中报 `unordered_map::at` | 每次在 trace 外用原生 helper 生成 4D mask，计入耗时 |
| `modes_20261027_v3/torchscript` | 可完成追踪，但第 3 次预热 OOM | 排查运行过程中临时张量引用 |
| `modes_20261027_v4/torchscript` | 关闭 JIT 图优化后仍在第 3 次预热 OOM | 排除“仅由默认图优化导致”的解释，未采用关闭优化作为最终方案 |
| `modes_20261027_v5/torchscript` | 修复后完整测试通过 | `_flatten` / `_unflatten` 改为模块级递归函数，消除局部递归闭包的循环引用；恢复默认图优化 |

循环引用定位使用真实 CPU Torch Tensor 作隔离验证：旧实现即使删除调用者引用，在禁用 cyclic GC 时仍保留 Tensor，手动 GC 后才释放；新实现连续 100 次处理共 200 个 Tensor，禁用 GC 且不调用 `gc.collect()` 时全部立即释放。GPU 完整复测也通过，因此最终实现没有通过每次强制 GC 掩盖引用问题。CUDA Graph 共享池、图外 mask、输出复制等适配均位于我们新增的 helper，未改上游模型实现。

### 结果位置与复查

统一入口：[comparison.md](evaluate_results/3090_modes/modes_20261027_verified/comparison.md)，机器可读文件：[comparison.json](evaluate_results/3090_modes/modes_20261027_verified/comparison.json)、[module_comparison.csv](evaluate_results/3090_modes/modes_20261027_verified/module_comparison.csv)。

`modes_20261027_verified` 是最终有效实验的汇总目录：eager 是该目录内的实跑，compile/cudagraph 链接至 v2，torchscript 链接至 v5；不是声称重新运行了一次完整批次。来源、报告 hash 记录在 [selection_manifest.json](evaluate_results/3090_modes/modes_20261027_verified/selection_manifest.json)。所有失败日志单独保留，没有进入有效加速统计。

原始模型下载地址、下载命令和实际加载路径继续在 [README_3090_inference.md](README_3090_inference.md) 维护，各运行 `benchmark.json.model_paths` 也记录实际路径。最终环境快照为 `artifacts/3090_setup/venv_after_execution_modes.json`；对比之前快照，Python、torch/CUDA、受保护 GPU 包以及项目 manifest 均保持一致。结果复算、模式生效、同输入同配置、README 链接和源文件 hash 的检查记录在 `artifacts/3090_setup/execution_modes_verification.json`。

</details>


## 2026-10-28：独立动作加载器

新增 [优化加载适配与实测记录](README_3090_optimized_loading.md)：保留共享 embedding 与 FLUX 别名，避免视觉重复加载，严格比较中间结果及最终动作；包括正常入口、性能对比和完整动作文件。原执行模式数据继续保留。
