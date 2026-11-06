# 3090：ImageWAM 动作推理加载优化

2026-10-28 实测。服务器 `ubuntu@47.109.191.96`，仓库 `/home/ubuntu/data_1/chenxing/ImageWAM/ImageWAM`。独立适配，不修改上游 README。模型来源、下载命令和最初 uv 环境配置见 [3090 环境记录](README_3090_inference.md)，执行模式对比见 [执行模式记录](README_3090_execution_modes.md)。

## 启动

```bash
ssh ubuntu@47.109.191.96
cd /home/ubuntu/data_1/chenxing/ImageWAM/ImageWAM
# 正常 LIBERO 任务入口：默认 task 0、一次回合
bash scripts/flux2/infer_3090_action_only.sh
# 独立进程分别加载 native / action_only，一次加载、多次推理并严格比较
bash scripts/flux2/compare_3090_loaders.sh
# 仅重新汇总已有测试，不启动模型
.venv/bin/python scripts/flux2/summarize_3090_loaders.py \
  evaluate_results/3090_loaders/loaders_20261028_v1
# 回到原加载方式
bash scripts/flux2/infer_3090_libero.sh
```

比较脚本使用唯一结果目录，拒绝覆盖；可用 `COMPARISON_ID` 指定目录名。正常任务脚本支持 `RUN_ID`、`TASK_ID`、`NUM_TRIALS`、`CKPT_PATH` 等原入口环境变量。维持仓库 `.venv`，本次没有安装依赖或修改 CUDA / PyTorch。

## 独立代码和加载路径

- `src/imagewam/optimized_action_loader.py`：新工厂 `create_imagewam_flux2_action_only`，使用 meta 构建结构，严格检查键和形状后一次填充参数。
- `scripts/flux2/infer_3090_action_only.sh`：通过 Hydra `_target_` 接入原 `experiments/libero/eval_libero_single.py`；仍然执行原动作推理流程。
- `scripts/flux2/benchmark_3090_inference.py`：新增 `--loader action_only --reference-dir ... --audit-loading`；原加载器仍为默认。
- `scripts/flux2/validate_3090_loading.py`：中间结果哈希、参数所属模块调用覆盖检查，仅精度验证期间安装钩子。
- `scripts/flux2/compare_3090_loaders.sh`、`summarize_3090_loaders.py`：自动测试、模块表格、完整动作 CSV。

| 组件 | 实际加载路径（仓库相对路径） | 处理 |
|---|---|---|
| 视觉 Transformer、ActionDiT、proprio | `../flux2_klein_4b/model.pt`，step 34720 | 一次 mmap 读取 checkpoint；直接填充最终参数 |
| AE | `checkpoints/3090/ae.safetensors` | 只读取 encoder 和必要 BN 缓冲区 |
| Qwen3 | `checkpoints/3090/Qwen3-4B` | 本地 config/tokenizer，选择 safetensors 分片中的 embedding 和前 27 层 |
| 动作反归一化 | `../flux2_klein_4b/dataset_stats.json` | 与 native 一致 |
| 原视觉初始化文件 | `checkpoints/3090/flux2_imagewam_visual.safetensors` | 新加载器读取次数为 0；原脚本的文件存在性检查仍保留 |

`model.pt` mmap 会映射原文件，不能把映射耗时当作完整磁盘读取时间；实际页面访问包含在后续组件加载时间中。原权重文件不删除、不改写。普通入口随后调用 `load_checkpoint` 时核对路径、大小和修改时间，确认同一 checkpoint 已恢复，避免再次加载；不支持改换 checkpoint 或恢复 optimizer。

## 删减边界与共享权重

| 组件 | 保留 | 不加载/不执行 |
|---|---|---|
| FLUX 视觉 | 输入投影、调制、全部 5 double / 20 single blocks、视觉/文本 KV | 动作路径不使用的视觉 `final_layer` |
| ActionDiT / proprio | 全部策略权重 | 无 |
| AE | 完整 encoder（含均值/方差输出卷积）、BN 必要缓冲区 | decoder |
| Qwen3 | 完整 `[151936,2560]` embedding、层 0–26、FP32 RoPE | 层 27–35、最终 norm、LM head 运算 |

**共享权重按底层使用关系保留，不按未调用名称删除。** 原 Qwen 的 LM head 与输入 embedding 共享参数，去掉 head 不等于删除这块权重：完整 embedding 保留，也不将其重复计入节省量。FLUX checkpoint 的 140 组视觉别名逐一核对存储地址及形状，按 canonical 路径加载一次；运行时检查 visual/action 与 MoT 中为同一对象，double/single blocks 别名和 `dit=mot` 均保留。

Qwen 输出仍取 hidden_states `[9,18,27]`。保留 27 层时末尾 norm 必须替换 Identity，保证第 27 个状态仍是原网络中间层原始输出；RoPE 保持 FP32，不随参数转换为 BF16。Qwen 后九层原先确实执行，但结果不用于当前动作输出，属于可去除的无效下游计算。

此工厂限定现有 `klein-base-4b`、BF16、在线文本编码、完整非 LoRA checkpoint 的动作推理，不作为训练、视频生成或联合视频动作生成入口。保留参数所属模块均有调用，并不表示每个 embedding 行、每个标量或 encoder 的每个输出通道都影响每个样本。

## 可复现条件和精度

RTX 3090，torch 2.8.0+cu128 / CUDA 12.8，transformers 4.56.1；Eager、原 no_grad 路径、BF16，不开启 compile / TorchScript / CUDA Graph / inference_mode。seed 42，文本长度 512，动作 horizon 16，去噪 10 步，重规划 12 步。

测试是 LIBERO `libero_spatial` task 0 的真实双相机观测，每相机 256×256×3，初始化状态 0，等待 30 步；固定观测重复在线编码，不缓存文本/图像特征。数据位于 `third_party/LIBERO/libero/libero/{bddl_files,init_files,assets}`；具体输入哈希、文本、配置及路径保存于各 `benchmark.json` / `resolved_config.yaml`。

独立 native 进程产生参考，优化进程比较 A（初始状态 0）、B（状态 1）、C_text（修改文本）、A_again。文本特征、proprio 拼接、AE latent、视觉 KV、十次动作 Transformer 输出的张量哈希全部一致；四组最终 16×7 动作逐位相等，最大绝对误差 0。优化器保留的 617 个参数所属模块均有调用，140 组视觉存储别名及所有共享对象检查通过。此结论覆盖这些实测样本，不宣称穷尽所有输入。

正常入口完整回合：`evaluate_results/3090/model/action_only_20261028_v1`，task 0 为 **1/1 successes**，106 步结束，保存视频、运行日志和 Hydra 配置。

## 性能结果

每种加载一次，6 次预热，20 次无模块钩子端到端测量，再做 10 次模块测量。精度审计在预热之前单独执行，因此预热时间不代表第一次前向初始化。端到端为原始观测/文本到 CPU 可执行动作，不含仿真、加载、预热和文件输出。

| 指标 | native | action_only |
|---|---:|---:|
| 模型加载 wall / s | 95.336 | 33.009 |
| 六次预热合计 / s | 2.387 | 2.239 |
| 稳态平均 / ms | 398.429 | 372.070 |
| 稳态 p95 / ms | 408.340 | 373.841 |
| 稳态 peak allocated / GiB | 17.360 | 15.526 |
| 稳态 peak reserved / GiB | 17.438 | 15.611 |
| 驻留参数 / billion | 8.624 | 7.647 |

本轮端到端下降 6.62%，峰值 allocated 减少 1.834 GiB。加载时间受进程顺序和系统页缓存影响，95.336→33.009 s 是本轮观测，不是受控冷启动速度保证。显存为 PyTorch 当前进程统计，不含其他进程和驱动上下文。

| 模块 CUDA event / ms（整次推理累计） | native | action_only |
|---|---:|---:|
| Qwen forward | 87.521 | 63.004 |
| AE encode | 14.038 | 14.064 |
| 视觉 prepare | 1.948 | 1.933 |
| 视觉 prefill | 143.323 | 143.635 |
| proprio linear | 0.060 | 0.045 |
| 动作 Transformer，10 次合计 | 139.828 | 137.579 |
| 动作去噪完整循环 | 145.602 | 143.306 |

CUDA event 可含 CPU 提交间隙，父子项不能相加；视觉/动作小幅差异不作为确定加速结论。主要计算节省来自 Qwen。各模型详细加载阶段、全部模块、最终动作见 [完整比较报告](evaluate_results/3090_loaders/loaders_20261028_v1/comparison.md)。原始记录在同目录 native / action_only 的 `benchmark.json`、`spans.csv`、`iterations.csv`、`validation_intermediates.json`、`parameter_owner_coverage.json`、`actions_*.npy`；并列动作见 `actions_comparison.csv`。

首个动作（A，native 与优化相同）：

```text
[0.1831054538, 0.1519775242, 0.1593017429, -0.0132756820,
 0.0183105469, -0.0054722172, -1.0]
```

环境保护核对通过：`artifacts/3090_setup/venv_after_optimized_loading.json` 对比 `venv_after_inference_mode.json`；原 `README.md` SHA256 保持 `301ab40295495f385fdb4cd1a73ca24544fe04893ddb63f1c3e46ef8f213221d`。
