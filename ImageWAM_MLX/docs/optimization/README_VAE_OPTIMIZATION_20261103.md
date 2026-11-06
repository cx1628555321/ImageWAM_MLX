# MLX 图像编码专项优化（2026-11-03）

本轮完成 5 个候选的局部实验、两轮各 100 对端到端测试及完整精度回归。实现了保留归一化计算边界的残差末端编译，已接入正式运行时的可选开关。**默认保持关闭：局部收益可测，端到端独立复测未确认稳定收益。** 原有默认 VAE GN+SiLU 融合继续启用。

## 测试环境与边界

M5 Pro 15 CPU/16 GPU、24 GiB 统一内存，MLX 0.32.4。同一个真实 RoboTwin 三图观测，拼图 288×256，BF16；端到端仍为在线文本/图像编码、16×14 动作、10 步去噪。固定相同状态、文本及初始噪声，没有减少图像尺寸、改权重精度或缓存图像特征。

局部测试在已经预处理的图像张量到最终 latent 之间同步计时；端到端覆盖 CPU 输入处理到反归一化动作返回。候选与基线共享只读权重，AB/BA 交替测量，保留全部样本；区间为配对差值 bootstrap 95% 区间（10000 次、seed 42，未校正时间序列相关性）。

## 局部结果

| 候选 | 原版图像编码 ms | 候选 ms | latent 最大差异 | 决定 |
|---|---:|---:|---:|---|
| 仅编译卷积及 bias | 41.441 | 41.445 | 0 | 无收益 |
| 整个残差块编译 | 41.422 | 38.519 | 0.1328125 | 精度不通过 |
| 编译 GN+SiLU 路径 | 41.502 | 38.965 | 0.1328125 | 精度不通过 |
| 1×1 卷积改矩阵乘法 | 41.454 | 40.858 | 0.140625 | 精度不通过 |
| **保留 GN，只编译残差末端** | **41.413** | **41.075** | **0** | 保留可选 |

前四项各 40 对，最后一项 80 对。末端方案局部节省 0.337 ms，区间 [0.314, 0.364] ms，约 0.81%。含归一化的编译路径出现差异；本轮没有进一步把误差唯一定位到某条底层指令，因此不将其泛称为某一个 kernel 的 bug。

MLX 编译允许算子融合，但性能、浮点运算行为需要在本模型中实测：[官方编译说明](https://ml-explore.github.io/mlx/build/html/usage/compile.html)。本轮不为了降低局部时间而放宽现有精度要求。

## 端到端复测

| 100 对交替测试 | 原默认 ms | 候选 ms | 节省 ms | 节省的 95% 区间 |
|---|---:|---:|---:|---:|
| 第一轮 | 280.775 | 279.935 | 0.840 | [0.416, 1.253] |
| 独立进程复测 | 285.633 | 285.809 | -0.176 | [-0.841, 0.471] |

两轮所有动作输出一致。第二轮不支持稳定收益，不能仅引用第一轮宣布加速，更不能声称图像编码已达到 PyTorch 的约 22 ms。

正式可选入口独立验证（8 次预热、30 次稳态、10 次 profile）：加载 2.620 s，首次 1128.615 ms，稳态平均 **281.370 ms**、P95 **283.909 ms**；图像编码同步阶段平均 **42.088 ms**。最终动作与原默认入口逐字节一致。该独立数值不替代上面的配对收益判断；局部独立 VAE 基准与全模型 profile 的调度/内存状态不同，不能把 41.075 与 42.088 当作回归。

## 精度与实现

完整 51 个离线案例及输入扰动/动态文本切换生成的 **162 个输出文件**，与优化前 MLX 默认记录逐字节一致。另检查 4 个真实观测的 **40 个残差块中间输出和 4 个完整 latent**，全部逐字节一致。相对 PyTorch 的既有最大动作绝对差约 0.01580137，没有增加；仍未评估闭环任务成功率。

`imagewam_mlx/compiled_vae.py` 实现 `CompiledResidualVAE`：保留两个 GN+SiLU 原计算路径；只编译第二个卷积、必要的 shortcut 和残差相加。每层拥有独立编译函数，按实际张量形状由 MLX 管理编译，不固定文本长度，也不缓存任何输入特征。

`ImageWAM(..., vae_tail_compile=True)` 显式启用；默认 False。legacy、关闭 fused_vae 或启用另一实验 vae_group_inv 时不叠加此方案，实际状态记录在 benchmark 的 `vae_tail_compile` 字段。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
# 正式可选入口：输出目录必须不存在。
.venv/bin/python scripts/benchmark.py --vae-tail-compile \
  --reference ../ImageWAM/evaluate_results/m5pro_robotwin/20261103_dynamic_profile \
  --output records/new_vae_tail_benchmark --warmup 8 --iterations 30 --profile-iterations 10

# 当前正式默认：不传 --vae-tail-compile。
# 完整回归候选（独立测试包装入口）
.venv/bin/python scripts/run_vae_tail_validation.py --output records/new_vae_tail_validation
# 同期配对实验
.venv/bin/python scripts/test_vae_tail_e2e.py --output records/new_vae_tail_pairs --pairs 100
```

## 原始记录

- `records/vae_granular_20261103`：四项初筛，含 baseline/candidate latent。
- `records/vae_res_safe_20261103`：80 对图像编码测试。
- `records/vae_tail_e2e_20261103`、`records/vae_tail_e2e_repeat_20261103`：两轮 100 对端到端原始时间及动作。
- `records/vae_tail_validation_20261103`：完整输入回归结果和最终动作。
- `records/vae_tail_checks_20261103`：逐层精度及 4 个 latent。
- `records/vae_tail_native_20261103`：正式可选入口基准、每轮输出、源码快照与中间 trace。
- `records/vae_tail_source_20261103`：修改前入口备份。

上述路径都在 `/Users/qklj/work/cx/ImageWAM_MLX` 下。后续仍需围绕归一化归约与卷积实现研究更大的收益；本轮已验证的简单改写不足以稳定缩短端到端延迟。
