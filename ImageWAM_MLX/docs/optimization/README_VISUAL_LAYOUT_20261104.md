# 视觉大矩阵与调制融合实验（2026-11-04）

本轮完成真实形状诊断及四个候选的端到端测试，没有确认新的加速收益。正式 `--profile speed` 仍使用 [已验证最佳版本](README_BEST_SPEED_20261104.md)，未改动其代码、模型或 accuracy 分支。历史独立最佳均值 226.641 ms；本轮后台负载较高，不能拿历史值直接计算本轮回退幅度。

## 环境及测量方法

M5 Pro，15 CPU / 16 GPU 核，24 GiB 统一内存，MLX 0.32.4，Python 3.12.13。运行目录 `/Users/qklj/work/cx/ImageWAM_MLX`。设备信息和源码 SHA256 见 [metadata.json](../../records/visual_layout_20261104/metadata.json)。

基线是正式 speed（包含 W1 action、自适应 NAX Split-K epilogue、KV workspace 等全部已验证优化）。各候选共享相同冻结权重，视觉保持 BF16。三张真实 RoboTwin 图像、在线文本编码、状态和相同初始噪声，16×14 动作，10 步去噪。每次完整执行，不缓存输入特征。端到端包含预处理到 CPU 反归一化动作，不含磁盘读取、模型加载或输出落盘。

两组实验各三个配置；每配置预热 5 次，100 轮随机顺序配对请求，计时前后同步。区间由逐轮基线减候选的耗时差做 10000 次 bootstrap 得到，仅描述本轮，未校正时序相关性或多重比较。另保存每配置 5 次同步模块分析，模块时间不能求和替代无模块同步的端到端。

测试期间观察到 mediaanalysisd 约 106% CPU、视频编解码服务和虚拟机活动，系统 memory_pressure 报 free 18%。未干预这些服务；没有 GPU timeline，不能认定某一后台服务占用了多少 GPU。结束时进程列表保存在 records 中。

## 实际形状与瓶颈

标准样本为 39 个文本 token，加状态后 40 个文本侧 token；288 个图像 token；单流块共 328 个 token。对实际 prefill 的 Linear 调用按形状分组，选取每组首次真实输入，比较原始、显式二维及转置连续存储三种矩阵布局，每种预热 3 次、测量 20 次。

|视觉投影|单次输入与权重形状|次数|总计算量（GFLOP）|
|---|---|---:|---:|
|单流 linear1|[1,328,3072] × [27648,3072]ᵀ|19|1058.626|
|单流 linear2|[1,328,12288] × [3072,12288]ᵀ|19|470.500|
|双流图像 MLP 输入|[1,288,3072] × [18432,3072]ᵀ|5|163.075|
|其余视觉 Linear|详见 probe.json|—|254.125|
|全部视觉 Linear|—|—|1946.326|

按每个乘加 2 FLOP 统计，不含 attention、归一化、激活和其他模块。两类单流投影占视觉 Linear FLOP 的 **78.57%**；这是运算量占比，不是实测 GPU 时间占比。每个大 GEMM 仅按输入、权重、输出各读写一次估算的算术强度约 289–293 FLOP/byte，实际访存量仍需 profiler 验证。

微测中单流输出投影中位数原始 3.0925 ms、二维 2.9162 ms、连续转置 2.9419 ms。微测包含独立提交/同步开销，只取每形状首个层，不能乘次数后当作完整图耗时；后续端到端未证实这个局部收益。转置连续权重在部分文本投影还改变结果（激活最大绝对差 0.25）并变慢，没有为整网增加转置副本。

## 完整请求对照

“节省”正值才表示候选更快。数值差异是相对同轮正式 speed 的最终反归一化动作，不是相对 PyTorch 或真实任务成功率。

|候选|同轮基线 ms|候选 ms|节省 ms|节省的 95% 区间 ms|样本动作最大绝对差|
|---|---:|---:|---:|---|---:|
|仅单流输出投影展平二维|262.003|264.072|-2.069|[-9.013, 4.795]|0|
|所有三维视觉投影展平二维|262.003|260.264|1.740|[-4.375, 7.594]|0|
|单流 LayerNorm 与调制融合|261.611|267.125|-5.514|[-12.071, 0.927]|0.0160012|
|双流和单流 LayerNorm 与调制融合|261.611|259.592|2.020|[-4.599, 8.641]|0.0160012|

所有区间跨零，不晋升默认。后两种利用原生 `mx.fast.layer_norm(x, weight=1+scale, bias=shift)` 合并调制，减少独立点操作，但改变 BF16 舍入边界。这里的“双流和单流”涵盖重写的 double_prepare/double_post/single_prepare，最后 compact KV-only 层仍保留原表达式。

二维候选按实际形状 reshape，未固定文本长度；融合候选同样没有截断或 padding。由于未获得可靠性能收益，本轮只验证标准样本，不声称完成全部文本长度/动作 horizon 的精度回归，也不宣称融合无损。

原始计时、模块记录及输出：
- [矩阵诊断](../../records/visual_layout_20261104/probe.json)
- [二维投影对照](../../records/visual_layout_20261104/paired/results.json)
- [调制融合对照](../../records/visual_layout_20261104/norm/results.json)
- 两个子目录各有 baseline 和候选的 `.npz`，包含 action、normalized。

## 复现命令

远端已整理到 `experiments_visual_layout_20261104/`，本地镜像为 `experiments/visual_layout_20261104/`。输出目录必须不存在；复测需修改脚本中的 root 为新目录，避免覆盖记录。按顺序执行，不并行运行 GPU 测试。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
# 微测，没有完整端到端数值
.venv/bin/python experiments_visual_layout_20261104/probe.py
# baseline / flat_output / flat_all: 262.003 / 264.072 / 260.264 ms
.venv/bin/python experiments_visual_layout_20261104/paired.py
# baseline / single_norm / all_norm: 261.611 / 267.125 / 259.592 ms
.venv/bin/python experiments_visual_layout_20261104/norm.py
```

## 下一步判断

当前优先级是为真实形状的单流 linear1/linear2 测试 BF16 NAX GEMM tile 与融合 epilogue，保留矩阵乘法并行度，避免为了减少一个小算子却退化大 GEMM 的吞吐。先在真实输入上确认局部收益，再做完整请求验证；本轮没有实现这些自定义 GEMM，不能把它们计为已有收益。

后台负载明显时，约 1–2 ms 的候选不足以晋升；需在更稳定时段独立复测。现有全视觉量化、行补齐、连续布局和小算子融合结果说明，降低存储位宽或减少表面算子数量并不自动提升实际速度。仍须分别核查权重解码、累加、输出舍入与内核调度成本。
