# 前缀与10步动作的低精度优化分析

依据当前正式缓存命中路径：前缀77.396 ms、动作10步62.518 ms、正常端到端169.401 ms。前两项来自独立同步剖析，不直接与正常端到端相加。本文为代码/历史测试分析，没有实施新量化或新增性能实测。

## 前缀：优先扩大有效INT8计算范围

model.py/prefill由5个double blocks和20个single blocks组成，最后single block只生成K/V；输入包括Qwen特征、机器人状态、图像latent。当前38个single linear1/linear2使用INT8权重/激活、INT32累加、BF16输出。其余double-block QKV、attention输出投影、MLP、txt_in/img_in及最后KV-only投影仍主要为BF16。

当前视觉INT8权重源于W4 group64，加载时已预展开为INT8。因此把源W4改为W2/W1、再展开同样的INT8，不会减少常驻INT8字节数或GEMM位宽；不承诺提速。要真正W4A4/W2A8，需新的低位计算内核，不能仅调整mx.quantize参数。已有staged解包模板不包含bits=1。

优先实验：保持现有38层路径不变，仅对剩余BF16投影按形状替换。先double-block图像MLP/大投影，再QKV及KV-only层；小文本流和窄投影允许回退。冻结权重预量化一次、激活量化融合在输入边界、INT32累加后一次输出BF16。不在每层先把整块权重还原BF16。

其次探索现有INT8量化的校准固定尺度/分组尺度：减少逐行absmax的代价，或与前置归一化融合。它不降低INT8位宽，而是减少量化开销；固定尺度会带来饱和风险，必须用多种文本/观察校准并统计clip比例，未知输入不能静默复用错误范围。单层局部节省可能被GEMM占比掩盖。

BF16改FP16仍是16位，不自动节省带宽；改变指数范围还可能溢出。归一化/softmax统计不建议先全改低精度，节省未知但影响可能传播至全部KV和10步动作。

## 动作：W1已低，瓶颈不等于权重位宽

workspace_step每步包含5个double+20个single，10步共250个block调用/attention调用。各步依赖上一步动作，不能直接把10步作为独立batch并行。当前65个合格Linear采用W1A16；大K输出投影有Split-K和融合epilogue。常见M=16，不能直接套用前缀M约328的大矩阵INT8布局。

降低激活到INT8需要激活量化+小M专用矩阵内核。如果把W1预展开INT8，核心权重编码部分从1bit变8bit，未计scale时体积约8倍；可能用更多带宽换计算吞吐，必须实测。W1码也不是可以直接与INT8激活做普通二进制点积的完整权重，还包括分组浮点scale/bias。

可以按真实形状比较W1A16、W2A16、W4A16、专用W8A8，而不是强制位宽越低越优；W2/W4可能选择到更好的内核，但它们不是进一步降精度。历史小M原生INT8动作实现未获整体收益，只有出现新的小M布局/融合实现才值得重试。

较低优先级：Split-K部分和存储FP32→FP16/BF16。保持FP32乘加，仅存储低位会减半partial流量，却增加舍入，且不减少矩阵乘本身；FP16需检查溢出，BF16有效位更少。当前代表投影28个split的partial为1.75MiB，改16位为0.875MiB；这些是逻辑数组大小，非实测DRAM流量。前两轮归约/主循环微调整网只改善约0.3ms，不应期待仅压缩partial就带来几十ms收益。

## KV与attention：只有消费端也改变才可能省时间

标准前缀约328token、24头、head_dim128、25层，K/V采用16位时仅数据约96.094MiB（不含动作槽和尺度）。INT8理论数据量约减半，但每步完整反量化后再调用浮点SDPA可能抵消收益。

真正的候选是量化KV常驻，并在融合attention内按tile读取/解码，保留浮点softmax/稳定累加，避免10步反复物化完整BF16 KV。已有INT8 QK/PV分开实现、浮点softmax的旧实验未获益；不是再加一个astype就能解决。M=16、短前缀下，量化与多次kernel调度尤其容易比节省的读取更贵。

当前已缓存纯文本特征；多模态第二层起文本表示包含当前图像/状态，不能按相同指令缓存全部多模态KV。缓存旧观察属于另一种近似算法，需要单独定义语义。

## MLX 0.32.4依据与已有反证

固定版本Metal QQMatmul::eval_gpu仍调用quantize_dequantize_input，返回与输入相同dtype的浮点张量，再进入量化权重matmul。不能把qqmm名字解释为全程低位激活流水：
https://raw.githubusercontent.com/ml-explore/mlx/v0.32.4/mlx/backend/metal/quantized.cpp

官方quantized_matmul接口：
https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html

既有低比特筛选（旧版本，不与当前169ms跨轮比较）：current_speed241.872 ms，native_action326.017 ms，int8_kv277.153 ms，all_lowbit370.687 ms。完整说明见README_LOWBIT_VALIDATION_20261104.md。这说明旧实现较慢，不证明所有专用低位内核都不可能更快。

## 优先级与验证门槛

1. 前缀剩余BF16投影：先捕获实际形状与子模块耗时，逐类试原生W8A8，保留既有38层路径。
2. 前缀激活量化边界：比较动态量化融合和校准尺度，统计饱和、转换/归约开销。
3. 动作小M内核：按形状选择位宽和布局，必须计入激活量化、权重展开常驻内存及所有输出转换。
4. KV量化融合attention：高工程成本，以不物化完整浮点KV为前提。
5. 部分和16位存储：低成本探索，但预期收益上限未知，优先级低于热点矩阵乘。

验证保留三图、动态文本、H16/10步和当前缓存命中口径；10次预热/50次稳态，另测50次模块，单项→组合→独立进程复核。至少覆盖51代表输入及长文本/状态/图像扰动，记录相对当前speed和BF16参考的MAE/RMSE/max_abs、逐步动作误差、NaN/Inf。低精度候选不要求逐值一致，但必须显式报告差异，不默认为精度不受影响。若只优化一个模块，不把另一个模块代码悄悄切回旧baseline。

仅作时间预算示例：前缀若快10%，模块约少7.74ms；动作若快10%，约少6.25ms。二者若同时实现，简单加法约少13.99ms，但同步剖析与端到端不可直接相加，不能把约155ms当成已测或保证目标。
