# 常规优化方向覆盖审计与新分析手段（2026-11-05）

结论：常见计算优化的大部分类别已有适配和实测，但不能说全部做完，更不能说都在最新speed上通过验证。已实现、历史试验、最新默认、正式入口确认收益是四个不同状态。本轮只做官方资料检索、源码/实验记录审计，没有运行新的推理基准或修改推理默认。

审计对象为本地项目镜像及最近已同步的m5pro生产代码。未发现测试证据表示在本次检索范围内没有相应代码/记录，不是证明所有历史工作中绝对未尝试。没有用关键词出现或API支持来冒充完整模型测试。

## 1. 常规方向对照

|方向|覆盖情况|当前默认/结论|主要项目证据|
|---|---|---|---|
|去除未使用层、输出与权重|已适配并回归|compact默认；Qwen后续层/LM head、VAE decoder、视觉末层无用分支排除，后续K/V与VAE通道裁剪已落地|README_COMPACT_ACTION_20261103.md、model.py|
|混合精度、权重量化|已做多组合整网测试|speed采用W1动作及混合精度；不是全图整数|README_SPEED_QUANTIZATION_20261103.md、quantized.py|
|真正INT8权重/激活GEMM|已适配，后续多轮优化|当前38个适用视觉投影为INT8输入、INT32累加、BF16输出|README_RUNTIME_TUNING_20261104.md、device_visual.py|
|MXFP4/8、NVFP4及动态激活量化|有API与早期整网测试|不是未尝试；没有证明这些组合优于最新speed|README_QUANTIZATION_ANALYSIS_20261103.md、README_SPEED_QUANTIZATION_20261103.md|
|图编译、去除多余同步|已适配|prefill/动作step编译；移除prefix强制求值等已保留|runtime.py、README_GRAPH_KV_20261103.md|
|跨2/5步合图、减少async提交|已做完整请求测试|无稳定收益，未默认|README_GRAPH_KV_20261103.md、README_GRAPH_DISPATCH_20261105.md|
|GN/SiLU、gate/concat/量化、epilogue融合|多个方案已适配|有效子集保留；不是所有融合均有效|fused_vae.py、gate_quant_visual.py、action_epilogue.py及对应README|
|RMS/RoPE、QKV/gate-up合并|多次适配测试|部分有数值变化或无稳定收益；不能重复列为全新方向|README_RMS_ROPE_20261103.md、README_ENCODER_QKV_20261103.md|
|静态权重预打包，避免前向整权重反量化|已适配|预展开INT8、直读权重、K片段展开为speed默认|README_PREPACKED_INT8_20261104.md、README_DEVICE_NAX_20261105.md|
|分块、Split-K、SIMD数、尾块、遍历顺序|广泛真实张量与整网测试|部分有效；新shape分派显式可选，最近Split-K变更因误差排除|README_ROADMAP_RESULTS_20261105.md、README_INT8_TAIL_20261105.md|
|双缓冲、预取式流水|已适配并复测|没有稳定整网收益|README_INT8_PIPELINE_20261105.md|
|attention融合、双段KV/Steel/NAX attention|多种适配与整网测试|默认保留已验证attention；force_fused无收益|README_SEGMENTED_KV_20261104.md、README_STEEL_KV_20261104.md、README_ACTION_CONDITION_20261105.md|
|KV工作区、复用与减少拼接|已适配|功能式slice_update工作区默认；真实物理复制仍未完整量化|workspace.py、README_STRUCTURAL_20261104.md、README_ALL_DIRECTIONS_20261105.md|
|INT8 KV/attention|已有完整动作路径实验|早期量化QK/PV含量化边界成本，未优于当时基线；非当前默认|README_LOWBIT_VALIDATION_20261104.md|
|VAE量化卷积、卷积转GEMM|已适配实验|im2col+INT8 GEMM完整VAE测试无收益；不是原生全整数Conv|README_LOWBIT_VALIDATION_20261104.md|
|VAE padding、残差编译、归一化查表|已适配并复测|部分可选；近轮padding/融合无整网收益|README_VAE_DEEP_20261103.md、README_ONLINE10_20261105.md|
|CPU预处理与GPU文本重叠|已适配|默认启用|runtime.py:predict|
|文本/VAE双GPU stream|已做同调度单流控制与整网回归|无稳定收益；这不是尚未尝试的方案|README_DUAL_STREAM_20261104.md|
|command buffer阈值、QoS、编译包装方式|已多轮测试|buffer256保留；其他开关无稳定收益|README_GRAPH_DISPATCH_20261105.md、README_ROADMAP_RESULTS_20261105.md|
|真实shape预热|已适配动态输入|有warmup接口，不等于shapeless或持久化机器码缓存|warmup.py、README_GEMM_DYNAMIC_RESULTS_20261103.md|
|文本缓存、减少去噪步数|已显式实验|违反本轮无缓存10步比较条件，不计作同条件加速|README_ALL_DIRECTIONS_20261105.md|
|系统性隐式连续化/拷贝审计|局部支持stride，未见完整审计|下一步高优先级|本轮kernel_contiguity_static.json|
|shapeless、导出/导入、C++调用路径|已有建议，未见完整模型A/B证据|不能标记为已完成；先验证自定义算子兼容性|README_REMAINING_OPPORTUNITIES_20261103.md及源码检索|
|进程级wired/cache预算|有内存诊断，没有策略A/B|无稳态换页证据，暂低优先级|README_GEMM_MEMORY_NEXT_20261103.md|
|GPU运行时时间戳/Counter、寄存器占用|capture存在，解析和计数器未完成|运行时原生探针是可探索替代路线|README_ROADMAP_RESULTS_20261105.md|
|稀疏化、token剪枝、少步蒸馏训练|未见适配/训练证据|改变模型或语义，单独精度路线|README_ALL_DIRECTIONS_20261105.md|
|多请求batch/流水线服务|未见生产适配与吞吐测试|目标主要为吞吐，不等于单请求延迟优化|当前单样本入口|

早期低比特、双流等报告基于当时的模型路径，而非今天的最终组合。历史失败不能证明永远无效，但重新打开方向应有新的shape、数据布局或瓶颈证据，不能把全部失败组合再扫一遍。

## 2. 新优先级A：隐式连续化、拷贝和布局审计

MLX官方说明 `mx.fast.metal_kernel` 的 `ensure_row_contiguous` 默认True；非连续输入必要时会被复制。关闭它需要内核正确按stride索引，不能直接批量改False。[官方自定义内核文档](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)

本轮AST检查 `imagewam_mlx/*.py`：发现23个直接声明的metal_kernel，其中18个未显式设置此参数，5个设False。这包含可选实现，不等于23个都运行于speed，也不等于18个每次都复制。已有RoPE、GN和gate量化的stride处理说明局部做过优化；缺口是当前实际热路径的完整审计。

具体方法：只对运行中的内核记录shape/stride/连续性、输入来源、必要复制字节及调用频次；结合未求值图检查transpose→reshape、concatenate、slice→custom-kernel边界。优先改一处真实重复复制，保持任意已支持文本长度和精度，不统一展开整个模型。

MLX提供 `export_to_dot`，递归展示输出的未求值依赖。它可辅助查重复节点/依赖，但不是GPU执行图或物理复制证明；后端插入的拷贝还需原生日志或运行证据。[官方DOT接口](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.export_to_dot.html)

## 3. 新优先级B：运行时GPU时间戳，不再只等待gputrace导入

Apple提供已完成command buffer的 `GPUStartTime/GPUEndTime`，可用于测该buffer的GPU执行区间；Counter API还可在设备支持的采样边界获取更细粒度时间信息。[Apple时间戳接口](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime)、[运行时Metal Counters说明](https://developer.apple.com/videos/play/tech-talks/10001/)

当前项目尚无该适配。建议先做独立诊断构建/原生探针：记录CPU编码/提交时刻、GPU开始/结束时刻、buffer编号和所属阶段，先回答GPU是否在等提交。可记录为JSON，再画CPU/GPU时间线，无需依赖本地xctrace导入.gputrace。这仍需要验证SDK、设备采样能力和MLX后端接入方式，不承诺零成本实现。

边界：一个command buffer可能包含多个算子，其时间不等于某一层kernel耗时；阶段边界采样也未必支持逐dispatch采样。跨队列重叠时不能直接求和。Counter能力需查询，不假定所有带宽/占用指标均开放。插桩环境与正式性能环境分别测试，不能每核synchronize改变原调度后再称原关键路径。

## 4. 新优先级C：CPU构图、编译和分配成本拆解

现有工作做过sample栈、进程CPU时间、编译包装及buffer阈值，不等于已拆出Python构图、MLX图优化、Metal编码、内存分配的准确成本。

可先用Python调用剖析限定高频Python部分，再用后端阶段计时/分配日志看原生部分。cProfile看不到GPU成本，等待栈不是CPU占用，GC开关也不能靠猜测启用。统计节点/临时数组/提交次数，再在无插桩整网验证改变是否有效。

若CPU构图实测占关键路径，可尝试先导出/导入一个动作step，再比较同数学图的Python和原生调用。MLX官方支持函数跨Python/C++前端使用，但当前自定义Metal算子可否序列化、权重绑定、动态shape和返回结构必须先做小型可行性测试；导出不是GPU图回放或持久化Metal机器码的同义词。[MLX函数导出说明](https://ml-explore.github.io/mlx/build/html/usage/export.html)

## 5. 新优先级D：混合长度请求与编译缓存

已有31–79及189/193-token等功能验证、真实样本预热，但未见“shape切换频率→重编译次数→生产P95”的系统化实验。

先用真实长度序列交替请求，分开首次shape、重复shape及持续稳态。可对实际shape预热并管理编译缓存，不截断、统一填充或修改attention mask来偷换输入。

`shapeless=True`可以避免部分shape变化导致的重编译，但官方明确警告shape相关Python控制流/reshape常量可能失效。当前代码恰有shape专用内核和分派，不能给整个predict加开关就认为可用。仅对适合的小子图验证，其主要收益可能是切换/首次延迟，而非固定39-token稳态。[MLX编译文档](https://ml-explore.github.io/mlx/build/html/usage/compile.html)

## 6. 其余方向如何排序

- 寄存器压力/occupancy：已有线程数/tile试验，但未测寄存器溢出和资源占用。Apple强调内存空间、数据类型与资源使用会影响占用率；应先实测再调线程块，不机械认为线程越多越快。[Metal shader最佳实践](https://developer.apple.com/videos/play/tech-talks/111373/)
- 分配器cache/wired预算：有API且尚未做策略A/B，但历史稳态缺页/换页没有支持证据。仅在内存压力与尾延迟关联明确时测试进程级预算并恢复，不先改系统sysctl。[MLX wired接口](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html)
- 量化KV：不是新空白项，已有全动作路径试验。只有长度或attention数据流发生实质变化，才重开；不能将长上下文LLM经验直接外推到本模型的短前缀和H16动作。
- 全图整数：已测试多种低比特组合；softmax/归一化/激活等仍有浮点边界，不以更低位宽保证提速。
- 剪枝/蒸馏/减少步数：可能有更大收益，但需要训练/校准与RoboTwin闭环评价，不能混入本轮保持speed输出的10步优化。
- 多请求batch/流水线：适合吞吐目标，需另测排队延迟、单请求P95及状态隔离，不能用吞吐倒数冒充响应延迟。

## 7. 下一轮建议

顺序为：热路径stride/隐式复制清单 → command buffer原生时间戳可行性 → CPU/原生构图与分配剖析 → 有证据后的定点改动 → 混合长度测试。若时间戳接入暂时受阻，前两类静态/运行时输入审计仍可推进，不能把所有优化都阻塞在Xcode上。

本轮没有新加速数值。最近正式同轮基线201.186 ms、shape候选200.849 ms仍按原报告解释；新方案需继续通过动态输入、逐层/最终输出、独立进程及正式CLI复核。详见 [上一轮完整结果](README_ROADMAP_RESULTS_20261105.md)。
