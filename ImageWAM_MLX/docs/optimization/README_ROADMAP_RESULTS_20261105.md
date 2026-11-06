# 新路线适配与验证报告（2026-11-05）

本轮在 M5 Pro 上完成视觉真实张量诊断、Split-K候选、51组及动态逐步回归、12进程独立复测、正式CLI复测，并补做编译入口控制实验。生产speed默认保持原四SIMD路径；新增显式 `--visual-shape-dispatch` 供复现。没有更改模型权重、步数或跨请求缓存策略。

本报告不将GPU捕获成功写成GPU性能分析完成：远端缺少完整Xcode/xctrace，GPU时间线/硬件计数器分析仍未完成。其余数据来自真实模型执行及同步主机计时，不能据此断言带宽或GPU占用是瓶颈。

## 测试范围与变更

M5 Pro，15 CPU /16 GPU，24 GiB统一内存；macOS27.2，MLX0.32.4，Python3.12.13。RoboTwin三张已解码图、状态、文本和显式固定噪声；标准性能输入39 token、H16；10步在线推理，无跨请求特征缓存，计时到CPU反归一化动作返回。加载、读图和落盘不计稳态。测试期间没有主动停止系统或其他用户服务。

新文件 `imagewam_mlx/shape_visual.py` 在 `(M,N,K)=(328,27648,3072)/(328,3072,12288)` 选择已有正确的双SIMD64行实现，其他shape回退原四SIMD；不固定/截断输入文本，不复用上一请求的编码结果。标准样本38个匹配投影均命中，因此该输入下与已有双SIMD64行实现等价，不包装成新的矩阵算法。`runtime.py`、`infer.py`、`benchmark.py`加入显式参数和结果元数据，默认关闭，与全局 `visual_group_rows=32/64` 互斥。

## 真实张量诊断

38次视觉投影（两种shape各19次）输出全部逐值一致。孤立同步微测累计：第一种shape原31.066 ms、双SIMD29.292 ms；第二种原18.056 ms、双SIMD16.234 ms。测量包含主机提交/同步，逐层顺序测量不是随机配对，不能将累计节省直接当作整网收益；候选选择后另做整网验证。

动作热点为 `(M,N,K)=(16,1024,7168)`，每步20次，10步200次。target512对应28分块；target128/256/1024分别对应8/16/56分块。每次FP32 partial逻辑写入+读取约3.67/1.05/2.10/7.34 MB（按28/8/16/56分块排序），不是DRAM计数器实测。

首个真实张量微测中8/16块与28块相同，但完整请求全部出现数值差异。这说明不能用一个张量的相等结论替代全模型回归。分组变化改变浮点归约路径；尚未逐条定位第一处底层算术差异，不把它直接判定为内核错误。

独立同步组件诊断中，Split-K中位0.228 ms、归并0.171 ms、组合0.210 ms，不能相加。主机调度/同步成本影响孤立测量；不能声称归并占实际GPU时间的81%。

实际KV缓冲区 `[1,24,344,128]` 大小2,113,536字节，保留旧引用的slice_update实测新增2,113,548字节分配。只证明该保留引用场景会新增分配；不证明正常编译整网的500次更新都发生全量物理复制。本轮未据此进行不安全的原地写或重复既有失败KV结构改造。

## 配对筛选与正确性

每候选预热5次，随机顺序60轮完整请求：

|候选|平均ms|标准输出|51组一致数|
|---|---:|---|---:|
|原speed|193.300|参考|—|
|视觉shape分派|191.346|逐值一致|51/51|
|Split-K target128|192.291|不同|0/51|
|Split-K target256|192.978|不同|0/51|
|Split-K target1024|194.677|不同|0/51|

三种Split-K候选51组最大绝对差分别0.022613、0.032003、0.022613，不推广到无新增误差路径。保留实验代码便于研究，不接入默认配置。

既有51组文本覆盖31–79 token；补充189-token、H1/H16/H64，以及相机降采样/状态扰动。视觉候选最终动作和全部19项trace一致，包含十步动作、输入/latent/text及抽查KV。H1走原小M分支，Split-K候选在H1相等不能证明H16/H64正确。数值一致性均相对当前量化speed，不等于原始PyTorch精度或机器人闭环成功率。

## 独立进程与正式入口

12进程复测为基线/候选各6个进程，每进程预热30次、稳态100次，随机平衡顺序，保存全部进程与CPU数据，不删除慢状态。worker基线202.912 ms、候选199.917 ms；P95分别206.233/204.114 ms。六组差值均正，平均节省2.995 ms，按进程配对bootstrap区间[2.038,3.827] ms；1200次稳态输出一致。

但worker的候选重包了prefill编译函数，而正式入口使用绑定方法，因此这项比较包含编译入口差异，不能单独归因为shape分派。

随后正式 `benchmark.py` 执行基线→候选→候选→基线，每进程预热30次、稳态100次，候选仅使用新增启动参数。基线201.186 ms、候选200.849 ms；P95分别203.659/203.640 ms。正式改善仅0.337 ms（0.168%），低于推广门槛，400次稳态输出一致。不得拿worker的2.995 ms作为已获得的正式部署收益，也不能跨轮与历史194.468 ms相减。

## 编译入口控制与正式模块耗时

8个独立进程，四种组合各两次，正反顺序，每进程30次预热/100次稳态，800次动作输出全部一致：

|布局/编译入口|平均ms|P95 ms|
|---|---:|---:|
|base_bound|201.028|202.838|
|base_lambda|201.423|203.776|
|shape_bound|200.030|202.287|
|shape_lambda|199.852|201.834|

固定bound入口时布局改善约0.998 ms，固定lambda时约1.571 ms。仅改lambda在基线上更慢约0.395 ms，在shape候选上快约0.178 ms；两进程/组合仅作控制诊断，不足以断言编译入口是所有差异的根因。不得将独立worker、控制实验、正式CLI的绝对耗时拼接为加速曲线。

正式CLI另测同步模块profile（每配置10次，改变重叠关系，不能相加作为端到端）：

|模块|原speed ms|shape候选 ms|
|---|---:|---:|
|tokenization|0.263|0.308|
|text_encoder|28.270|28.779|
|preprocess|3.513|4.837|
|image_encoder|28.107|28.036|
|visual_prefill|77.801|77.958|
|action_denoise|62.522|63.005|
|postprocess|0.187|0.199|

|配置|加载秒（两进程）|首次前向ms（两进程，含首次编译）|
|---|---|---|
|baseline|3.229, 3.066|911.074, 824.991|
|shape|3.064, 3.077|861.514, 831.707|

## 集成与精度验收

显式参数接入后，重新执行51组样本，与此前逐值一致的golden输出比较全部通过；另做四组动态输入，全部十步动作及19项trace一致。正式speed默认保持原实现。候选不加载新权重，不缓存输入特征，不改变文本长度或采样步数。实验与正式集成源码、修改前备份、计时、输入与动作结果均保存在本轮records目录。

## GPU捕获与尚未完成项

在完整模型进程内捕获单个真实投影仍记录了全部驻留资源，生成18,329,936,658字节记录。随后导出该投影真实输入，在独立进程只加载必要张量重新捕获，生成216,139,127字节（约206 MiB）记录。已清理本轮冗余大副本，保留大小/替代原因清单；原历史记录未删除。可用捕获为 `capture/isolated_projection.gputrace`，真实输入为 `projection_inputs.npz`，可通过export_projection.py/capture_isolated.py复现。

远端仅有Command Line Tools、没有xctrace。本地发现Xcode16.2；实际执行xctrace import，正常缓存权限下返回exit 11：Cannot find matching importer。说明该CLI没有匹配的.gputrace导入器，不等于已证明Xcode GUI无法打开。尚未完成兼容的GPU调试器重放/解析流程；不能把本机重放时长当作M5 Pro执行时长。本轮未获得完整CPU/GPU关键路径、真实物理复制计数或带宽/占用计数器。因此P0只完成了计时协议、工具检查与最小捕获，GPU时间线分析未完成，不宣称“所有方向均已验证完毕”。

P1完成真实shape诊断、候选接入与完整测试；P2完成分块/归并与KV诊断，分块候选因误差淘汰，KV没有足够证据支持新的结构性修改。P3的文本/VAE暂不新增优化，既有失败方案不重复扫参。模型加载继续按要求不优化。

## 启动与复现

生产默认（本轮正式均值201.186 ms，历史194.468 ms为另一时段测量）：

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
.venv/bin/python scripts/benchmark.py \
  --profile speed --text-cache-size 0 --steps 10 \
  --reference examples/robotwin_ep0_frame0 \
  --warmup 30 --iterations 100 --profile-iterations 5 \
  --output "records/speed_baseline_$(date +%Y%m%d_%H%M%S)"
```

显式实验候选（本轮正式均值200.849 ms，尚无达到推广门槛的收益）：

```bash
.venv/bin/python scripts/benchmark.py \
  --profile speed --visual-shape-dispatch --text-cache-size 0 --steps 10 \
  --reference examples/robotwin_ep0_frame0 \
  --warmup 30 --iterations 100 --profile-iterations 5 \
  --output "records/speed_shape_$(date +%Y%m%d_%H%M%S)"
```

`infer.py`同样支持 `--visual-shape-dispatch`；省略该参数保持原speed。它不能与全局双SIMD `--visual-group-rows 32/64`同时启用。accuracy默认profile不变。

全部实验脚本位于 `experiments/roadmap_20261105`，结果位于 `records/roadmap_20261105`。复现实验需指定新的记录目录，脚本拒绝覆盖既有测试。主记录包括diagnostic.json、screen/、independent_summary.json、cli_summary.json、factorial_summary.json、integrated_validation/results.json及capture/。正式CLI保存各次动作和源码快照；动态trace保留远端，审阅压缩包可省略大trace。

下一步应补齐兼容M5 Pro的GPU分析工具链后，针对真实占用/访存/提交间隙做定点修改。现有结果不支持继续盲扫更低Split-K或宣称已稳定达到191 ms。

补充入口烟雾测试：infer.py使用新参数运行成功，动作与原speed逐值一致。单次请求191.975 ms不计入正式稳态统计，不作为新的最佳速度。
