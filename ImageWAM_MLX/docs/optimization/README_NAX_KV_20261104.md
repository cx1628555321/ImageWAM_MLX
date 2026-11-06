# NAX 双段 KV attention 实验（2026-11-04）

基于 MLX 0.32.4 的 steel_attention_nax.h 实现双段 KV。保持 NAX 硬件矩阵乘与在线 softmax；完整落在前缀或动作段的 fragment 使用原 load，只有跨段/尾部 fragment 做逐元素带界限寻址。默认 BQ64/BK32/WM4/WN1。无前缀与动作的完整拼接张量，query<=8则明确回退官方SDPA+拼接以保持原vector路径精度。

官方调度源码在 https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/scaled_dot_product_attention.cpp 。根据源码，本机NAX可用且BF16/head_dim128/query>8满足NAX分支条件；这里是源码依据，未捕获kernel时间线确认。上一轮普通Steel分块未匹配该路径。

模型以当前speed为基础，动作W1、视觉/Qwen BF16、fast VAE不变。候选关闭原工作区，改用双段attention。M5 Pro 15CPU/16GPU、24GiB统一内存、macOS27.2、MLX0.32.4。

## 筛选性能

ABBA四个独立进程，各首次前向+3次预热+20次稳态+3次同步模块分析，每配置40次。真实三图、完整动态文本、状态、16×14动作10步。端到端含CPU输出，不含加载、文件解码、保存。

|配置|端到端平均 ms|P95 ms|两轮均值 ms|动作模块 ms（独立同步）|
|---|---:|---:|---|---:|
|baseline|240.867|241.517|241.033 / 240.702|75.282|
|segmented|240.760|241.283|241.001 / 240.519|76.055|

## 数值与动态输入

逐项动作误差相对本轮speed，不是BF16/PyTorch真值或任务成功率；包含39-token/horizon8/3步、193-token/horizon16/10步、39-token/horizon32/3步。

|轮次/输入|max abs|RMSE|
|---|---:|---:|
|01_segmented/output.npz|0.00000000|0.00000000|
|01_segmented/dynamic_0.npz|0.00000000|0.00000000|
|01_segmented/dynamic_1.npz|0.00000000|0.00000000|
|01_segmented/dynamic_2.npz|0.00000000|0.00000000|
|02_segmented/output.npz|0.00000000|0.00000000|
|02_segmented/dynamic_0.npz|0.00000000|0.00000000|
|02_segmented/dynamic_1.npz|0.00000000|0.00000000|
|02_segmented/dynamic_2.npz|0.00000000|0.00000000|

## 复现与记录

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
RUN_TAG=$(date +%Y%m%d_%H%M%S)
.venv/bin/python experiments_naxkv_20261104/bench.py \
  --case segmented --iterations 20 --output records/$RUN_TAG-naxkv
```

--case baseline为当前speed；对应耗时见表。source包含MIT LICENSE.mlx、生成内核、生成器、上游源文件哈希。build.py要求已安装MLX0.32.4的对应头文件。全部记录位于records/naxkv_20261104。probe与chain是局部探针，不能替代端到端：依赖链中的K/V固定，编译可能外提拼接。

本阶段是筛选，默认代码暂未修改；有正向结果也需进一步配对与完整样本验证。

## 同进程配对复核

同一模型与权重，交替 AB/BA 顺序，100 对，双方各 4 次预热。每次都重新执行三图、文本、视觉前缀与 10 步动作推理，不复用跨请求特征。基线启用当前 speed KV 工作区，候选切换至 NAX 双段内核。

基线平均 **241.344 ms**，候选 **241.733 ms**；平均节省 **-0.389 ms**，配对 bootstrap 95% 区间 **[-0.458, -0.318] ms**。该区间只反映本次配对样本，不涵盖跨运行系统波动。

100 对最终动作及归一化动作均逐值一致，max abs=0。动态输入的额外测试见上表。这是相对当前量化 speed 的一致性，不代表相对 PyTorch 的误差为零，也不是闭环任务成功率验证。

### 处理结论

候选保留为独立实验，不改生产默认路径。首轮端到端收益仅 0.107 ms，而独立同步的动作模块从 75.282 ms 增至 76.055 ms；100 对配对复核进一步确认候选慢 0.389 ms（约 0.16%），节省区间整体小于零，因此不合入默认路径。不能凭微基准宣传整体提速。

原因分析：双段寻址省去了工作区尾部更新，但当前工作区已避免重复完整拼接；NAX 计算本身与官方实现相同，跨段 fragment 新增的索引和分支可能抵消节省。此为代码结构推断，尚无 GPU 时间线证明各项成本。后续更值得研究动作 Linear 与门控/残差 epilogue 的融合；本轮未实现该项，不能计入已获得收益。
