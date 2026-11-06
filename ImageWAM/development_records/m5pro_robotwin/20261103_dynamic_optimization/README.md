# M5 Pro 可变文本长度优化（2026-11-03）

本轮优化保持原始文本、三张 RGB 图、状态、归一化、BF16 权重、seed=42、16×14 动作输出和 10 步去噪不变。没有文本缓存、量化或减少去噪步数。

## 实现

正式代码在 scripts/m5pro/：
- optimizations.py：可恢复的优化上下文。dynamic 按当前输入实际长度 tokenize，padding=True、truncation=False；不同长度每次重新生成 mask 和位置，不缓存上一输入的状态。支持长度以 Qwen 位置容量为上限，超出明确报错，不截断。
- action_prefill.py：动作专用视觉 prefill，保留最后一层 K/V，省去后续未被动作路径使用的最终视觉更新。不是图像生成入口。
- scan_robotwin_text.py：全量文本长度扫描。
- test_dynamic_optimization.py：消融、全部实际长度、多真实观测及交错性能比较。
- benchmark_robotwin.py：新增 --optimization baseline|tail|trim|dynamic。默认仍为 baseline，历史复测命令不改变语义。

优化不硬编码 39、79、512 的实际计算长度，不固定输入图像张量尺寸；正式 RoboTwin 预处理仍按原入口拼成 H×W=288×256，符合本次测试约定。批量文本内部按本批最长长度 padding；完整动作入口仍为原有 batch=1。

## 覆盖与输入核验

完整扫描 921,032 条任务文本，使用与推理一致的提示前缀及 chat template。实际范围31～79 token，48种长度。
对每种长度的一个真实文本进行完整原版/优化版推理比较，另测 episode0/frame20、episode1000/frame0、episode10000/frame0，共51组。
代表样本的有效 token ID 与原版完全一致。实际三图、状态、指令和原预处理结果也已逐元素核对。
这是全量文本长度覆盖，不是对全部921,032条文本进行推理精度验证。

## 性能

|测试|原版|dynamic|
|---|---:|---:|
|交错测量20次平均|722.527 ms|392.177 ms|
|交错测量p95|729.515 ms|409.672 ms|
|优化版单独运行20次平均|—|351.494 ms|
|优化版单独运行p95|—|352.850 ms|

主要结论使用交错结果：耗时减少45.72%，加速1.84倍。
正式优化入口新进程模型加载9.354秒，首次推理2.477秒；未控制系统页缓存及Metal编译缓存，不解释为受控冷启动提升。

5次模块测量中位数：文本39.497 ms、图像22.079 ms、视觉prefill138.439 ms、动作去噪174.560 ms。
模块阶段包含同步开销，与无钩子端到端不同；父子项不可相加。
有一次文本模块耗时356.430 ms，使文本模块均值为102.467 ms、该轮端到端701.334 ms。未删除此异常样本，全部原始记录保留。

## 数值差异

51组测试最大反归一化动作绝对差0.0125736217，最大归一化差0.015625，样本平均MAE约0.00045250。
保留有效输入的数学语义，不等于改变计算形状后BF16输出逐位一致。本轮不能认定这些误差对闭环任务成功率没有影响；未运行RoboTwin闭环。
tail单独消融在基准样本上逐位一致，但收益较小。dynamic与trim不支持用历史“逐位相等”检查当作通过条件；应使用本轮数值误差比较脚本。
未改变所有用户输入；没有承诺所有动作逐位一致，也没有用户指定的业务误差容限，因此未将dynamic设为默认。

## 复跑

```bash
cd /Users/qklj/work/cx/ImageWAM
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv-mps/bin/python -u scripts/m5pro/benchmark_robotwin.py \
  --optimization dynamic --audit \
  --output-dir evaluate_results/m5pro_robotwin/NEW_DYNAMIC_RUN

# 全文本长度代表、多观测和交错比较；每次自动创建唯一目录
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv-mps/bin/python -u scripts/m5pro/test_dynamic_optimization.py --full
```

--audit检查真实MPS执行和输入依赖；dynamic不要同时传历史--reference-dir（该选项坚持逐位相等）。数值比较使用 --full 测试脚本。
每轮在线编码当前文本与图像，不使用固定指令或固定观测缓存。

## 记录目录

本目录：
- text_lengths.json：全量扫描直方图、各长度代表文本、原任务文件哈希。
- optimization_summary.json：性能、数值误差、输入核验、模块异常样本汇总。
- comparison_results/：51组比较、交错测量、每轮归一化/反归一化完整动作、case_errors.csv、源码快照与输出哈希。
- profile_results/：正式入口完整基准、真实三图、状态、指令、初始噪声、每轮输出、设备审计。
- ablation_results/：baseline/tail/trim/dynamic初步消融。
- code/：链接到正式 scripts/m5pro。

所有文件位于 /Users/qklj/work/cx/ImageWAM 内，历史基线不覆盖。
