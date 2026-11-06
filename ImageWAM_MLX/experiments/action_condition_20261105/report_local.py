from pathlib import Path
import json
root=Path(__file__).resolve().parents[2];d=json.loads((root/'records/action_condition_20261105/summary.json').read_text())
s='''# 每请求时间调制、准备图与融合Attention实验（2026-11-05）

结论：本轮没有确认新的端到端提速，生产推理源码、默认配置、模型与权重不变。候选仅保留在experiments/action_condition_20261105，不增加默认开关。最佳路径仍为上一轮已验证的INT8权重直读＋K fragment展开，四SIMD布局；10步、无跨请求特征缓存。

## 范围与计时

M5 Pro，15核CPU、16核GPU，24 GiB统一内存，macOS27.2，Python3.12.13，MLX0.32.4。输入为RoboTwin三张已解码图像、状态、39-token文本、H16与固定初始噪声，在线编码、10步动作去噪到CPU反归一化动作。加载、输入读取和输出落盘不计入稳态。

每次请求仍重新计算调制；没有缓存上次请求的文本、图像、KV或动作。编译函数缓存的是计算图，不是输入特征或调制结果。文本不截断，动作长度与请求步骤数仍由输入决定。

## 候选

- unrolled：每请求集中计算各步时间embedding、double/single调制和head shift/scale，保留每步原来的batch=1形状，然后供各去噪步消费。
- batched：将10个时间步按batch=10一起计算调制；输出发生变化，淘汰，不纳入无损路径。
- prepare：将噪声类型转换、调度量、RoPE位置编码与KV工作区padding合并为准备图，每请求执行。
- prepare_unrolled：准备图与保持batch=1的集中调制组合。
- attention_action / attention_visual / attention_both：分别在动作、视觉前缀或两者使用实际安装MLX支持的force_fused=True，不修改Q/K/V与scale。

当前MLX接口文档说明force_fused只是绕过启发式强制使用融合内核，某些形状可能更慢；服务器原始docstring存为attention_api.txt。本轮不能仅凭“融合”名称推断提速，也未证明默认所有形状必然使用同一个内核。

## 配对筛选

各阶段随机顺序80轮，每配置预热5次，都是完整请求。下表按各自同轮基线比较，不跨阶段相减。

|阶段／候选|基线ms|候选ms|节省ms|节省95%区间ms|标准样本输出|
|---|---:|---:|---:|---|---|
'''
for phase in ('screen','prepare','attention'):
 for k,v in d[phase].items():
  if k=='baseline':continue
  s+=f"|{phase}/{k}|{d[phase]['baseline']['mean_ms']:.3f}|{v['mean_ms']:.3f}|{v['saving_ms']:.3f}|[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]|{'逐值一致' if v['exact'] else '不一致'}|\n"
s+='''
bootstrap区间未校正时间相关性，不代表跨进程复现性。两轮unrolled配对仅节省约0.219/0.332 ms。prepare及组合无明确收益。强制融合attention也未显示收益。

batched标准样本动作最大绝对差0.053438、RMSE 0.017840。batch形状改变且输出不一致，尚未追踪到单个底层内核的数值差异来源；不能把这一版本的约0.52 ms改善当作无损优化。

## 独立进程复核

对unrolled进行基线→候选→候选→基线四进程复核，每进程预热8次、稳态60次，每配置120次：

|配置|平均ms|P95 ms|正向ms|反向ms|
|---|---:|---:|---:|---:|
'''
for k,v in d['independent'].items():s+=f"|{k}|{v['mean_ms']:.3f}|{v['p95_ms']:.3f}|{v['blocks'][0]:.3f}|{v['blocks'][1]:.3f}|\n"
s+='''
候选两个方向均未胜过基线，合并慢0.231 ms（约0.115%）。因此不继续推广为生产默认，也不拿配对阶段的0.2–0.3 ms作为已获得的部署收益。本轮没有进行候选正式CLI推广或新的全量CLI发布基准，下面是独立worker结果，不能与此前正式CLI混淆。

|进程|加载ms|首次前向ms|稳态均值ms|
|---|---:|---:|---:|
'''
for k,v in d['independent'].items():
 for i,label in enumerate(('forward','reverse')):s+=f"|{label}_{k}|{v['load_ms'][i]:.3f}|{v['first_ms'][i]:.3f}|{v['blocks'][i]:.3f}|\n"
s+='''
## 模块诊断

每进程另测3次同步模块profile。同步会影响重叠，不能将模块和当作端到端。

|模块|基线ms|集中调制ms|
|---|---:|---:|
'''
for k,v in d['independent']['baseline']['modules'].items():s+=f"|{k}|{v:.3f}|{d['independent']['unrolled']['modules'][k]:.3f}|\n"
s+='''
动作去噪约62.688→62.568 ms，仅局部0.121 ms变化；视觉prefill仍约78 ms。分离调制缩短了单步图，但也增加独立图输出与提交边界，可能抵消少量节省；这是代码层面的解释候选，并无GPU时间线证明各项开销。当前不能断言带宽、CPU或某个后台服务是唯一瓶颈。

## 精度和可变输入

每个阶段各51组真实输入：

|阶段|候选|一致组数|
|---|---|---:|
'''
for phase in ('screen','prepare','attention'):
 for k,n in d[phase+'_validation']['exact_counts'].items():s+=f"|{phase}|{k}|{n}/51|\n"
s+='''
补充39/189-token文本、H1/H16/H64，以及相机尺寸/图像内容/状态扰动：unrolled、prepare、prepare_unrolled、attention_both的动作、预处理和latent全部逐值一致。速度数据仅对应39-token/H16，其他输入用于正确性验证。

另外对标准样本保留并逐值比较unrolled、prepare、prepare_unrolled的action_step_0…9，十步全部一致；输入图、状态、文本、latent、token ids及抽查的KV也一致。没有仅用最后动作掩盖中间差异。

一致性相对当前量化speed，不代表相对原始PyTorch无损或已验证机器人闭环成功率。批量调制0/51一致，明确排除。实际输入、噪声、候选最终动作、归一化输出和中间trace均已保留。

## 当前启动命令

生产路径保持不变：

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
.venv/bin/python scripts/benchmark.py \\
  --profile speed --text-cache-size 0 --steps 10 \\
  --reference examples/robotwin_ep0_frame0 \\
  --output "records/current_speed_$(date +%Y%m%d_%H%M%S)" \\
  --warmup 30 --iterations 100 --profile-iterations 5
```

此前正式CLI均值194.468 ms、P95 196.212 ms；本轮独立worker基线200.814 ms，是另一种测试流程/运行时段，不替换历史正式数据，不当作代码退化或提速。实际速度以本次输出的benchmark.json为准。

实验脚本：screen.py、prepare_test.py、driver.py、attention_test.py、dynamic.py。它们在项目根目录运行，并拒绝覆盖已有记录；如要重跑，先为脚本指定/改用新的记录目录。它们没有被接入默认推理入口。

## 记录与后续

远端项目`/Users/qklj/work/cx/ImageWAM_MLX`；实验代码`experiments/action_condition_20261105`；结果`records/action_condition_20261105`；汇总summary.json，逐步比较dynamic/trace_comparison.json。中间*_trace.npz保留远端，审阅压缩包保留最终动作、输入、计时、源码和比较结果。

当前这些请求级小改动未产生稳定收益。后续应优先取得GPU内核级时间线，再针对约78 ms视觉prefill及约63 ms动作矩阵乘/attention做定点优化；本轮没有拿未验证的kernel瓶颈结论或错误输出作为提速成绩。
'''
(root/'README_ACTION_CONDITION_20261105.md').write_text(s)
p=root/'README.md';p.write_text(p.read_text()+'\n每请求动作准备与融合Attention实验：[README_ACTION_CONDITION_20261105.md](README_ACTION_CONDITION_20261105.md)。完成独立复核与逐步精度检查，无新稳定收益，默认不变。\n')
