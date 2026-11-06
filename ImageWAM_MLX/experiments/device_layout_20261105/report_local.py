from pathlib import Path
import json
r=Path(__file__).resolve().parents[2];d=json.loads((r/'records/device_layout_20261105/summary.json').read_text())
s='''# 无缓存10步：SIMD布局与正确性审计（2026-11-05）

结论：本轮未更换speed默认。保留上一轮INT8权重直读＋K fragment展开；新增双SIMD候选为显式实验参数`--visual-group-rows 32/64`。默认/0仍是原四SIMD布局。未修改权重、量化策略、步数或输入缓存策略。

## 正式入口结果

M5 Pro，15 CPU/16 GPU，24 GiB统一内存，macOS27.2，Python3.12.13，MLX0.32.4。RoboTwin三张已解码图像、状态、39-token文本、H16，在线编码、10步去噪，计时到CPU动作返回。无跨请求特征缓存，不含加载、输入读取及落盘。

三个配置以0→32→64→64→32→0顺序分别启动独立进程，每进程预热30次、稳态100次、模块计时5次。全部样本保留，每配置200次：

|配置|均值ms|P95 ms|决定|
|---|---:|---:|---|
'''
for mode,label in [('0','当前默认，四SIMD'),('32','双SIMD，32行'),('64','双SIMD，64行')]:
 x=d['cli'][mode];s+=f"|{label}|{x['mean_ms']:.3f}|{x['p95_ms']:.3f}|{'保持默认' if mode=='0' else '仅显式实验'}|\n"
s+='''
64行候选两块均值191.093/198.796 ms，合并194.944 ms，较基线慢0.476 ms（0.245%）；不能只取191.093 ms作为新的默认成绩。32行候选更慢。上一轮198.65 ms与本轮默认194.47 ms不是同轮对照，不能把差额归因于新优化。

## 逐进程、加载与首次

|块|行配置|均值ms|P95 ms|加载s|首次ms|
|---|---|---:|---:|---:|---:|
'''
for name,b in d['cli_blocks'].items():s+=f"|{name}|{b['mode']}|{b['steady']['mean_ms']:.3f}|{b['steady']['p95_ms']:.3f}|{b['load_s']:.3f}|{b['first']['mean_ms']:.3f}|\n"
s+='''
首次含首次前向/编译执行，与稳态分开。默认入口追加3次烟雾验证均值200.068 ms，仅确认默认参数与输出，不混入正式统计；同样反映速度存在运行波动。

## 模块诊断

|模块|默认ms|32行ms|64行ms|
|---|---:|---:|---:|
'''
for k in d['cli']['0']['modules']:
 s+='|'+k+'|'+ '|'.join(f"{d['cli'][m]['modules'][k]:.3f}" for m in ('0','32','64'))+'|\n'
s+='''
64行候选视觉prefill从79.381降至75.915 ms，局部减少约3.466 ms（4.37%），但没有形成正式端到端均值/P95收益。模块计时为单独同步测量，不能相加作为端到端。未改动模块与预处理的波动提示执行状态差异；没有GPU时间线，不能断言具体是CPU调度、频率、带宽或某个后台任务。

## 配对和独立进程筛选

随机顺序80轮，每种候选预热5次，均为完整10步无缓存请求：

|候选|同轮基线ms|候选ms|节省ms|输出检查|
|---|---:|---:|---:|---|
'''
for group in ('screen','rounding','fixed','group2'):
 for k,v in d[group].items():
  if k=='baseline':continue
  s+=f"|{k}|{d[group]['baseline']['mean_ms']:.3f}|{v['mean_ms']:.3f}|{v['saving_ms']:.3f}|{'逐值一致' if v['exact'] else '不一致，禁止作为有效加速'}|\n"
s+='''
双SIMD完整配对的节省95% bootstrap区间：32行[1.745,1.872] ms，64行[1.876,2.037] ms；未校正时序相关性，不代表跨进程稳定性。

独立worker正反顺序各60次：

|配置|合并均值ms|正向ms|反向ms|
|---|---:|---:|---:|
'''
for k,x in d['independent'].items():s+=f"|{k}|{x['mean_ms']:.3f}|{x['blocks'][0]:.3f}|{x['blocks'][1]:.3f}|\n"
s+='''
这些初筛出现收益，最终CLI未稳定复现，因此没有默认推广，也没有删掉冲突结果。

## 方案与发现

1. `m4n1`：四SIMD按M4/N1分工，数学结果一致但更慢。
2. `m1n4`：四SIMD按M1/N4分工，初测188.178 ms但输出错误，成绩作废。
3. `k2`：再展开外层两个K块，一致但更慢。
4. `left/right`：显式限制浮点缩放乘法顺序，用于判断舍入误差；均未修复差异。完成80轮计时后，确认整数错误，停止后续过时的51例舍入回归，日志保留；不宣称该回归完成。
5. `fixed`：修正本地双M片段矩阵乘描述，恢复一致，但均值200.584 ms，慢于同轮193.781 ms。
6. `g2_32/g2_64`：线程组128→64线程，WM=1、WN=2，保留正确的N=32 native片段配对、INT8整数计算和原累加顺序。32行版本按原64行激活布局寻址；64行版本增加每个SIMD内M方向权重复用。N不为64整倍数时回退原安全路径。未增加完整权重反量化。

### 为什么188 ms无效

逐层导出标准观测的38个真实线性层INT32累加器，m1n4与基线在38层全部存在差异，排除了“仅末端BF16舍入”的假设。检查本地metal/epilogue.h发现，双M片段重载处理32×16输出，却沿用16×32描述。实验头文件改为32×16后，38层整数累加与浮点输出全部一致，完整51组动作也一致。

这是本地实验触发的辅助分支问题，不将其归因于未经核查的上游版本。当前默认、上一轮直读版本及本轮安全双SIMD候选使用N方向成对片段，不走该错误分支。生产共用头文件未改动；独立修正与原头文件快照存放实验目录，避免未验证全局改动。此例说明最终动作误差较小，也不能证明内部矩阵计算正确。

## 精度范围

- 初筛51组：m4n1、k2各51/51一致；错误m1n4为0/51一致。
- 修正分支：51/51一致，38/38层整数及浮点输出一致。
- 两个安全双SIMD候选：各51/51动作及归一化动作逐值一致。
- 补充动态输入：39/189-token文本、H1/H16/H64、相机尺寸/内容和状态扰动，预处理、latent、动作均一致。
- 正式CLI共600组输出全部与当前默认逐值一致；默认参数恢复/保持检查也通过。

以上一致性均相对当前量化speed，不代表相对PyTorch无损或闭环任务成功率结论。速度数据对应标准39-token/H16样本，其他长度只验证正确性，未证明所有输入形状都提速。

## 启动与回退

推荐继续使用默认：

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
.venv/bin/python scripts/benchmark.py \\
  --profile speed --text-cache-size 0 --steps 10 \\
  --reference examples/robotwin_ep0_frame0 \\
  --output "records/default_$(date +%Y%m%d_%H%M%S)" \\
  --warmup 30 --iterations 100 --profile-iterations 5
```

本轮默认正式均值194.468 ms、P95 196.212 ms。所有结果以实际新运行记录为准。

追加`--visual-group-rows 64`测试安全候选（本轮合并194.944 ms）；追加`--visual-group-rows 32`测试32行（200.493 ms）。使用`--visual-group-rows 0`或不传恢复当前默认。`--no-device-visual`是更早一代共享内存路径的回退，不是本轮基线，不能混淆。

实际输入脚本scripts/infer.py也接受上述参数；完整prompt不截断、不固定文本长度，RoboTwin图像处理逻辑未改。

## 文件与后续

远端开发目录：`/Users/qklj/work/cx/ImageWAM_MLX`。安全候选在imagewam_mlx/group_visual.py；runtime.py及两套正式入口提供参数和执行模式记录。实验与诊断在experiments/device_layout_20261105，原始记录在records/device_layout_20261105，汇总summary.json。所有输入、初始噪声、动作、归一化输出、源码快照和失败日志保留；审阅包不含大型trace.npz，完整trace保留远端。

下一步应先控制或解释跨进程运行状态差异，再评估是否推广双SIMD。视觉prefill约79 ms与动作去噪约63 ms仍是重点；本轮未通过增加反量化、减少步数、缓存输入或容忍错误矩阵运算换取数字。
'''
(r/'README_DEVICE_LAYOUT_20261105.md').write_text(s)
p=r/'README.md';p.write_text(p.read_text()+'\nSIMD布局后续实验与正确性审计：[README_DEVICE_LAYOUT_20261105.md](README_DEVICE_LAYOUT_20261105.md)。正式默认保持不变；新增32/64行候选仅显式启用。\n')
p=r/'README_FASTEST_START.md';p.write_text(p.read_text()+'\n## SIMD布局后续复核\n\n当前默认本轮正式194.468 ms（200次，P95 196.212 ms）；与历史数字差异不算代码收益。双SIMD候选局部prefill更快，但正式端到端未优于基线，未默认启用。[完整报告](README_DEVICE_LAYOUT_20261105.md)。\n')
