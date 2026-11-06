from pathlib import Path
import json
root=Path(__file__).resolve().parents[2];d=json.loads((root/'records/device_nax_20261105/summary.json').read_text())
s='''# 无缓存10步：INT8权重直读与K循环展开（2026-11-05）

结论：已将直接读取INT8权重、展开两个K fragment的组合设为`--profile speed`默认路径；保留`--no-device-visual`回退。未启用VAE padding候选或动作四元素归约。accuracy路径和模型文件未修改。

本轮始终使用在线三图编码、在线文本编码、状态输入、H16、10步去噪、CPU动作输出。文本长度可变，无跨请求文本/图像/动作特征缓存。权重仍为既有预展开INT8视觉权重、动态INT8视觉激活、INT32累加与BF16输出，动作W1路径不变；未新增反量化。

环境：M5 Pro，15核CPU/16核GPU，24 GiB统一内存，macOS27.2，Python3.12.13，MLX0.32.4。端到端不含模型加载、输入文件读取和输出落盘。

## 性能结论与波动

|测量|旧基线 ms|新组合 ms|说明|
|---|---:|---:|---|
'''
s+=f"|随机配对80轮|{d['screen2']['baseline']['mean_ms']:.3f}|{d['screen2']['combined']['mean_ms']:.3f}|减少{d['screen2']['combined']['saving_ms']:.3f} ms；配对bootstrap区间{d['screen2']['combined']['ci95']} ms|\n"
s+=f"|独立进程正反顺序，各120次|{d['independent']['baseline']['mean_ms']:.3f}|{d['independent']['combined']['mean_ms']:.3f}|反向单块出现退化，全部保留|\n"
for title,key in [('正式CLI首轮ABBA，各200次','cli'),('正式CLI复核BAAB，各200次','confirm'),('全部8个CLI进程，各400次','all_cli')]:
 s+=f"|{title}|{d[key]['old']['mean_ms']:.3f}|{d[key]['new']['mean_ms']:.3f}|{'首轮候选较慢，未排除' if key=='cli' else '包括全部稳态样本'}|\n"
a=d['all_cli'];s+=f"\n全量CLI均值减少 **{a['saving_ms']:.3f} ms（{a['reduction_percent']:.2f}%）**。P95从{a['old']['p95_ms']:.3f}变为{a['new']['p95_ms']:.3f} ms，尾延迟尚未改善。复核候选两块均值193.697/194.654 ms，合并194.176 ms、P95 195.681 ms；这是较稳定阶段的实测，不是任何负载下的保证。最小单次192.920 ms不作为稳定速度宣传。\n"
s+='''
历史198.66 ms来自另一轮环境，不能直接与本轮194.18 ms相减归因。随机配对的约3.43 ms改善与视觉模块加速相符；复核的7.59 ms差额含运行状态变化，不能全部归因于新内核。bootstrap未校正时序相关性，也不代表独立进程置信区间。

系统快照未报告热/性能警告，已用交换空间约2038 MiB；观察到系统UI、虚拟机/Docker等进程，但没有停止任何其他服务，也没有GPU时间线证据证明某个后台进程导致波动。

## 各正式进程与加载、首次推理

首轮预热10次，复核预热30次；每进程100次稳态与5次单独模块测量。首次推理含首次编译/执行，单独报告，不混入稳态。

|块|方案|均值ms|P95 ms|加载s|首次ms|
|---|---|---:|---:|---:|---:|
'''
for label,b in dict(d['cli_blocks'],**d['confirm_blocks']).items():
 s+=f"|{label}|{b['mode']}|{b['steady']['mean_ms']:.3f}|{b['steady']['p95_ms']:.3f}|{b['load_s']:.3f}|{b['first']['mean_ms']:.3f}|\n"
s+='''
old为显式`--no-device-visual`，new为`--device-visual`。最终默认已经与new相同。默认入口额外5次烟雾检查均值201.019 ms，实际infer单次202.582 ms，均输出一致；这两项只用于入口验证，不混入正式统计。

## 模块耗时

|模块|全量旧ms|全量新ms|复核旧ms|复核新ms|
|---|---:|---:|---:|---:|
'''
for key in ('tokenization','text_encoder','preprocess','image_encoder','visual_prefill','action_denoise','postprocess'):
 s+=f"|{key}|{d['all_cli']['old']['modules'][key]:.3f}|{d['all_cli']['new']['modules'][key]:.3f}|{d['confirm']['old']['modules'][key]:.3f}|{d['confirm']['new']['modules'][key]:.3f}|\n"
s+='''
模块计时额外同步，不能直接相加作为端到端。复核视觉prefill约82.43→78.27 ms（约5.05%），动作去噪约62.66→62.69 ms；本轮实际改进位于视觉prefill。其余未改模块的变化反映运行波动，不属于优化收益。

## 已测试方案

第一轮配对：

|方案|均值ms|相对同轮基线节省ms|
|---|---:|---:|
'''
for k,v in d['screen'].items():s+=f"|{k}|{v['mean_ms']:.3f}|{v['saving_ms']:.3f}|\n"
s+='\n第二轮配对：\n\n|方案|均值ms|相对同轮基线节省ms|\n|---|---:|---:|\n'
for k,v in d['screen2'].items():s+=f"|{k}|{v['mean_ms']:.3f}|{v['saving_ms']:.3f}|\n"
s+='''
- direct：B fragment直接从已展开INT8 device权重读取，移除每个K=64块的共享内存暂存和两次threadgroup barrier。
- direct_tail：仅32行尾块使用直读。
- unroll：保留原共享内存路径，仅将两个K fragment循环显式展开。
- combined：直读与展开组合，采纳到默认。相较单项效果更好，但没有GPU指令时间线，不能断言具体由缓存命中率或寄存器占用变化造成。
- reduce4：动作Split-K归约每线程处理四元素，保持逐元素BF16舍入。无收益，未采纳。
- combo：combined叠加reduce4，无明确额外收益，未采纳。

新实现保持原累加顺序、量化参数、输出类型、动态M和32行尾块。N不是64整倍数时回退原实现。自动启用范围沿用既有M5/MLX版本及speed量化路径检查；关闭tail/gate/concat/prepacked路径时不会错误套用新内核。

## 输出与输入可变性验证

第一轮3个候选、第二轮4个候选，各51组真实输入的动作及归一化动作全部逐值一致。补充39/189-token文本、H1/H16/H64、改变相机尺寸/内容及状态，直读、组合、归约及叠加候选的预处理、latent和最终动作均逐值一致。正式CLI共800组输出均与旧speed逐值一致；默认benchmark与infer检查也通过。

一致性仅相对于当前量化speed，不代表相对PyTorch无损或RoboTwin闭环任务成功率不变。动作、归一化动作、噪声、输入、源码快照均保留用于后续对比。

## 启动命令

在m5pro执行，10步、无缓存、新内核默认启用：

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
.venv/bin/python scripts/benchmark.py \\
  --profile speed --text-cache-size 0 --steps 10 \\
  --reference examples/robotwin_ep0_frame0 \\
  --output "records/device_default_$(date +%Y%m%d_%H%M%S)" \\
  --warmup 30 --iterations 100 --profile-iterations 5
```

对应新版本全量正式均值198.652 ms，复核阶段194.176 ms；以实际运行的benchmark.json为准。追加`--no-device-visual`回退本轮之前的speed。

实际输入推理：

```bash
.venv/bin/python scripts/infer.py \\
  --profile speed --text-cache-size 0 --steps 10 \\
  --inputs examples/robotwin_ep0_frame0/inputs.npz \\
  --prompt-file examples/robotwin_ep0_frame0/prompt.txt \\
  --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy \\
  --output "records/device_infer_$(date +%Y%m%d_%H%M%S)" --warmup 3
```

## 记录与后续

远端项目：`/Users/qklj/work/cx/ImageWAM_MLX`。实现：`imagewam_mlx/device_visual.py`；入口：runtime.py、scripts/benchmark.py、scripts/infer.py。实验：`experiments/device_nax_20261105`，原始执行源码存放`records/device_nax_20261105/experiment_source_before_default`；默认启用后，筛选脚本显式关闭新默认以保持旧基线定义。驱动输出目录拒绝覆盖，重跑请使用新的记录目录。

全部结果在`records/device_nax_20261105`，汇总summary.json，默认验证default_verification.json。审阅压缩包保留源码、输入、动作与计时；大型trace.npz仅保留远端完整记录，不纳入本地压缩包。

后续重点仍是约78 ms视觉prefill与约63 ms动作去噪。应先取得GPU时间线与主机提交证据，继续针对真实形状优化INT8读取/计算重叠和动作W1算子；本轮没有解决尾延迟波动，不将减少步数或跨请求缓存列为本目标下的优化。
'''
(root/'README_DEVICE_NAX_20261105.md').write_text(s)
