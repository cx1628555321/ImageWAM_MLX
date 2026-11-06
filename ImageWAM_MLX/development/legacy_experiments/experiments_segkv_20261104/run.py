import subprocess,sys,json,tarfile,hashlib
from pathlib import Path
import numpy as np
root=Path('records/segkv_20261104/e2e');root.mkdir(exist_ok=False)
for i,name in enumerate(['baseline','segmented','segmented','baseline']):
 label=f'{i:02d}_{name}'
 with (root/(label+'.log')).open('w') as f:r=subprocess.run([sys.executable,'development/legacy_experiments/experiments_segkv_20261104/bench.py','--case',name,'--iterations','20','--output',str(root/label)],stdout=f,stderr=subprocess.STDOUT)
 print(label,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
runs={p.parent.name:json.loads(p.read_text()) for p in sorted(root.glob('*/results.json'))};summary={};diff={}
for mode in ('baseline','segmented'):
 rs=[r for r in runs.values() if r['case']==mode];ts=[t for r in rs for t in r['iterations']]
 summary[mode]=dict(mean_ms=float(np.mean(ts)),p95_ms=float(np.percentile(ts,95)),round_means=[r['mean_ms'] for r in rs],action_ms=float(np.mean([m['action_denoise'] for r in rs for m in r['modules']])))
for name in runs:
 if not name.endswith('segmented'):continue
 diff[name]={}
 for file in ('output.npz','dynamic_0.npz','dynamic_1.npz','dynamic_2.npz'):
  a=np.load(root/name/file);b=np.load(root/'00_baseline'/file);d=a['action']-b['action'];diff[name][file]=dict(max_abs=float(np.max(np.abs(d))),rmse=float(np.sqrt(np.mean(d*d))))
record=dict(summary=summary,difference=diff,runs=runs,hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('development/legacy_experiments/experiments_segkv_20261104').glob('*.py')})
(root.parent/'summary.json').write_text(json.dumps(record,indent=2))
lines=['# 双段 KV 自定义 attention 实验（2026-11-04）','',
'实现单个 Metal kernel，直接读取前缀 K/V 与动作 K/V，进行统一 FP32 QK、softmax、PV，输出 BF16。无拼接结果张量，无缓存原位修改，不跨请求复用旧输入。模型中仅替换动作 attention，去噪仍10步。','',
'当前实现为通用标量/向量计算和线程组归约，没有 simdgroup matrix/TensorOps 加速，不应等同于官方 SDPA 的优化程度。每个 query/head 独立处理，K/V 读取的跨 query 复用有限。消除拼接不等于整体更快。','',
'## 数值与微基准','',
'支持动态query及prefix，当前模型限定batch1/head_dim128。随机验证query=1/8/16/32/64，prefix=320/328/368/482，输出有限，与官方SDPA的最大绝对差不超过0.001953125。这个门槛仅是局部测试，不是任务精度保证。','',
'真实捕获Q/K/V输入的8步依赖链，25次计时；单步均值官方SDPA约0.146ms、双段内核约0.487ms。链中K/V固定，编译器可能外提重复拼接，因此此实验主要比较attention计算，不能当作完整每层动态K/V搬运开销。正式收益以下面端到端为准。','',
'## 完整模型','',
'M5 Pro（15CPU/16GPU，24GiB统一内存），macOS27.2，MLX0.32.4。baseline是当前默认speed，包括已合入KV工作区；segmented关闭工作区改为双段kernel。顺序baseline→segmented→segmented→baseline，每进程首次前向+3次预热+20次稳态，另3次同步模块分析。每配置40次。','',
'|路径|稳态平均 ms|P95 ms|两轮均值 ms|动作模块 ms（独立同步）|','|---|---:|---:|---|---:|']
for name,r in summary.items():lines.append(f"|{name}|{r['mean_ms']:.3f}|{r['p95_ms']:.3f}|{r['round_means'][0]:.3f} / {r['round_means'][1]:.3f}|{r['action_ms']:.3f}|")
lines += ['','端到端包括在线三图/状态处理、完整动态文本编码、VAE、视觉前缀、16×14动作的10步去噪和CPU输出；不含加载、文件解码和保存。加载/首次/逐次耗时均在results.json保留，模块时间不直接求和。','',
'## 输出差异','',
'以下为对本轮speed基线的最终动作误差，不是对PyTorch/BF16真值或闭环任务成功率。另测39-token/horizon8/3步、193-token/horizon16/10步、39-token/horizon32/3步，全部输出有限。','',
'|候选轮次/输入|max abs|RMSE|','|---|---:|---:|']
for name,files in diff.items():
 for file,d in files.items():lines.append(f"|{name}/{file}|{d['max_abs']:.8f}|{d['rmse']:.8f}|")
lines += ['','## 决策与下一步','',
'本版保留为独立实验，不替换默认speed/accuracy。下一步若继续双段KV，需要以官方SDPA的分块/矩阵计算实现为基础加入两段地址读取，避免通用点积内核效率不足。没有证据表明“只要零拼接就一定提速”。','',
'```bash','cd /Users/qklj/work/cx/ImageWAM_MLX','RUN_TAG=$(date +%Y%m%d_%H%M%S)','.venv/bin/python development/legacy_experiments/experiments_segkv_20261104/bench.py \\', '  --case segmented --iterations 20 --output records/$RUN_TAG-segkv','```','',
'--case baseline 是当前speed工作区对照；启动耗时对应上表。输出目录必须不存在。源码位于development/legacy_experiments/experiments_segkv_20261104；全部记录在records/segkv_20261104，包含probe.json、chain.json、summary.json、e2e输出及源码哈希。本轮没有解析GPU kernel时间线。']
Path('README_SEGMENTED_KV_20261104.md').write_text('\n'.join(lines)+'\n')
with tarfile.open(root.parent/'review.tar.gz','w:gz') as t:
 for p in root.parent.rglob('*'):
  if p.is_file() and p.suffix in ('.json','.npz'):t.add(p,arcname=str(p))
 for p in Path('development/legacy_experiments/experiments_segkv_20261104').glob('*.py'):t.add(p,arcname=str(root.parent/'source'/p.name))
print(json.dumps({'summary':summary,'difference':diff},indent=2),flush=True)
