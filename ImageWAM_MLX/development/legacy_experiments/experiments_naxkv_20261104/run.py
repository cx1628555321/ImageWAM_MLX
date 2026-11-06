import subprocess,sys,json,tarfile,hashlib
from pathlib import Path
import numpy as np
root=Path('records/naxkv_20261104/e2e');root.mkdir(exist_ok=False)
for i,name in enumerate(['baseline','segmented','segmented','baseline']):
 label=f'{i:02d}_{name}'
 with (root/(label+'.log')).open('w') as f:r=subprocess.run([sys.executable,'development/legacy_experiments/experiments_naxkv_20261104/bench.py','--case',name,'--iterations','20','--output',str(root/label)],stdout=f,stderr=subprocess.STDOUT)
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
record=dict(summary=summary,difference=diff,runs=runs,hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('development/legacy_experiments/experiments_naxkv_20261104').glob('*.py')})
(root.parent/'summary.json').write_text(json.dumps(record,indent=2))


lines=['# NAX 双段 KV attention 实验（2026-11-04）','',
'基于 MLX 0.32.4 的 steel_attention_nax.h 实现双段 KV。保持 NAX 硬件矩阵乘与在线 softmax；完整落在前缀或动作段的 fragment 使用原 load，只有跨段/尾部 fragment 做逐元素带界限寻址。默认 BQ64/BK32/WM4/WN1。无前缀与动作的完整拼接张量，query<=8则明确回退官方SDPA+拼接以保持原vector路径精度。','',
'官方调度源码在 https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/scaled_dot_product_attention.cpp 。根据源码，本机NAX可用且BF16/head_dim128/query>8满足NAX分支条件；这里是源码依据，未捕获kernel时间线确认。上一轮普通Steel分块未匹配该路径。','',
'模型以当前speed为基础，动作W1、视觉/Qwen BF16、fast VAE不变。候选关闭原工作区，改用双段attention。M5 Pro 15CPU/16GPU、24GiB统一内存、macOS27.2、MLX0.32.4。','',
'## 筛选性能','',
'ABBA四个独立进程，各首次前向+3次预热+20次稳态+3次同步模块分析，每配置40次。真实三图、完整动态文本、状态、16×14动作10步。端到端含CPU输出，不含加载、文件解码、保存。','',
'|配置|端到端平均 ms|P95 ms|两轮均值 ms|动作模块 ms（独立同步）|','|---|---:|---:|---|---:|']
for name,r in summary.items():lines.append(f"|{name}|{r['mean_ms']:.3f}|{r['p95_ms']:.3f}|{r['round_means'][0]:.3f} / {r['round_means'][1]:.3f}|{r['action_ms']:.3f}|")
lines += ['','## 数值与动态输入','',
'逐项动作误差相对本轮speed，不是BF16/PyTorch真值或任务成功率；包含39-token/horizon8/3步、193-token/horizon16/10步、39-token/horizon32/3步。','',
'|轮次/输入|max abs|RMSE|','|---|---:|---:|']
for name,files in diff.items():
 for file,d in files.items():lines.append(f"|{name}/{file}|{d['max_abs']:.8f}|{d['rmse']:.8f}|")
lines += ['','## 复现与记录','',
'```bash','cd /Users/qklj/work/cx/ImageWAM_MLX','RUN_TAG=$(date +%Y%m%d_%H%M%S)','.venv/bin/python development/legacy_experiments/experiments_naxkv_20261104/bench.py \\', '  --case segmented --iterations 20 --output records/$RUN_TAG-naxkv','```','',
'--case baseline为当前speed；对应耗时见表。source包含MIT LICENSE.mlx、生成内核、生成器、上游源文件哈希。build.py要求已安装MLX0.32.4的对应头文件。全部记录位于records/naxkv_20261104。probe与chain是局部探针，不能替代端到端：依赖链中的K/V固定，编译可能外提拼接。','',
'本阶段是筛选，默认代码暂未修改；有正向结果也需进一步配对与完整样本验证。']
Path('README_NAX_KV_20261104.md').write_text('\n'.join(lines)+'\n')
with tarfile.open(root.parent/'review.tar.gz','w:gz') as t:
 for p in root.parent.rglob('*'):
  if p.is_file() and p.suffix in ('.json','.npz','.log'):t.add(p,arcname=str(p))
 for p in Path('development/legacy_experiments/experiments_naxkv_20261104').glob('*'):
  if p.is_file():t.add(p,arcname=str(root.parent/'source'/p.name))
print(json.dumps({'summary':summary,'difference':diff},indent=2),flush=True)
