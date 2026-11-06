import subprocess,sys,json,tarfile,hashlib
from pathlib import Path
import numpy as np
root=Path('records/steelkv_20261104/e2e');root.mkdir(exist_ok=False)
for i,name in enumerate(['baseline','segmented','segmented','baseline']):
 label=f'{i:02d}_{name}'
 with (root/(label+'.log')).open('w') as f:r=subprocess.run([sys.executable,'development/legacy_experiments/experiments_steelkv_20261104/bench.py','--case',name,'--iterations','20','--output',str(root/label)],stdout=f,stderr=subprocess.STDOUT)
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
record=dict(summary=summary,difference=diff,runs=runs,hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('development/legacy_experiments/experiments_steelkv_20261104').glob('*.py')})
(root.parent/'summary.json').write_text(json.dumps(record,indent=2))

chain=json.loads((root.parent/'chain.json').read_text());probe=json.loads((root.parent/'probe.json').read_text())
lines=['# Steel 分块矩阵双段 KV attention（2026-11-04）','',
'已基于安装版 MLX 0.32.4 Steel attention 改造双段 K/V 加载器：保留分块 QK/PV 矩阵乘、在线 softmax、共享内存 tile；在加载 tile 时依据 token 位置选择前缀或动作数据，跨段边界可位于 tile 内。没有物化拼接 K/V，没有跨请求复用旧输入。','',
'当前配置 BQ16/BK32/BD128/BV128、WM2/WN2，128线程。支持动态query/prefix，适用于模型batch1/head_dim128。默认模型权重与代码未修改。该实现是对一种 Steel 模板配置的适配，不等于复用了 MLX 针对所有形状的最佳调度，尚未适配其 NAX 或其他 SDPA 分支。','',
'移植期间修正了 Metal helper 的线程组内存传参和转置 K tile 的5120元素容量；早期编译错误与失败探针日志保留，均不计入通过结果。','',
'## 探针','',
'5组query=1/8/16/32/64、prefix=320/328/368/482，输出有限，局部最大差异='+str(max(r['max_abs'] for r in probe))+'。阈值只用于局部数值排错，不代表机器人精度验收。','',
'8步真实依赖链，每点25次计时：'+str(chain),'',
'链内K/V固定，编译器可能外提重复拼接，不用于推断完整每层动态K/V成本；此前通用点积版0.487ms来自另一轮，不能直接计算严格配对收益。','',
'## 完整模型测试','',
'M5 Pro，15CPU/16GPU，24GiB统一内存，macOS27.2，MLX0.32.4。baseline是当前speed（含KV工作区）；候选关闭工作区而使用双段kernel。ABBA四个独立进程，各首次前向+3次预热+20次稳态+3次同步模块剖析，每配置共40次。','',
'|配置|端到端平均 ms|P95 ms|两轮均值 ms|动作模块 ms（独立同步）|','|---|---:|---:|---|---:|']
for name,r in summary.items():lines.append(f"|{name}|{r['mean_ms']:.3f}|{r['p95_ms']:.3f}|{r['round_means'][0]:.3f} / {r['round_means'][1]:.3f}|{r['action_ms']:.3f}|")
lines+=['','端到端包含在线三图/状态处理、动态文本编码、VAE、视觉前缀、16×14动作10步去噪与CPU输出，不含加载、文件解码与保存。加载、首次与逐次耗时保留在results.json；模块同步不等于无同步端到端。','',
'## 动作误差','',
'对本轮speed的误差，不是对BF16/PyTorch真值。另测193-token文本以及horizon8/32，均为有限输出；不声称逐值一致或闭环成功率不变。','',
'|轮次/输入|max abs|RMSE|','|---|---:|---:|']
for name,files in diff.items():
 for file,d in files.items():lines.append(f"|{name}/{file}|{d['max_abs']:.8f}|{d['rmse']:.8f}|")
lines+=['','## 决策与复现','',
'保留独立实验，未替换默认speed。是否有收益按本轮完整模型与模块结果判断；仅改造加载器不保证超过 MLX 自身针对短 query 的内核。下一步需要确认官方实际调度，并优化加载时的条件寻址和跨query复用，不能把改善标量版本等同于超过默认。','',
'```bash','cd /Users/qklj/work/cx/ImageWAM_MLX','RUN_TAG=$(date +%Y%m%d_%H%M%S)','.venv/bin/python development/legacy_experiments/experiments_steelkv_20261104/bench.py \\', '  --case segmented --iterations 20 --output records/$RUN_TAG-steelkv','```','',
'--case baseline为当前speed；对应端到端耗时见表。记录在records/steelkv_20261104，源码在development/legacy_experiments/experiments_steelkv_20261104。build.py按已安装版本生成segmented_steel.h；upstream.json保留源文件哈希，LICENSE.mlx保留MIT许可及Apple版权。源码来源：MLX 0.32.4安装包 mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h，官方仓库 https://github.com/ml-explore/mlx 。本轮未解析GPU kernel时间线。']
Path('README_STEEL_KV_20261104.md').write_text('\n'.join(lines)+'\n')
with tarfile.open(root.parent/'review.tar.gz','w:gz') as t:
 for p in root.parent.rglob('*'):
  if p.is_file() and p.suffix in ('.json','.npz','.log'):t.add(p,arcname=str(p))
 for p in Path('development/legacy_experiments/experiments_steelkv_20261104').glob('*'):
  if p.is_file():t.add(p,arcname=str(root.parent/'source'/p.name))
print(json.dumps({'summary':summary,'difference':diff},indent=2),flush=True)
