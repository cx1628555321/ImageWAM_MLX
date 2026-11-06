from pathlib import Path
import json,hashlib,shutil,tarfile
import numpy as np
root=Path('records/tiles_20261104');paths=sorted((root/'e2e').glob('*/results.json'))
assert len(paths)==5
runs={p.parent.name:json.loads(p.read_text()) for p in paths};assert all(r['status']=='completed' for r in runs.values())
reference=Path('records/lowbit_20261104/confirm/03_speed_native_visual')
checks={}
for name,r in runs.items():
 if not r['case'].startswith('tile'):continue
 checks[name]={}
 for file in ['output.npz','dynamic_0.npz','dynamic_1.npz','dynamic_2.npz']:
  a=np.load(root/'e2e'/name/file);b=np.load(reference/file)
  checks[name][file]={k:dict(exact=bool(np.array_equal(a[k],b[k])),max_abs=float(np.max(np.abs(a[k]-b[k])))) for k in ('action','normalized')}
summary={'runs':runs,'comparison_to_previous_int8':checks,'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('development/legacy_experiments/experiments_tiles_20261104').glob('*.py')}}
(root/'summary.json').write_text(json.dumps(summary,indent=2))
base=[r['mean_ms'] for r in runs.values() if r['case']=='current_speed'];mean=sum(base)/len(base)
lines=['# INT8 GEMM 分块专项复测（2026-11-04）','',
'本轮继续定位上一轮真实 INT8 内核的形状适配问题，测试 BM/BN/BK 分块，保持输入文本动态、三相机图像、16×14 动作和 10 步不变。未更改生产默认代码、权重或配置。','',
'环境：Apple M5 Pro，15 CPU / 16 GPU，24 GiB 统一内存；macOS 27.2，MLX 0.32.4，Python 3.12.13。实现沿用上一轮 MIT Cider 内核（许可证在 `development/legacy_experiments/experiments_lowbit_20261104/kernels/LICENSE.cider`），新 wrapper 位于 `development/legacy_experiments/experiments_tiles_20261104/tune.py`。','',
'## 已执行','',
'- 在 5 组真实动作/视觉输入矩阵上测试 8 组分块，共 40 次候选测试；每组预热 4 次、测量 15 次。全部输出与原 INT8 内核逐值一致。','- 分块改变线程组大小、M/N padding 和 K 循环划分，没有将输入长度固定成样例长度。原 INT8 权重一次性量化；每次动态激活在线量化。','- 选择 32/128/128、32/128/512、64/64/128 三个候选，分别接入原有 speed 的视觉模块；动作仍为原 W1A16。','- 按 current_speed → 三个候选 → current_speed 顺序，独立进程串行测试；每组首次前向、3 次预热、50 次稳态、另 3 次模块剖析。','',
'## 完整模型结果','',
'本轮观察到 mediaanalysisd 约 109% CPU，另有视频编解码和虚拟机后台进程，已保存 `system_processes.txt`。这证明环境存在并发负载，但不能据此定量归因 GPU 干扰；不与上一轮静态数字直接计算收益。本轮候选需稳定超过同轮前后基线才考虑合入。','',
'|配置|稳态平均 ms|P95 ms|加载 s|首次 ms|视觉前缀 ms（独立剖析）|','|---|---:|---:|---:|---:|---:|']
for name,r in runs.items():
 visual=sum(m['visual_prefill'] for m in r['modules'])/len(r['modules'])
 lines.append(f"|{name}|{r['mean_ms']:.3f}|{r['p95_ms']:.3f}|{r['load_s']:.3f}|{r['first_ms']:.3f}|{visual:.3f}|")
lines += ['',f'前后现有 speed 基线均值 {mean:.3f} ms。模块剖析有额外同步，不能与无同步端到端值直接相加。首次/加载并非严格冷启动测试。微基准包含 dispatch/eval，波动明显，不能直接据此推算完整模型收益。','',
'## 精度与动态形状','',
'与上一轮融合 INT8 视觉路径对比本次输出和 normalized 数组；正式 39-token/horizon16/10步之外，还验证 39-token/horizon8/3步、193-token/horizon16/10步、39-token/horizon32/3步。逐项结论保存在 `summary.json`。这些验证不代表机器人任务精度或整个 RoboTwin 数据集验证。','',
'|候选|4 组输入的 action/normalized 是否全部逐值一致|','|---|---|']
for name,c in checks.items():lines.append(f"|{name}|{all(v['exact'] for file in c.values() for v in file.values())}|")
lines += ['','## 复现','',
'```bash','cd /Users/qklj/work/cx/ImageWAM_MLX','RUN_TAG=$(date +%Y%m%d_%H%M%S)','.venv/bin/python development/legacy_experiments/experiments_tiles_20261104/e2e.py \\', '  --case tile32k128 --iterations 50 --output records/$RUN_TAG-tile32k128','```','',
'可替换 `--case` 为 `tile32k512`、`tile64n64` 或 `current_speed`。各启动方式对应上表端到端耗时；输出目录不能已存在。','',
'记录：`records/tiles_20261104/tuning.json`（40 组单算子）、`e2e/`（完整计时与输出）、`summary.json`（精度对比与源码哈希）、`source/`（源码快照）。上一轮 GPU trace 仍保留；本轮未解析 kernel timeline，也未新增大型 GPU capture。']
best=min((r for r in runs.values() if r['case'].startswith('tile')),key=lambda r:r['mean_ms'])
lines += ['',f"本轮最快候选 {best['case']} 为 {best['mean_ms']:.3f} ms；前后基线分别为 {base[0]:.3f}、{base[1]:.3f} ms。暂不修改生产默认；背景负载和单轮顺序测试不足以证明可重复收益。"]
Path('README_INT8_TILES_20261104.md').write_text('\n'.join(lines)+'\n')
shutil.copytree('development/legacy_experiments/experiments_tiles_20261104',root/'source',ignore=shutil.ignore_patterns('__pycache__'))
with tarfile.open(root/'review.tar.gz','w:gz') as t:
 for p in root.rglob('*'):
  if p.is_file() and p.suffix in ('.json','.npz','.py','.txt'):t.add(p,arcname=str(p))
print(json.dumps({'means':{k:r['mean_ms'] for k,r in runs.items()},'checks':checks},indent=2))
