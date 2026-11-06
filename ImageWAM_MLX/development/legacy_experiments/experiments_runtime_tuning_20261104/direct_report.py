import json,hashlib,tarfile
from pathlib import Path
import numpy as np
p=Path.cwd();r=p/'records/direct_int8_confirm_20261104'
old_manifest=json.loads((p/'records/staged_fold_20261104/manifest.json').read_text())
unchanged={}
for f in ['imagewam_mlx/runtime.py','imagewam_mlx/prepacked_visual.py','scripts/infer.py','scripts/benchmark.py']:
 unchanged[f]=hashlib.sha256((p/f).read_bytes()).hexdigest()==old_manifest[f]
assert all(unchanged.values())
bs=[json.loads((r/f'final_cli/{name}/benchmark.json').read_text()) for name in ['0_baseline','1_tiled','2_tiled','3_baseline']]
old=[bs[0],bs[3]];new=bs[1:3]
def steady(g):return np.array([t['wall_ms'] for b in g for t in b['iterations'] if t['phase']=='steady'])
o,n=steady(old),steady(new)
datasets={k:json.loads((p/f'records/{d}/results.json').read_text()) for k,d in [('direct','direct_int8_20261104'),('activation','activation_int8_20261104'),('confirm','direct_int8_confirm_20261104')]}
for d in ['direct_int8_20261104','activation_int8_20261104','direct_int8_confirm_20261104']:
 v=json.loads((p/f'records/{d}/validation.json').read_text());assert v['all_exact'] and len(v['exact_cases'])==51
dyn=json.loads((r/'dynamic/results.json').read_text());assert all(x['exact'] for x in dyn)
for bpath in ['0_baseline','1_tiled','2_tiled','3_baseline']:
 assert np.array_equal(np.load(r/f'final_cli/{bpath}/actions_steady_000.npy'),np.load(r/'baseline.npz')['action'])
lines=['# INT8 权重读取与激活量化专项：2026-11-04','',
f'本轮测试了四个候选，未发现可固化的单模型端到端收益。当前默认仍为上一轮预展开 INT8 + threadgroup staging。分块直接读取虽在两轮多模型配对中更快，但独立单模型 ABBA 测试反而从 {o.mean():.3f} 增至 {n.mean():.3f} ms（慢 {n.mean()-o.mean():.3f} ms，{100*(n.mean()/o.mean()-1):.2f}%），因此不替换默认路径。','',
'## 方法与结果','',
'设备 M5 Pro，15 CPU / 16 GPU，24 GiB 统一内存，MLX 0.32.4，macOS 27.2。基线是已启用预展开 INT8 的当前 speed；buffer256，三图、在线可变文本、10 步动作推理、horizon16，固定噪声。端到端是已解码 RGB/state/text 到 CPU 动作，不含模型加载、文件读取和输出写盘，无跨请求特征缓存。','',
'| 候选 | 配对基线 ms | 候选 ms | 节省 ms，bootstrap 95% 区间 | 决策 |','|---|---:|---:|---|---|']
for group,key,title,decision in [('direct','direct','原行布局直接读取 INT8','区间跨零，不采用'),('direct','tiled','64×64 分块布局直接读取 INT8','进入独立复测'),('activation','q128','128 线程激活量化','变慢，不采用'),('activation','q256','256 线程激活量化','变慢，不采用'),('confirm','tiled','分块布局独立配对复测','进入单模型 ABBA')]:
 d=datasets[group][key];b=datasets[group]['baseline'];lo,hi=d['ci95'];lines.append(f'| {title} | {b["mean_ms"]:.3f} | {d["mean_ms"]:.3f} | {d["saving_ms"]:.3f} [{lo:.3f}, {hi:.3f}] | {decision} |')
lines+=['','前两组分别 100 组随机顺序配对，分块独立复测 120 组。配对实验共享未改动的模块，但同时保留候选权重。置信区间仅反映该实验条件内的采样，不保证单模型部署收益。','',
'单模型交错测试使用同一个 scripts/benchmark.py 测量逻辑；候选通过实验入口替换 visual 与编译 prefill，不改默认代码。old/new/new/old 各新进程、各 4 次预热和 40 次稳态：','',
'| 进程 | 平均 ms | P95 ms |','|---|---:|---:|']
for label,b in zip(['0 baseline','1 tiled','2 tiled','3 baseline'],bs):lines.append(f'| {label} | {b["steady"]["mean_ms"]:.3f} | {b["steady"]["p95_ms"]:.3f} |')
lines += ['',f'合并均值：基线 {o.mean():.3f} ms，候选 {n.mean():.3f} ms。基线/候选 P95：{np.percentile(o,95):.3f}/{np.percentile(n,95):.3f} ms。本轮绝对值不能直接与上一轮 203.354 ms 相减归因。','',
'单独同步模块诊断各 6 次，会改变调度，仅用于定位，不可将其相加替代端到端：','',
'| 模块 | 基线 ms | 分块直接读取 ms |','|---|---:|---:|']
for key in ['text_encoder','image_encoder','visual_prefill','action_denoise']:
 def mod(g):return np.mean([t['modules'][key] for b in g for t in b['iterations'] if t['phase']=='profile'])
 lines.append(f'| {key} | {mod(old):.3f} | {mod(new):.3f} |')
lines += ['','## 为什么没有固化','',
'权重直接读取候选移除了每个 K 块的 threadgroup 复制和屏障；分块候选额外在加载时将 INT8 权重排列为 N/64 × K/64 × 64 × 64。INT32 累加及输出缩放不变。减少显式拷贝并不必然更快：threadgroup staging 提供块内复用，直接 device 读取的访存代价可能抵消屏障收益。这是机制分析，尚无 GPU counter/timeline 证明具体瓶颈。','',
'配对与单模型结果相反，提示工作集、权重缓存及调度状态对该优化敏感。不能将多模型配对收益直接当作单模型收益；本轮以单模型部署路径为最终依据。','',
'激活量化候选改为按元素分配线程、两遍读取，减少每线程保存 64 个值的需求，同时保留当前 kernel ABI 的指数占位数组。降低寄存器压力可能被第二遍读取、归约和布局写入成本抵消；完整路径实测略慢，未采用。第一版占位数组过小导致 Metal constant/device 地址空间编译错误，修复后所有正式计时通过；失败日志保留。','',
'## 精度与可变输入','',
'四个候选均在 51 个案例中与当前 speed 的 action/normalized 逐值一致；分块候选独立复测再次通过 51 例。分块候选还通过 '+', '.join(f'{x["tokens"]} tokens / horizon{x["horizon"]}' for x in dyn)+' 的逐值一致性回归。四个单模型进程标准样本输出也逐值一致。该结果只说明本轮候选未增加已测数值差异，原 speed 的量化误差仍存在。','',
'runtime.py、prepacked_visual.py、infer.py、benchmark.py 的 SHA256 与上一轮正式版本完全相同，本轮未改变默认路径或覆盖模型权重/配置。新增 tiled_visual.py 是实验候选模块，默认未引用；精度路径无改动。','',
'## 复现','',
'工作目录 /Users/qklj/work/cx/ImageWAM_MLX。当前默认约 '+f'{o.mean():.3f}'+' ms：','',
'```bash\n.venv/bin/python scripts/benchmark.py --profile speed --reference examples/robotwin_ep0_frame0 --output records/direct_followup_baseline --warmup 4 --iterations 40 --profile-iterations 3\n```','',
'仅供实验的分块候选约 '+f'{n.mean():.3f}'+' ms，未默认启用：','',
'```bash\n.venv/bin/python development/legacy_experiments/experiments_runtime_tuning_20261104/direct_bench_entry.py tiled --profile speed --reference examples/robotwin_ep0_frame0 --output records/direct_followup_candidate --warmup 4 --iterations 40 --profile-iterations 3\n```','',
'输出目录须使用新名称。原始配对/回归结果分别在 records/direct_int8_20261104、records/activation_int8_20261104、records/direct_int8_confirm_20261104；最后一个目录包含 ABBA、动态输入、输入/输出与源码快照。实验脚本为 development/legacy_experiments/experiments_runtime_tuning_20261104/direct_*.py 和 activation_test.py。','',
'下一步应保留 threadgroup 权重复用，针对激活量化与相邻归一化/拼接的融合做局部诊断，或测试 INT8 tile 参数。在投入新 kernel 前，应先以单模型 profile 量化目标开销；继续采用配对筛选与独立进程复测两道验证。']
name='README_INT8_DATAFLOW_20261104.md';(p/name).write_text('\n'.join(lines)+'\n')
intro=f'最新 INT8 数据搬运实验：[{name}]({name})，四项候选未确认单模型提速，默认预展开 INT8 路径保持。\n\n'
read=p/'README.md';s=read.read_text();read.write_text(intro+s if not s.startswith(intro) else s)
(r/'default_unchanged.json').write_text(json.dumps(unchanged,indent=2))
archive=p/'records/int8_dataflow_20261104_review.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 paths=[p/name,read,p/'imagewam_mlx/tiled_visual.py',p/'development/legacy_experiments/experiments_runtime_tuning_20261104/activation_test.py']
 paths+=list((p/'development/legacy_experiments/experiments_runtime_tuning_20261104').glob('direct_*.py'))
 paths+=[p/'records'/d for d in ['direct_int8_20261104','activation_int8_20261104','direct_int8_confirm_20261104']]
 paths+=list((p/'records').glob('*int8*20261104*.log'))
 for f in paths:tar.add(f,arcname=str(f.relative_to(p)),filter=lambda t:None if t.name.endswith('trace.npz') or '__pycache__' in t.name else t)
print('SHA256',hashlib.sha256(archive.read_bytes()).hexdigest(),'bytes',archive.stat().st_size)
print('\n'.join(lines[:34]))
