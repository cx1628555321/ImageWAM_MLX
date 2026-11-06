import json,hashlib,tarfile
from pathlib import Path
import numpy as np
p=Path.cwd();r=p/'records/int8_padding_followup_20261104'
verify=json.loads((p/'records/speed_verify_followup_20261104/benchmark.json').read_text())
v=verify['steady'];res=json.loads((r/'results.json').read_text())
validation=json.loads((r/'validation.json').read_text());assert validation['all_exact'] and len(validation['exact_cases'])==51
bs=[json.loads((r/f'final_cli/{s}/benchmark.json').read_text()) for s in ['0_80','1_64','2_64','3_80']]
def vals(indices):return np.array([it['wall_ms'] for i in indices for it in bs[i]['iterations'] if it['phase']=='steady'])
o,n=vals([0,3]),vals([1,2])
old=json.loads((p/'records/staged_fold_20261104/manifest.json').read_text());checks={f:hashlib.sha256((p/f).read_bytes()).hexdigest()==old[f] for f in ['imagewam_mlx/runtime.py','imagewam_mlx/prepacked_visual.py','scripts/infer.py','scripts/benchmark.py']};assert all(checks.values())
for s in ['0_80','1_64','2_64','3_80']:
 assert np.array_equal(np.load(r/f'final_cli/{s}/actions_steady_000.npy'),np.load(r/'baseline.npz')['action'])
lines=['# 正式 speed 的 200 ms 核验与 INT8 共享内存布局实验','',
'结论：历史约 200 ms 确实来自正式 --profile speed，而非不能部署的候选。当前正式实现未改变，但本轮负载下不能确认稳态平均仍为 200 ms。不能用最快一次代替平均，也不能把不同时间段的差异归因于代码优化。','',
'## 正式入口重新测量','',
'M5 Pro（15 CPU / 16 GPU）、24 GiB 统一内存，macOS 27.2，MLX 0.32.4。正式预展开 INT8 speed，buffer256，三张已解码图片 + state + 在线文本，10 步、horizon16；计时至 CPU 动作输出，无跨请求特征缓存，不包括文件读取、模型加载或输出写盘。10 次预热后统计 100 次稳态。','',
'| 指标 | 本轮正式 speed |','|---|---:|',
f'| 平均 | {v["mean_ms"]:.3f} ms |',f'| P50 | {v["median_ms"]:.3f} ms |',f'| P95 | {v["p95_ms"]:.3f} ms |',f'| 最小 / 最大 | {v["min_ms"]:.3f} / {v["max_ms"]:.3f} ms |',f'| 加载 | {verify["load_s"]:.3f} s |',f'| 首次前向（含首次编译等） | {verify["first"]["mean_ms"]:.3f} ms |','',
'同期观察到 mediaanalysisd 约占一个 CPU 核，并有 VideoToolbox 编解码服务活动。过程快照保存在本轮记录内；没有暂停其他服务，也没有 GPU 时间线证明其与每次延迟尖峰的因果关系。历史正式均值 200.071 ms 和 203.354 ms 的记录保留有效，但不是当前负载下的保证值。','',
'## 继续优化：保留块内复用，改变权重共享内存行跨度','',
'原内核 INT8 权重 tile 的 K=64，threadgroup 行跨度为 80 字节。候选只将跨度改为 64、96、128，测试 bank conflict/对齐/占用率的性能影响；数学运算、权重量化、激活精度和输入处理均不变。没有移除 threadgroup 权重复用。','',
'每个候选 100 组随机顺序配对：','',
'| 路径 | 平均 ms | 相对基线节省 ms | bootstrap 95% 区间 ms |','|---|---:|---:|---|']
for key in ['baseline','pad64','pad96','pad128']:
 d=res[key];lines.append(f'| {key} | {d["mean_ms"]:.3f} | {d["saving_ms"]:.3f} | [{d["ci95"][0]:.3f}, {d["ci95"][1]:.3f}] |')
lines+=['','三个候选收益区间都跨零。进一步对配对均值最好的 64 字节候选进行单模型独立进程 ABBA，顺序 80/64/64/80，各 10 次预热、60 次稳态：','',
'| 进程 | 平均 ms | P95 ms |','|---|---:|---:|']
for label,b in zip(['0_80','1_64','2_64','3_80'],bs):lines.append(f'| {label} | {b["steady"]["mean_ms"]:.3f} | {b["steady"]["p95_ms"]:.3f} |')
lines += ['',f'合并平均：当前 80 字节路径 {o.mean():.3f} ms，64 字节候选 {n.mean():.3f} ms，候选差值 {n.mean()-o.mean():+.3f} ms。未确认稳定收益，不更改正式默认。独立进程使用相同 benchmark 计时逻辑，实验入口仅替换候选 kernel；这是实验路径，不是正式默认发生变化。','',
'三个候选都通过 51 个代表样本的 action/normalized 逐值一致性验证；独立进程标准样本也逐值一致。此处仅说明本轮未增加已测数值差异，原 speed 相对 accuracy/PyTorch 的量化误差依然存在。本轮未重新执行完整 accuracy，因为其实现没有改变；SHA256 已确认 runtime.py、prepacked_visual.py、infer.py、benchmark.py 与上轮正式版本相同。','',
'## 复现与记录','',
'工作目录 /Users/qklj/work/cx/ImageWAM_MLX。正式路径重测（本轮平均 '+f'{v["mean_ms"]:.3f}'+' ms）：','',
'```bash\n.venv/bin/python scripts/benchmark.py --profile speed --reference examples/robotwin_ep0_frame0 --output records/speed_verify_next --warmup 10 --iterations 100 --profile-iterations 5\n```','',
'64 字节实验候选，仅供复查：','',
'```bash\n.venv/bin/python development/legacy_experiments/experiments_runtime_tuning_20261104/padding_bench_entry.py 64 --profile speed --reference examples/robotwin_ep0_frame0 --output records/padding64_next --warmup 10 --iterations 60 --profile-iterations 3\n```','',
'输出目录应使用新名称。records/speed_verify_followup_20261104 保存正式 100 次结果、输入、输出、模块诊断与源码快照；records/int8_padding_followup_20261104 保存三个候选、51 例输出、进程快照和 ABBA。实验脚本为 development/legacy_experiments/experiments_runtime_tuning_20261104/padding_*.py。大型 trace.npz 保留远端，审阅归档不含该文件。','',
'下一步优先量化相邻归一化/激活处理与 INT8 量化的单算子成本，再选择融合边界；在当前明显抖动的环境下，小于数毫秒的端到端改善需要多轮单模型确认，不能仅按最低耗时推广。']
name='README_SPEED_VERIFY_PADDING_20261104.md';(p/name).write_text('\n'.join(lines)+'\n')
intro=f'最新正式速度核验：[{name}]({name})。历史约 200 ms 属于正式路径；本轮平均 {v["mean_ms"]:.3f} ms、最小 {v["min_ms"]:.3f} ms，受测试时段影响。三个布局候选未固化，默认实现不变。\n\n'
read=p/'README.md';s=read.read_text();read.write_text(intro+s if not s.startswith(intro) else s)
(r/'default_unchanged.json').write_text(json.dumps(checks,indent=2))
archive=p/'records/speed_verify_padding_20261104_review.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for f in [p/name,read,r,p/'records/speed_verify_followup_20261104',p/'records/int8_padding_followup_20261104.log',*list((p/'development/legacy_experiments/experiments_runtime_tuning_20261104').glob('padding_*.py'))]:
  tar.add(f,arcname=str(f.relative_to(p)),filter=lambda t:None if t.name.endswith('trace.npz') or '__pycache__' in t.name else t)
print('SHA256',hashlib.sha256(archive.read_bytes()).hexdigest(),'bytes',archive.stat().st_size)
print('FINAL',dict(verified_mean_ms=v['mean_ms'],old_mean_ms=float(o.mean()),candidate_mean_ms=float(n.mean())))
