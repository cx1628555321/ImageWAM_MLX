import json,hashlib,tarfile
from pathlib import Path
import numpy as np
p=Path.cwd();r=p/'records/staged_fold_20261104'
bs=[json.loads((r/f'final_cli/{name}/benchmark.json').read_text()) for name in ['0_old','1_new','2_new','3_old']]
old=[bs[0],bs[3]];new=bs[1:3]
def steady(g):return np.array([t['wall_ms'] for b in g for t in b['iterations'] if t['phase']=='steady'])
o,n=steady(old),steady(new)
assert json.loads((r/'validation.json').read_text())['all_exact']
assert json.loads((r/'final_validation/completed.json').read_text())['native_exact']
assert np.array_equal(np.load(r/'cli_infer/action.npy'),np.load(r/'final_cli/1_new/actions_steady_000.npy'))
dyn=json.loads((r/'dynamic/results.json').read_text());assert all(x['exact'] for x in dyn)
lines=['# 冻结权重预展开 INT8：2026-11-04','',f'已固化到 speed 默认路径。本轮正式 ABBA 独立进程测试：旧 speed {o.mean():.3f} ms → 新 speed {n.mean():.3f} ms，减少 {o.mean()-n.mean():.3f} ms（{100*(1-n.mean()/o.mean()):.2f}%）。不应把上一轮 230 ms 与本轮直接相减：后台负载与时段不同，本轮只对同期对照归因。','',
'瓶颈：上一轮虽采用 INT8 矩阵乘，但每个 M 方向 tile 都重复将冻结 W4 权重解包，并把 scale/bias 折叠为 INT8。现在加载时用相同整数规则完成一次转换，前向直接读取 INT8 权重；保留在线激活量化、INT32 累加和 BF16 输出。这是以常驻内存换取前向时间，不是进一步降低比特数，也不会把整个模型变成全 INT8。','',
'38 个 single_blocks.linear1/linear2 使用预展开 INT8，来源仍是先前 W4 group64；Qwen/VAE/其余视觉层和动作 W1 路径保持原逻辑。磁盘权重未重写。内核继续使用同一 MLX PR #4722 mode-2 算术，来源/许可见 staged_int8_upstream.json 和 LICENSE.mlx。','',
'## 测试条件与结果','',
'M5 Pro（15 CPU / 16 GPU）、24 GiB 统一内存、macOS 27.2、MLX 0.32.4、Python 3.12.13。命令缓冲阈值 256。三张已解码 RGB + state + 在线可变文本 → CPU 动作；10 步、默认 horizon16。计时不含文件读取、加载和结果写盘，无跨请求输入/特征缓存。','',
'100 组随机顺序配对：基线 206.118 ms，只预计算整数系数 206.807 ms（慢 0.690 ms，未采用），预展开 INT8 199.730 ms（节省 6.387 ms，配对 bootstrap 95% 区间 6.316–6.463 ms）。','',
'正式入口另做 old/new/new/old 四进程测试，各 4 次预热、40 次稳态；四块均值依次为 '+', '.join(f'{b["steady"]["mean_ms"]:.3f}' for b in bs)+' ms。期间有整机时间漂移，前半段和后半段比较均支持提速；不声称已消除温度/后台负载影响。','',
'| 指标 | 本轮旧 speed | 本轮新 speed |','|---|---:|---:|',
f'| 稳态平均 ms，各 80 次 | {o.mean():.3f} | {n.mean():.3f} |',
f'| 合并 P50 / P95 ms | {np.median(o):.3f} / {np.percentile(o,95):.3f} | {np.median(n):.3f} / {np.percentile(n,95):.3f} |',
f'| 加载平均 s，各 2 次 | {np.mean([b["load_s"] for b in old]):.3f} | {np.mean([b["load_s"] for b in new]):.3f} |',
f'| 首次平均 ms，受编译缓存影响 | {np.mean([b["first"]["mean_ms"] for b in old]):.3f} | {np.mean([b["first"]["mean_ms"] for b in new]):.3f} |',
f'| MLX active GiB | {old[0]["memory"]["active_bytes"]/2**30:.3f} | {new[0]["memory"]["active_bytes"]/2**30:.3f} |',
f'| MLX peak GiB | {old[0]["memory"]["peak_bytes"]/2**30:.3f} | {new[0]["memory"]["peak_bytes"]/2**30:.3f} |','',
'MLX active 增加约 1.22 GiB；该指标不等于系统进程 RSS。peak 包含模型加载阶段，本次 peak 相同不代表常驻内存没有增加。','',
'模块单独同步诊断每路径 6 次，会改变调度；不能直接相加代替稳态端到端。','',
'| 模块 | 旧 ms | 新 ms |','|---|---:|---:|']
for key in ['text_encoder','image_encoder','visual_prefill','action_denoise']:
 def mod(g):return np.mean([it['modules'][key] for b in g for it in b['iterations'] if it['phase']=='profile'])
 lines.append(f'| {key} | {mod(old):.3f} | {mod(new):.3f} |')
lines+=['','## 精度与输入可变性','',
'两个实验候选均对 51 个案例的 action/normalized 与上一版 speed 逐值一致。正式默认版本再次对全部 51 个案例与上一轮保存的结果逐值一致；accuracy 标准样本与历史精度参考逐值一致，关闭 staged 的原路径回归通过。infer CLI 与 benchmark 输出逐值一致。',
'补充生产路径与 --no-prepacked-visual 路径比较：'+', '.join(f'{x["tokens"]} tokens / horizon {x["horizon"]}' for x in dyn)+'，action/normalized 均逐值一致。未固定或截断文本长度。该优化没有新增已测数值误差，但 speed 本身已有先前 W1/W4 量化误差，不能称其与 accuracy/PyTorch 等精度，未验证机器人任务成功率。','',
'## 使用与回退','',f'项目目录 /Users/qklj/work/cx/ImageWAM_MLX。新 speed 同期平均 {n.mean():.3f} ms：','',
'```bash\n.venv/bin/python scripts/benchmark.py --profile speed --reference examples/robotwin_ep0_frame0 --output records/prepacked_retest_new --warmup 4 --iterations 40 --profile-iterations 3\n```','',
f'回退本轮预展开（保留上一轮 staged INT8 + buffer256），同期平均 {o.mean():.3f} ms：','',
'```bash\n.venv/bin/python scripts/benchmark.py --profile speed --no-prepacked-visual --reference examples/robotwin_ep0_frame0 --output records/prepacked_retest_old --warmup 4 --iterations 40 --profile-iterations 3\n```','',
'```bash\n.venv/bin/python scripts/infer.py --profile speed --inputs examples/robotwin_ep0_frame0/inputs.npz --prompt-file examples/robotwin_ep0_frame0/prompt.txt --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy --output records/prepacked_infer --warmup 1\n```','',
'输出目录需新建。Python API 用 prepacked_visual=False 回退；--no-staged-visual 关闭整个 staged 路径；accuracy 仍是未指定 profile 时的默认路径。自动启用范围仍受原 M5/MLX0.32.4 策略限制。','',
'## 记录与后续','',
'records/staged_fold_20261104 保存前版源码、100 组原始耗时、51 样本输出、正式 ABBA 输入/输出/源码快照、精度与动态长度结果。development/legacy_experiments/experiments_runtime_tuning_20261104/fold_*.py 保存复现脚本。最初一次内核尺寸变量编译错误已修复，原始失败日志保留为 records/staged_fold_20261104.log。',
'下一步可围绕预展开 INT8 权重的分块布局、重复从 device 到 threadgroup 的复制和 activation quantize 的额外 kernel 做专项实验；需继续以端到端验证，因为单算子加速不必然带来整体收益。无需重新量化或下载权重。']
name='README_PREPACKED_INT8_20261104.md';(p/name).write_text('\n'.join(lines)+'\n')
intro=f'最新正式 speed：[冻结权重预展开 INT8]({name})，同期 {o.mean():.3f} → {n.mean():.3f} ms，已测输出逐值一致，active 内存增加约 1.22 GiB。以下保留历史报告。\n\n'
read=p/'README.md';s=read.read_text();read.write_text(intro+s if not s.startswith(intro) else s)
manifest={}
for base in [p/'imagewam_mlx',p/'scripts',r/'final_cli']:
 for f in base.rglob('*'):
  if f.is_file() and '__pycache__' not in str(f):manifest[str(f.relative_to(p))]=hashlib.sha256(f.read_bytes()).hexdigest()
(r/'manifest.json').write_text(json.dumps(manifest,indent=2))
archive=p/'records/staged_fold_20261104_review.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for f in [r,p/name,read,p/'imagewam_mlx',p/'scripts',*list((p/'development/legacy_experiments/experiments_runtime_tuning_20261104').glob('fold_*.py'))]:
  tar.add(f,arcname=str(f.relative_to(p)),filter=lambda t:None if t.name.endswith('trace.npz') or '__pycache__' in t.name else t)
print('SHA256',hashlib.sha256(archive.read_bytes()).hexdigest(), 'bytes',archive.stat().st_size)
print('\n'.join(lines[:35]))
