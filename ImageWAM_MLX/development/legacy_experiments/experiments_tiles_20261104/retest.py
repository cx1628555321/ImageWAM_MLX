import subprocess,sys,json,time,hashlib,tarfile
from pathlib import Path
import numpy as np
root=Path('records/tiles_retest_20261104');root.mkdir(exist_ok=False)
order=['current_speed','speed_native_visual','tile32k512','tile64n64','tile64n64','tile32k512','speed_native_visual','current_speed']
for i,name in enumerate(order):
 label=f'{i:02d}_{name}'
 (root/(label+'_processes.txt')).write_text(subprocess.check_output(['ps','-axo','pid,pcpu,comm'],text=True))
 with (root/(label+'.log')).open('w') as f:
  r=subprocess.run([sys.executable,'development/legacy_experiments/experiments_tiles_20261104/e2e.py','--case',name,'--iterations','30','--output',str(root/label)],stdout=f,stderr=subprocess.STDOUT)
 print(label,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
runs=[json.loads((root/f'{i:02d}_{name}'/'results.json').read_text()) for i,name in enumerate(order)]
summary={}
for name in dict.fromkeys(order):
 rs=[r for r in runs if r['case']==name];ts=[t for r in rs for t in r['iterations']]
 summary[name]=dict(mean_ms=float(np.mean(ts)),p95_ms=float(np.percentile(ts,95)),round_means=[r['mean_ms'] for r in rs],visual_ms=float(np.mean([m['visual_prefill'] for r in rs for m in r['modules']])))
checks={}
ref=root/'01_speed_native_visual'
for i,name in enumerate(order):
 if not name.startswith('tile'):continue
 checks[f'{i:02d}_{name}']={}
 for filename in ('output.npz','dynamic_0.npz','dynamic_1.npz','dynamic_2.npz'):
  x=np.load(root/f'{i:02d}_{name}'/filename);y=np.load(ref/filename)
  checks[f'{i:02d}_{name}'][filename]=all(np.array_equal(x[k],y[k]) for k in ('action','normalized'))
result=dict(summary=summary,checks=checks,order=order,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('development/legacy_experiments/experiments_tiles_20261104').glob('*.py')})
(root/'summary.json').write_text(json.dumps(result,indent=2))
lines=['# INT8 分块交替顺序复测','',
'环境：M5 Pro，15 CPU / 16 GPU，24 GiB 统一内存，macOS 27.2，MLX 0.32.4。后台 mediaanalysisd、视频编解码及虚拟机仍活动，没有暂停其他服务。各轮之前保存进程负载快照，不能称为空闲环境测试。','',
'本轮不再仅比较旧的历史数字，加入原 INT8 融合视觉路径作为同轮控制。顺序：speed → 原 INT8 → tile32k512 → tile64n64 → tile64n64 → tile32k512 → 原 INT8 → speed。每配置两个独立进程，各首次前向、3 次预热、30 次稳态、3 次模块剖析；共 240 次正式稳态。','',
'|配置|60 次平均 ms|两轮均值 ms|P95 ms|视觉前缀 ms（独立同步）|','|---|---:|---|---:|---:|']
for name,r in summary.items():lines.append(f"|{name}|{r['mean_ms']:.3f}|{r['round_means'][0]:.3f} / {r['round_means'][1]:.3f}|{r['p95_ms']:.3f}|{r['visual_ms']:.3f}|")
lines += ['','三个相机输入、在线分词、状态处理、文本/图像编码、视觉前缀、10 步动作去噪、CPU 输出均包含在端到端内；不包含模型加载、图像文件解码和文件保存。仍为 16×14 动作，不固定或截断文本长度。','',
'与本轮原 INT8 输出比较：'+str(checks),'',
'动态测试包含 39-token/horizon8/3步、193-token/horizon16/10步、39-token/horizon32/3步。逐值一致仅表示分块改动没有再改变对应 INT8 路径的输出，不代表其与 accuracy 相同或机器人任务精度已验证。','',
'启动：','```bash','cd /Users/qklj/work/cx/ImageWAM_MLX','RUN_TAG=$(date +%Y%m%d_%H%M%S)','.venv/bin/python development/legacy_experiments/experiments_tiles_20261104/e2e.py --case tile32k512 \\', '  --iterations 30 --output records/$RUN_TAG-tile-retest','```','',
'--case 也可取 current_speed、speed_native_visual、tile64n64；对应耗时见上表。完整逐次耗时、加载、首次、模块和输出文件位于 `records/tiles_retest_20261104/<round_case>/`。复测脚本 `development/legacy_experiments/experiments_tiles_20261104/retest.py` 拒绝覆盖本次固定记录路径，重新使用需更换目录。','',
'默认 accuracy / speed 保持不变。背景负载和独立进程之间的漂移仍限制小幅收益判断；不会将单算子最快或单轮最快直接升级成默认。']
Path('README_INT8_RETEST_20261104.md').write_text('\n'.join(lines)+'\n')
with tarfile.open(root/'review.tar.gz','w:gz') as t:
 for p in root.rglob('*'):
  if p.is_file() and p.suffix in ('.json','.npz','.txt'):t.add(p,arcname=str(p))
 for p in Path('development/legacy_experiments/experiments_tiles_20261104').glob('*.py'):t.add(p,arcname=str(root/'source'/p.name))
print(json.dumps(result,indent=2),flush=True)
