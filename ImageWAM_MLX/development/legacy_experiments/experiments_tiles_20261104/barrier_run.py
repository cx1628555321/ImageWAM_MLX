import subprocess,sys,json,hashlib,tarfile
from pathlib import Path
import numpy as np
root=Path('records/barrier_20261104');root.mkdir(exist_ok=False)
r=subprocess.run([sys.executable,'development/legacy_experiments/experiments_tiles_20261104/probe_no_barrier.py']);assert r.returncode==0
for i,mode in enumerate(['original','removed','removed','original']):
 label=f'{i:02d}_{mode}';script='e2e.py' if mode=='original' else 'e2e_no_barrier.py'
 (root/(label+'_load.txt')).write_text(subprocess.check_output(['ps','-axo','pid,pcpu,comm'],text=True))
 with (root/(label+'.log')).open('w') as f:r=subprocess.run([sys.executable,'development/legacy_experiments/experiments_tiles_20261104/'+script,'--case','tile32k512','--iterations','30','--output',str(root/label)],stdout=f,stderr=subprocess.STDOUT)
 print(label,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
runs={p.parent.name:json.loads(p.read_text()) for p in root.glob('*/results.json')};groups={}
for mode in ('original','removed'):
 rs=[r for name,r in runs.items() if name.endswith(mode)];ts=[t for r in rs for t in r['iterations']]
 groups[mode]=dict(mean_ms=float(np.mean(ts)),round_means=[r['mean_ms'] for r in rs],visual_ms=float(np.mean([m['visual_prefill'] for r in rs for m in r['modules']])))
checks={}
for name in runs:
 checks[name]={}
 for filename in ('output.npz','dynamic_0.npz','dynamic_1.npz','dynamic_2.npz'):
  a=np.load(root/name/filename);b=np.load(root/'00_original'/filename)
  checks[name][filename]=all(np.array_equal(a[k],b[k]) for k in ('action','normalized'))
result=dict(groups=groups,checks=checks,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('development/legacy_experiments/experiments_tiles_20261104').glob('*.py')});(root/'summary.json').write_text(json.dumps(result,indent=2))
lines=['# INT8 GEMM K 循环屏障移除实验','',
'本轮只移除 `threadgroup_barrier(mem_flags::mem_none)`。K 循环只读取输入矩阵，各 simdgroup 的 accumulator 位于自己的寄存器；没有跨组共享可写 tile。计算、累加次序、量化参数和 32/128/512 分块不变。独立实验 `tune_no_barrier.py`，默认推理代码未修改。','',
'环境：M5 Pro 16 GPU / 24 GiB 统一内存，MLX 0.32.4。后台 mediaanalysisd 等仍在运行，未暂停其他服务；各轮进程快照保留。不能称为无干扰纯 GPU 测量。','',
'顺序：原屏障→移除→移除→原屏障。每轮独立进程，首次前向、3 次预热、30 次计时，另 3 次模块测量。各方案 60 次稳态。三图、动态文本、16×14 动作和 10 步不变，包含 CPU 输出，不含加载、文件解码和保存。','',
'|配置|端到端均值 ms|两轮均值 ms|视觉前缀 ms（独立同步）|','|---|---:|---|---:|']
for name,r in groups.items():lines.append(f"|{name}|{r['mean_ms']:.3f}|{r['round_means'][0]:.3f} / {r['round_means'][1]:.3f}|{r['visual_ms']:.3f}|")
lines += ['','6 组 GEMM 尺寸逐值验证通过。完整模型与动态输入输出一致性：'+str(checks),'',
'包含 39/193 token、horizon 8/16/32 的测试；仅证明相对对应 INT8 路径的分块/屏障改动一致，不代表量化与 BF16 accuracy 一致或机器人任务成功率不变。','',
'```bash','cd /Users/qklj/work/cx/ImageWAM_MLX','RUN_TAG=$(date +%Y%m%d_%H%M%S)','.venv/bin/python development/legacy_experiments/experiments_tiles_20261104/e2e_no_barrier.py \\', '  --case tile32k512 --iterations 30 --output records/$RUN_TAG-no-barrier','```','',
'原屏障对照将脚本名改为 e2e.py；各启动模式端到端耗时见上表。完整加载、首次、逐次时间、模块和输出在 `records/barrier_20261104/`。原始内核参考 Cider MIT，许可证位于 `development/legacy_experiments/experiments_lowbit_20261104/kernels/LICENSE.cider`。','',
'保持默认 accuracy/speed：该实验不改变原有量化误差，且需要相对现有 speed 的稳定端到端优势才可升级默认。']
Path('README_INT8_BARRIER_20261104.md').write_text('\n'.join(lines)+'\n')
with tarfile.open(root/'review.tar.gz','w:gz') as t:
 for p in root.rglob('*'):
  if p.is_file() and p.suffix in ('.json','.npz','.txt'):t.add(p,arcname=str(p))
 for p in Path('development/legacy_experiments/experiments_tiles_20261104').glob('*.py'):t.add(p,arcname=str(root/'source'/p.name))
print(json.dumps(result,indent=2))
