from pathlib import Path
import json,hashlib,tarfile,shutil
import numpy as np
root=Path('records/gate_concat_20261104');src=Path('development/legacy_experiments/experiments_gate_concat_20261104')
runs={n:json.loads((root/n/'results.json').read_text()) for n in ('paired','exact_paired','paired_2d')}
lines=['# 门控与激活拼接融合（2026-11-04）','','## 范围与基准','','在上一轮自适应 Split-K 路径上，把动作单流块中的 concatenate([attention, SiLU(a)*b]) 合为一个 Metal 内核，尝试省去中间门控输出及一次拼接读取。涉及20个单流块，每次请求10步。三图预处理、文本长度、权重、量化配置和去噪步数不变。没有增加全权重反量化。','','基线是上一轮 adaptive Split-K，不是生产 speed 或 PyTorch。生产代码和上一轮实验入口未修改。','','环境：M5 Pro，15 CPU核/16 GPU核，24GiB统一内存；macOS27.2，Python3.12.13，MLX0.32.4。','','## 三种实现','','1. 首版：一维索引，直接 sigmoid=1/(1+exp(-x))。与MLX公式及精度行为不同，产生差异，淘汰。','2. 公式对齐版：按已安装 MLX unary_ops.h 的 Sigmoid 实现，使用 precise::exp(abs(x)) 和符号分支，保留 BF16 舍入。','3. 二维版：公式不变，直接以线程 x/y/z 对应列/行/批次，去掉逐元素除法、取模计算。读取真实 strides，不要求MLP切片连续。','','## 算子与端到端验证','','随机算子探针包含 strided MLP 输入、horizon 1/8/14/16/32/64。公式对齐版及二维版在所有这些探针中逐值一致；二维版增加 max_abs==0 断言。首版最大差0.25（随机激活输出，不是最终动作误差），不能使用首版替换原逻辑。','','每版各100对完整请求，AB/BA交替，双方各首次及3次预热后计时。真实三图、39token文本、16×14动作、10步；每次重新执行完整前向。计时包含CPU输出，不包含模型加载、磁盘读取及保存。模块分析另做3次同步前向。首次调用共享编译缓存，不能比较冷启动收益。','','|版本|基线均值 ms|候选均值 ms|节省 ms|节省95%区间 ms|最终动作max abs|','|---|---:|---:|---:|---|---:|']
for name,r in runs.items():lines.append(f"|{name}|{r['base_ms']:.3f}|{r['fused_ms']:.3f}|{r['saving_ms']:.3f}|[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]|{r['max_abs']:.8f}|")
lines+=['','区间为本轮配对bootstrap，不涵盖跨运行系统变化。不同版本各自与同轮基线比较。','','## 动态输入与模块计时','','|版本|tokens|horizon|steps|max abs|RMSE|','|---|---:|---:|---:|---:|---:|']
for name,r in runs.items():
 for d in r['dynamic']:lines.append(f"|{name}|{d['tokens']}|{d['horizon']}|{d['steps']}|{d['max_abs']:.8f}|{d['rmse']:.8f}|")
lines+=['','这些结果只与上一版自适应融合路径比较，不表示相对PyTorch没有误差，也不是闭环任务成功率。没有为已确认不提速的候选追加51组样本测试。','','|版本|动作模块基线 ms|候选 ms|','|---|---:|---:|']
for name,r in runs.items():
 a=np.mean([v['action_denoise'] for v in r['modules']['base']]);b=np.mean([v['action_denoise'] for v in r['modules']['fused']]);lines.append(f'|{name}|{a:.3f}|{b:.3f}|')
lines+=['','逐模块同步会改变调度，不能拿模块求和代替端到端。','','## 结论','','本轮未将候选合入默认路径。若完整对照没有正向收益，减少一个中间张量或内核并不构成有效优化。原图已通过 mx.compile 编译，自定义融合改变了访存、向量化和调度方式；没有GPU时间线，本轮不能把退化精确归因到其中某一项。','','保留代码和负结果，后续优先通过可用的算子级计时/内存记录定位主要开销，避免继续仅按算子数量猜测收益。可以考虑在量化投影的输入读取中直接消费两段激活，但需防止多个输出块重复计算门控；该方案本轮未实现，不计入收益。','','## 复现','','```bash','cd /Users/qklj/work/cx/ImageWAM_MLX','.venv/bin/python development/legacy_experiments/experiments_gate_concat_20261104/probe.py','.venv/bin/python development/legacy_experiments/experiments_gate_concat_20261104/paired.py','.venv/bin/python development/legacy_experiments/experiments_gate_concat_20261104/probe_exact.py','.venv/bin/python development/legacy_experiments/experiments_gate_concat_20261104/paired_exact.py','.venv/bin/python development/legacy_experiments/experiments_gate_concat_20261104/probe_2d.py','.venv/bin/python development/legacy_experiments/experiments_gate_concat_20261104/paired_2d.py','```','','这些脚本拒绝覆盖既有记录；重复运行请在脚本副本中设置新的 root。每条paired命令的耗时见上表。源文件依赖上一轮 development/legacy_experiments/experiments_split_tune_20261104，依赖源码哈希已归档。模型加载/首次耗时、逐对计时、动态输出npz保存在records/gate_concat_20261104。']
Path('README_GATE_CONCAT_20261104.md').write_text('\n'.join(lines)+'\n')
files=[p for p in src.iterdir() if p.is_file()]
deps=[Path('development/legacy_experiments/experiments_split_tune_20261104')/n for n in ('tile_candidates.py','tiled.py','epilogue.h','LICENSE.mlx','upstream.json')]
deps+=[Path('imagewam_mlx')/n for n in ('runtime.py','workspace.py','ops.py','quantized.py')]
(root/'source_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files+deps},indent=2))
readme=Path('README.md');backup=root/'README.before_gate.md'
if not backup.exists():shutil.copy2(readme,backup)
link='[门控与激活拼接融合实验](README_GATE_CONCAT_20261104.md)'
if link not in readme.read_text():readme.write_text(readme.read_text()+'\n- '+link+'：三种实现、每项100对完整请求对照，保留实验记录。\n')
with tarfile.open(root/'review.tar.gz','w:gz') as t:
 for p in root.rglob('*'):
  if p.is_file() and p.suffix in ('.json','.npz','.log','.md'):t.add(p,arcname=str(p))
 for p in files:t.add(p,arcname=str(root/'source'/p.name))
 for p in deps:t.add(p,arcname=str(root/'dependencies'/p))
print({n:{k:r[k] for k in ('base_ms','fused_ms','saving_ms','ci95','max_abs')} for n,r in runs.items()})
