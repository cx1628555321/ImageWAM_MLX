from pathlib import Path
import json,hashlib,shutil,tarfile,subprocess,platform
from importlib.metadata import version
import mlx.core as mx
root=Path('records/visual_swizzle_20261104');exp=Path('development/legacy_experiments/experiments_visual_swizzle_20261104')
prior=json.loads(Path('records/visual_layout_20261104/metadata.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in prior['source_sha256'].items() if p.startswith('imagewam_mlx/'))
full=json.loads((root/'validation/full_cases.json').read_text());assert len(full)==51 and all(x['exact'] for x in full)
val=json.loads((root/'validation/results.json').read_text());assert len(val)==12 and all(x['differences']['sw2']['max_abs']==0 for x in val)
meta=dict(device=mx.device_info(),python=platform.python_version(),mlx=version('mlx'),production_unchanged=True,full_cases=len(full),extra_cases=len(val),source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in exp.iterdir() if f.is_file()});(root/'metadata.json').write_text(json.dumps(meta,indent=2));(root/'processes_after.txt').write_text(subprocess.check_output(['ps','-axo','pid,pcpu,comm'],text=True));shutil.copytree(exp,root/'source',ignore=shutil.ignore_patterns('__pycache__'))
p=json.loads((root/'prefill/results.json').read_text());e=json.loads((root/'paired/results.json').read_text());rows=[]
for title,d,n in [('视觉前缀（sw1）',p,'sw1'),('视觉前缀（sw2）',p,'sw2'),('在线端到端（sw2）',e,'sw2')]:
 r=d[n];rows.append(f"|{title}|{d['baseline']['mean']:.3f}|{r['mean']:.3f}|{r['saving']:.3f}|[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]|")
report='''# 视觉 BF16 NAX 线程块遍历优化（2026-11-04）

本轮完成四种遍历顺序的内核适配与微测、完整视觉前缀和在线端到端对照，以及 51 样本与 12 组动态边界验证。`swizzle=2` 在本轮视觉前缀诊断中有正收益，但在线端到端收益区间仍跨零，保留为实验候选，不进入默认最佳路径。

## 实现

复用上一轮 BF16 NAX 输入投影实现，只替换 19 个单流块的 linear1，tile 为 BM=64、BN=64、WM=2、WN=2、BK=64。测试 swizzle 1、2、4、8，逻辑线程块坐标为：

```
block_m = group_y * swizzle + group_x % swizzle
block_n = group_x / swizzle
```

调整线程块遍历次序以尝试改善输入/权重缓存局部性；每个输出元素的乘加计算保持不变。动态 token 数决定 M，额外线程块统一提前退出，尾块按实际 M 安全加载和写回。没有 padding 固定文本长度，没有跨请求特征缓存，也没有新增权重反量化。

BF16 输入/权重、FP32 累加、BF16 输出。复用 MLX 0.32.4 安装版本 gemm_nax.h，源哈希和 LICENSE 随代码保留。该内核仅作为 M5 Pro 上的实验，尚未泛化验证其他设备。

## 环境及方法

M5 Pro，15 CPU / 16 GPU 核、24 GiB 统一内存；MLX 0.32.4，Python 3.12.13。项目路径 `/Users/qklj/work/cx/ImageWAM_MLX`。正式基线使用 `--profile speed` 全部已验证优化，未改动 accuracy 或正式生产源码。

1. 微测使用真实首层输入，4 个候选和原生投影，每配置预热 3 次、100 轮随机交错测量。四种输出均逐值一致。sw1/sw2 微测均值约 3.005/3.011 ms，原生 3.216 ms；sw2 并未在微测中优于 sw1。
2. 视觉前缀诊断：真实样本的 hidden/latent/state 预先算好，仅计时完整 prefill，比较所有 25 层 KV；每配置预热 5 次，150 轮随机顺序。**这是模块诊断，不是端到端，也不是缓存输入的部署路径。**
3. 在线端到端：基线和 sw2 各预热 5 次，150 轮随机顺序配对；每个请求重新执行三图预处理、在线文本/图像编码、prefill、10 步动作去噪及 CPU 反归一化。16×14 动作，同一初始噪声，计时前后同步；不含磁盘读取、模型加载与输出落盘。

测试开始时 mediaanalysisd 约 118% CPU、XProtectRemediatorAdload 约 86% CPU，同时有视频服务、虚拟机活动。未停止系统服务；没有 GPU timeline，不能确定后台 GPU 干扰程度。源码哈希、设备信息及结束时进程列表保存在 records。

## 性能结果

节省 = 基线 − 候选。区间为逐轮差值的 10000 次 bootstrap 95% 区间，仅描述本轮，未校正时序相关性、多重比较或候选选择效应。

|范围|基线 ms|候选 ms|节省 ms|节省的 95% 区间 ms|
|---|---:|---:|---:|---|
'''+ '\n'.join(rows)+'''

sw2 视觉前缀平均减少 3.506 ms（约 2.71%），全部 KV 逐值一致。端到端平均减少 2.141 ms（约 0.82%），但区间跨零，不能据此宣称整网稳定提速。历史最佳约 226.641 ms 为不同负载下的结果，不能直接与本轮 259–261 ms 作优化前后对比。

端到端另保存每配置 5 次同步模块计时；这些额外同步改变调度，不得把模块之和当作正常端到端。仅凭本轮数据不能唯一确定前缀收益为何没有成为可靠端到端收益，可能包含调度、缓存状态及系统噪声差异，仍需独立复测。

## 精度与输入可变性

- 标准样本：25 层 K/V 全部逐值一致。
- 51 个代表样本/文本长度场景：最终动作与归一化动作全部与正式 speed 逐值一致。
- 31、39、79、199 token × horizon 1、16、64，12 组全部逐值一致且有限。
- 51 组验证使用各自对应的三图/状态/文本；额外 12 组固定三图和状态，改变文本与 horizon。没有声称遍历整个 RoboTwin 数据集或测量闭环成功率。

这里的精度基准是当前 W1 action 的 speed，不代表 speed 与 PyTorch/accuracy 分支本身无损。默认 accuracy 和正式 speed 的源码哈希均确认未改变。

## 记录与启动命令

所有脚本位于远端 development/legacy_experiments/experiments_visual_swizzle_20261104，本地镜像 experiments/visual_swizzle_20261104。重跑需把脚本 root 改为新目录，避免覆盖原始记录；按顺序执行，不并行测试 GPU。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
# 微测：不作为端到端
.venv/bin/python development/legacy_experiments/experiments_visual_swizzle_20261104/probe.py
# 模块诊断：129.390 -> 125.884 ms，不是端到端
.venv/bin/python development/legacy_experiments/experiments_visual_swizzle_20261104/prefill.py
# 在线端到端：261.171 -> 259.031 ms，收益尚不显著
.venv/bin/python development/legacy_experiments/experiments_visual_swizzle_20261104/paired.py
# 51 代表样本 + 12 组动态边界
.venv/bin/python development/legacy_experiments/experiments_visual_swizzle_20261104/validate.py
```

内核头文件 gemm.h、上游 LICENSE 和源文件哈希随实验目录及 source 快照保存，无需下载其他模型。

- [微测](records/visual_swizzle_20261104/probe/probe.json)
- [视觉前缀](records/visual_swizzle_20261104/prefill/results.json)
- [在线端到端](records/visual_swizzle_20261104/paired/results.json)
- [51 样本验证](records/visual_swizzle_20261104/validation/full_cases.json)
- [动态边界](records/visual_swizzle_20261104/validation/results.json)

验证目录保存实际输出数组和 token IDs；完整日志为 validation.log。review.tar.gz 包含记录、源码快照与本报告。

下一步应优先在稳定负载下独立复测 sw2 的在线端到端收益，再决定是否固化。尚不建议继续堆叠更多未经整网证实的内核变更。默认最佳版本保持 [此前已验证版本](README_BEST_SPEED_20261104.md)。
'''
Path('README_VISUAL_SWIZZLE_20261104.md').write_text(report);p=Path('README.md');s=p.read_text();link='最新视觉遍历优化：[README_VISUAL_SWIZZLE_20261104.md](README_VISUAL_SWIZZLE_20261104.md)（前缀诊断改善，端到端收益待确认）。\n\n';p.write_text(link+s) if 'README_VISUAL_SWIZZLE_20261104.md' not in s else None
with tarfile.open(root/'review.tar.gz','w:gz') as t:
 for f in sorted(root.rglob('*')):
  if f.is_file() and f.name!='review.tar.gz':t.add(f,arcname=str(f))
 t.add('README_VISUAL_SWIZZLE_20261104.md')
print('archive_sha256',hashlib.sha256((root/'review.tar.gz').read_bytes()).hexdigest());print('EXACT',len(full),len(val))
