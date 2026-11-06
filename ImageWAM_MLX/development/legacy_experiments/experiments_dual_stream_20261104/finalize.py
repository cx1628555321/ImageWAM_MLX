from pathlib import Path
import json,hashlib,shutil,tarfile,subprocess,platform
from importlib.metadata import version
import mlx.core as mx
root=Path('records/dual_stream_20261104');exp=Path('development/legacy_experiments/experiments_dual_stream_20261104')
prior=json.loads(Path('records/visual_layout_20261104/metadata.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in prior['source_sha256'].items() if p.startswith('imagewam_mlx/'))
cases=json.loads((root/'validation/cases.json').read_text());extra=json.loads((root/'validation/extra.json').read_text());assert len(cases)==51 and len(extra)==12 and all(x['exact'] for x in cases+extra)
tr=json.loads((root/'validation/trace_comparison.json').read_text());assert all(x['exact'] for group in tr.values() for x in group.values())
trace_files={}
for f in (root/'validation').glob('*_trace.npz'):
 h=hashlib.sha256()
 with f.open('rb') as r:
  for buf in iter(lambda:r.read(8*1024*1024),b''):h.update(buf)
 trace_files[str(f)]=dict(bytes=f.stat().st_size,sha256=h.hexdigest())
meta=dict(device=mx.device_info(),python=platform.python_version(),mlx=version('mlx'),production_unchanged=True,trace_files=trace_files,source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in exp.glob('*.py')});(root/'metadata.json').write_text(json.dumps(meta,indent=2));(root/'processes_after.txt').write_text(subprocess.check_output(['ps','-axo','pid,pcpu,comm'],text=True));shutil.copytree(exp,root/'source',ignore=shutil.ignore_patterns('__pycache__'))
e=json.loads((root/'paired/results.json').read_text());enc=json.loads((root/'encoders/results.json').read_text());rows=[]
for name,r in e['results'].items():rows.append(f"|{name}|{r['mean_ms']:.3f}|{r['median_ms']:.3f}|{r['p95_ms']:.3f}|{r['saving_ms']:.3f}|[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]|")
report='''# 文本与图像双 GPU stream 适配及测试（2026-11-04）

已实现可复用的双流适配器、同调度单流控制、完整端到端与编码阶段对照及精度回归。本轮没有确认稳定提速，保留实验，不修改正式 speed 或 accuracy 默认代码。

## 实现与依赖

`development/legacy_experiments/experiments_dual_stream_20261104/candidate.py` 中 `install(base, dual=True)` 返回共享冻结权重的独立实验模型对象。默认文本 stream 执行 Qwen 并异步提交；一个长期复用的新 GPU stream 执行图像预处理中的 GPU 运算和 VAE，并异步提交结果。视觉 prefill 在默认 stream 消费 hidden/latent/state，通过数组依赖汇合；随后仍用原动作去噪。未新增权重副本、跨请求特征缓存或固定文本长度。

图像预处理的 GPU 操作必须一并放到图像 stream，否则它会排在默认 stream 的 Qwen 后面，形成非预期串行依赖。VAE 使用独立编译包装器，在所属 stream 上首次跟踪。没有使用 Python 多线程，也没有让多个请求并发修改同一模型实例的 trace/profile。

两种 stream 在创建线程内重复使用，未逐请求创建新 stream。计时前后分别同步默认与图像 stream；最后 CPU 动作转换也确保完整请求完成。适配器明确拒绝逐模块 profile 同步，以免把重叠测试强行串行化。当前没有 GPU timeline，因此这里证明的是双 stream 提交和正确依赖，不宣称已经测到两个 GPU kernel 物理重叠的比例。

MLX 依据：[Streams](https://ml-explore.github.io/mlx/build/html/usage/using_streams.html)、[异步求值](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.async_eval.html)。GPU 多 stream 对计算密集负载不保证提速，见[维护者说明](https://github.com/ml-explore/mlx/discussions/2083)。

## 环境与测量

M5 Pro：15 CPU / 16 GPU 核，24 GiB 统一内存，MLX 0.32.4、Python 3.12.13。项目 `/Users/qklj/work/cx/ImageWAM_MLX`。使用正式最佳 speed：W1 action、BF16 视觉/文本及已有优化，三张真实 RoboTwin 图像、状态、原文本、同一初始噪声，16×14 动作和 10 步去噪。

端到端包含在线分词、三图预处理、全部编码与推理、CPU 反归一化，不含文件读取、模型加载和保存。三种配置各记录首次调用、预热 5 次，然后 120 轮随机顺序配对测量：
- baseline：正式当前版本。
- serial_control：与双流相同显式提交和 VAE 包装器，但使用同一个 GPU stream。
- dual：文本和图像使用不同 GPU stream。

首次调用按配置顺序进行、共享已加载权重及部分编译缓存，不能将其比较作为独立冷启动收益。逐轮差值做 10000 次 bootstrap，区间仅描述本轮，未校正时序相关性或多重比较。

## 在线端到端结果

节省为正式 baseline 减候选。

|配置|均值 ms|中位数 ms|P95 ms|节省 ms|节省 95% 区间 ms|
|---|---:|---:|---:|---:|---|
'''+ '\n'.join(rows)+f'''

双流相对同调度单流控制，平均节省 {e['dual_vs_serial']['saving_ms']:.3f} ms，95% 区间 [{e['dual_vs_serial']['ci95'][0]:.3f}, {e['dual_vs_serial']['ci95'][1]:.3f}] ms。没有可靠的端到端加速证据，均值方向反而更慢。

编码阶段整体另做 100 轮配对，保留在线分词和预处理，一起等待文本与图像完成：单流 **{enc['means_ms']['serial']:.3f} ms**、双流 **{enc['means_ms']['dual']:.3f} ms**，节省 {enc['saving_ms']:.3f} ms，区间 [{enc['ci95'][0]:.3f}, {enc['ci95'][1]:.3f}] ms。区间同样跨零。这是编码阶段诊断，不是整网端到端。

历史“两个约 27 ms 编码器完全重叠可节省约 27 ms”只是无资源争用的乐观上限，本轮未实现该收益。当前单 GPU 上算力/带宽争用、提交同步成本及系统噪声都可能影响结果；没有时间线证据进一步归因。测试结束附近 mediaanalysisd 约 128% CPU，视频服务及虚拟机活跃，未停止任何无关服务。历史约 226.641 ms 的最佳端到端不与本轮负载下的 263–267 ms 直接作前后比较。

## 精度与可变输入

- 标准样本 baseline 与 dual：编码结果、原基线记录的 KV 层、逐步动作 trace 全部一致。
- 标准样本 serial_control 与 dual：所有 25 层 K/V、编码结果及逐步动作 trace 全部逐值一致。
- 51 个代表样本/文本长度场景：动作及归一化动作全部逐值一致。
- 31、39、79、199 token × horizon 1、16、64，共 12 组全部逐值一致。

51 组使用各自对应的图像/状态/文本，额外 12 组固定图像/状态而变化文本和 horizon。这里的参照是正式 speed，不代表其已有 W1 量化相对 accuracy/PyTorch 无损，也不是闭环任务成功率验证。

## 复现及记录

输出目录必须不存在，重跑应更改脚本内 root，避免覆盖结果。GPU 测试按顺序执行。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
# 正式基线 263.914 / 单流控制 262.356 / 双流 267.131 ms
.venv/bin/python development/legacy_experiments/experiments_dual_stream_20261104/paired.py
# 编码阶段诊断，非端到端
.venv/bin/python development/legacy_experiments/experiments_dual_stream_20261104/encoders.py
.venv/bin/python development/legacy_experiments/experiments_dual_stream_20261104/validate.py
```

代码调用方式：在项目根目录将实验目录加入 Python 搜索路径，再调用 `candidate.install(base, dual=True)`；`dual=False` 为同调度单流控制。调用 `model.predict(...)` 保持原输入和返回值格式，支持 trace，不支持 profile=True。不能把适配器当成支持多线程共享实例的服务接口。

- [端到端计时](records/dual_stream_20261104/paired/results.json)
- [编码阶段计时](records/dual_stream_20261104/encoders/results.json)
- [中间结果对比](records/dual_stream_20261104/validation/trace_comparison.json)
- [51 样本](records/dual_stream_20261104/validation/cases.json)
- [动态边界](records/dual_stream_20261104/validation/extra.json)
- [stream 标识](records/dual_stream_20261104/validation/streams.json)

完整编码/KV trace 数组保留在远端 validation/*_trace.npz，其尺寸及 SHA256 记录在 metadata.json；为避免同步数百 MB 的重复中间量，本地 review.tar.gz 不含这三个大文件。归档包含各样本最终输出、比较结果、源码快照和报告。

本轮只实现并验证了双编码器并行，没有实施逐层视觉 KV/动作第一步流水。后者会拆分现有编译图，并让同一 GPU 承担更多交错工作；建议先取得可解释的时间线或稳定负载证据，再决定是否投入该改造，不将未实现的流水计为本轮成果。
'''
Path('README_DUAL_STREAM_20261104.md').write_text(report);p=Path('README.md');s=p.read_text();link='文本/图像双流适配与测试：[README_DUAL_STREAM_20261104.md](README_DUAL_STREAM_20261104.md)（精度回归通过，未确认提速，不默认启用）。\n\n';p.write_text(link+s) if 'README_DUAL_STREAM_20261104.md' not in s else None
with tarfile.open(root/'review.tar.gz','w:gz') as t:
 for f in sorted(root.rglob('*')):
  if f.is_file() and f.name!='review.tar.gz' and not f.name.endswith('_trace.npz'):t.add(f,arcname=str(f))
 t.add('README_DUAL_STREAM_20261104.md')
print('archive_sha256',hashlib.sha256((root/'review.tar.gz').read_bytes()).hexdigest());print('EXACT',len(cases),len(extra))
