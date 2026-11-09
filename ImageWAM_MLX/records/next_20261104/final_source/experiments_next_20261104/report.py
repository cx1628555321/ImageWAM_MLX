from pathlib import Path
import json,hashlib,tarfile
import numpy as np
root=Path('records/next_20261104');runs={p.parent.name:json.loads(p.read_text()) for p in sorted((root/'screen').glob('*/results.json'))}
assert len(runs)==6 and all(r['status']=='completed' for r in runs.values())
ref=root/'screen/00_baseline';checks={}
for name in runs:
 checks[name]={}
 for file in ('output.npz','dynamic_0.npz','dynamic_1.npz','dynamic_2.npz'):
  a=np.load(root/'screen'/name/file);b=np.load(ref/file)
  checks[name][file]={k:dict(exact=bool(np.array_equal(a[k],b[k])),max_abs=float(np.max(np.abs(a[k]-b[k]))),rmse=float(np.sqrt(np.mean((a[k]-b[k])**2)))) for k in ('action','normalized')}
summary=dict(runs=runs,checks=checks,hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('experiments_next_20261104').glob('*.py')});(root/'summary.json').write_text(json.dumps(summary,indent=2))
lines=['# 结构性优化适配与筛选（2026-11-04）','',
'在 M5 Pro 的现有 speed 路径基础上适配四项独立候选，全部通过真实串行端到端运行测试；未修改原模型权重、配置与默认推理入口。','',
'## 实现与范围','',
'|候选|实现|尚未证明的部分|','|---|---|---|',
'|workspace|每次请求分配带动作尾部的 25 层 KV；单步用 slice_update 更新动作区，并将更新后的 cache 返回下一步|函数式更新不保证底层原位写入或无完整复制，不能称为已实现零拷贝|',
'|qwen_compile|对现有 Qwen 整体调用增加 mx.compile，保持原始投影及全部 27 层|不是合并 QKV、不是减少层数，也不是 shapeless 编译|',
'|visual_addmm|视觉线性层改为 mx.addmm(bias,x,weight.T)，给后端选择 GEMM+bias 路径|不是已经实现 GEMM+SiLU/gate 的自定义 epilogue；当前权重无 bias 的层实际仍走原 matmul|',
'|vae_nhwc|直接在 NHWC 分组视图归约 mean/variance，避免 GroupNorm 前后的 transpose/reshape 重排|增加 FP32 统计/中间张量，未承诺减少实际搬运；允许与 speed 原归一化有舍入差异|','',
'视觉自定义 GEMM+gate epilogue、原生双段 KV attention、动作 W1 内核解包复用仍未实现；本轮先验证已有 MLX 算子可完成的结构调整，没有把这些后续项目标记为完成。','',
'## 环境与协议','',
'M5 Pro，15 CPU/16 GPU，24 GiB 统一内存；macOS 27.2，Python 3.12.13，MLX 0.32.4。每配置独立进程、首次前向、3 次预热、20 次稳态及3次独立同步模块剖析；前后各一组 speed 基线。每请求重新处理真实三图、状态和完整文本，无旧输入特征缓存；16×14 动作，10步。端到端包含 CPU 输出，不含模型加载、文件解码和保存。模块同步改变调度，不能直接求和。','',
'|配置|端到端均值 ms|P95 ms|加载 s|首次 ms|文本 ms|VAE ms|视觉 ms|动作 ms|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for name,r in runs.items():
 m={k:float(np.mean([v[k] for v in r['modules']])) for k in r['modules'][0]}
 lines.append(f"|{name}|{r['mean_ms']:.3f}|{r['p95_ms']:.3f}|{r['load_s']:.3f}|{r['first_ms']:.3f}|{m['text_encoder']:.3f}|{m['image_encoder']:.3f}|{m['visual_prefill']:.3f}|{m['action_denoise']:.3f}|")
lines += ['','## 数值验证','',
'与本轮 speed 基线比较，不是与 accuracy 或 PyTorch 真值比较。基准输入外，还验证 39-token/horizon8/3步、193-token/horizon16/10步、39-token/horizon32/3步；所有输出 shape 正确且有限。','',
'|配置|四组 action/normalized 是否全部逐值一致|最大动作绝对差|','|---|---|---:|']
for name,c in checks.items():lines.append(f"|{name}|{all(v['exact'] for f in c.values() for v in f.values())}|{max(f['action']['max_abs'] for f in c.values()):.8f}|")
lines += ['','## 启动','',
'```bash','cd /Users/qklj/work/cx/ImageWAM_MLX','RUN_TAG=$(date +%Y%m%d_%H%M%S)','.venv/bin/python experiments_next_20261104/bench.py \\', '  --case workspace --iterations 50 --output records/$RUN_TAG-workspace','```','',
'--case 可替换 baseline、qwen_compile、visual_addmm、vae_nhwc；端到端参考值见上表。输出目录必须不存在。记录目录 `records/next_20261104/screen/` 包含逐次时间、模块、首次/加载、输出及动态输入 noise；`summary.json` 保存数值对比和源码哈希。','',
'筛选不代表可靠的因果收益；只有具有足够幅度且复测通过的方案才升级默认。默认 accuracy 与现有 speed 均保持原样。']
Path('README_STRUCTURAL_20261104.md').write_text('\n'.join(lines)+'\n')
with tarfile.open(root/'review.tar.gz','w:gz') as t:
 for p in root.rglob('*'):
  if p.is_file() and p.suffix in ('.json','.npz','.txt'):t.add(p,arcname=str(p))
 for p in Path('experiments_next_20261104').glob('*.py'):t.add(p,arcname=str(root/'source'/p.name))
print(json.dumps({'timing':{n:r['mean_ms'] for n,r in runs.items()},'checks':checks},indent=2))
