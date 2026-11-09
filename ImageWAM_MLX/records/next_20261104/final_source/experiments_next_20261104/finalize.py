import json,hashlib,tarfile,shutil
from pathlib import Path
root=Path('records/next_20261104');done=json.loads((root/'native_validation/completed.json').read_text());assert done['cases']==51 and done['accuracy_exact']
pair=json.loads((root/'native_paired/results.json').read_text())['workspace']
checks=json.loads((root/'native_validation/results.json').read_text());assert all(r['exact'] for r in checks)
extra=json.loads((root/'native_validation/extra.json').read_text());assert all(r['trace_exact'] for r in extra)
p=Path('README_STRUCTURAL_20261104.md');old=p.read_text();(root/'screening_report.md').write_text(old)
new=f'''# KV 工作区正式适配与结构优化结果（2026-11-04）

已将 KV 工作区适配到正式 speed 运行时，accuracy 默认不启用。原始模型权重/配置未修改。实现位于 `imagewam_mlx/workspace.py`；每次请求重新创建工作区，仅在本请求的去噪步骤内传递 KV，下一请求不复用旧图像/文本特征。`slice_update` 是函数式操作，未取得底层原位复用证据，不宣称零拷贝或已消除全部前缀搬运。

## 正式入口复测

同一模型实例、共享同一份权重，切换 kv_workspace=False/True，100 组 AB/BA，双方各100次正式稳态，预热后计时。GPU 在每次端到端计时边界同步；模型内部正常使用 async_eval。不计加载/文件解码/保存；包含在线三图预处理、分词、编码、视觉前缀、16×14动作的10步去噪、CPU反归一化输出。

|模式|平均 ms|
|---|---:|
|原 speed，关闭工作区|{pair['baseline_mean']:.3f}|
|新 speed，启用工作区|{pair['candidate_mean']:.3f}|

平均节省 **{pair['saving_ms']:.3f} ms（{100*pair['saving_ms']/pair['baseline_mean']:.3f}%）**，逐对 bootstrap 95% 区间 [{pair['ci95_ms'][0]:.3f}, {pair['ci95_ms'][1]:.3f}] ms。Bootstrap 10000次、seed42，未校正时序相关性和多重比较；只证明本轮小幅收益，不保证所有场景。先前实验入口60组配对节省0.731ms，方向一致。

## 验证

- 原生 speed 新旧路径51个真实样本，覆盖全部已观测31–79 token长度（48种，缺78）和3个额外真实观察，动作与归一化数组全部逐值一致。
- 筛选额外验证193-token文本、horizon8/16/32；原生验证另覆盖horizon1/64、3步及trace中间量，逐值一致。
- accuracy 默认 kv_workspace=False；基准样本最终动作与已保存 accuracy 参考逐值一致。
- 不是全数据集或闭环机器人成功率验证；speed 原有量化误差保持，不能把“相对旧speed一致”称为“相对BF16无损”。

## 启动和回退

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
RUN_TAG=$(date +%Y%m%d_%H%M%S)
.venv/bin/python scripts/infer.py --profile speed \\
  --inputs examples/robotwin_ep0_frame0/inputs.npz \\
  --prompt-file examples/robotwin_ep0_frame0/prompt.txt \\
  --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy \\
  --output records/$RUN_TAG-speed
```

该配置本轮稳态242.085ms；单次CLI请求包含自身预热流程，不能直接视为稳态平均。run.json及benchmark报告新增kv_workspace字段。`--profile accuracy`仍走原精度路径。

Python显式回退：`ImageWAM('weights/bf16','assets',profile='speed',kv_workspace=False)`。原生构造参数默认为None，按profile选择：speed启用、accuracy关闭；legacy关闭。两条路径仍支持动态文本和动作长度。

## 未采用的候选与后续范围

- Qwen编译：基准配对有小幅收益，但动态输入有数值差异，未默认启用。
- 视觉addmm：筛选未显示收益；不是完整GEMM+gate自定义epilogue。
- VAE NHWC统计：显著变慢且有数值差异，未采用。
- 原生双段KV attention、视觉GEMM+gate epilogue和W1内核解包复用仍是后续工作，本轮没有实现或宣称完成。

环境为M5 Pro（15CPU/16GPU，24GiB统一内存），macOS27.2，MLX0.32.4。当前没有kernel时间线，不能把收益精确归因于实际减少的DRAM字节数。

记录全部位于 `records/next_20261104/`：screen是候选筛选；paired是实验配对；workspace_validation是实验精度；native_paired是正式入口100组配对；native_validation是原生精度及trace；before是修改前代码；final_source是最终源码；source_hashes.json是哈希。下方保留原筛选报告，其中“默认未修改”只描述筛选当时状态。

---

'''
p.write_text(new+old)
snap=root/'final_source';snap.mkdir(exist_ok=False)
for dirname in ('imagewam_mlx','experiments_next_20261104'):
 shutil.copytree(dirname,snap/dirname,ignore=shutil.ignore_patterns('__pycache__'))
(snap/'scripts').mkdir()
for name in ('infer.py','benchmark.py'):shutil.copy2(Path('scripts')/name,snap/'scripts'/name)
(root/'source_hashes.json').write_text(json.dumps({str(p.relative_to(snap)):hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.rglob('*.py')},indent=2))
with tarfile.open(root/'final_review.tar.gz','w:gz') as t:
 for p in root.rglob('*'):
  if p.is_file() and p.suffix in ('.json','.npz','.py','.md'):t.add(p,arcname=str(p))
p=Path('README.md');shutil.copy2(p,root/'README.before.md');p.write_text('2026-11-04：speed 默认启用 KV 工作区，100组配对平均节省0.865ms；[适配与验证报告](README_STRUCTURAL_20261104.md)。accuracy不变。\n\n'+p.read_text())
print(pair['baseline_mean'],pair['candidate_mean'],done)
