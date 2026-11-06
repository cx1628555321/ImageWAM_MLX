import json,tarfile,shutil
from pathlib import Path
root=Path('records/naxkv_20261104')
r=json.loads((root/'paired/results.json').read_text())['segmented']
p=Path('README_NAX_KV_20261104.md')
s=p.read_text()
s+='\n## 同进程配对复核\n\n同一模型与权重，交替 AB/BA 顺序，100 对，双方各 4 次预热。每次都重新执行三图、文本、视觉前缀与 10 步动作推理，不复用跨请求特征。基线启用当前 speed KV 工作区，候选切换至 NAX 双段内核。\n\n'
s+=f"基线平均 **{r['baseline_mean']:.3f} ms**，候选 **{r['candidate_mean']:.3f} ms**；平均节省 **{r['saving_ms']:.3f} ms**，配对 bootstrap 95% 区间 **[{r['ci95_ms'][0]:.3f}, {r['ci95_ms'][1]:.3f}] ms**。该区间只反映本次配对样本，不涵盖跨运行系统波动。\n\n"
s+=f"100 对最终动作及归一化动作均逐值一致，max abs={r['max_abs']}。动态输入的额外测试见上表。这是相对当前量化 speed 的一致性，不代表相对 PyTorch 的误差为零，也不是闭环任务成功率验证。\n\n"
s+='### 处理结论\n\n候选保留为独立实验，不改生产默认路径。首轮端到端收益仅 0.107 ms，而独立同步的动作模块从 75.282 ms 增至 76.055 ms；100 对配对复核进一步确认候选慢 0.389 ms（约 0.16%），节省区间整体小于零，因此不合入默认路径。不能凭微基准宣传整体提速。\n\n原因分析：双段寻址省去了工作区尾部更新，但当前工作区已避免重复完整拼接；NAX 计算本身与官方实现相同，跨段 fragment 新增的索引和分支可能抵消节省。此为代码结构推断，尚无 GPU 时间线证明各项成本。后续更值得研究动作 Linear 与门控/残差 epilogue 的融合；本轮未实现该项，不能计入已获得收益。\n'
p.write_text(s)
readme=Path('README.md');shutil.copy2(readme,root/'README.before_nax.md')
link='[NAX 双段 KV 对照测试](README_NAX_KV_20261104.md)'
if link not in readme.read_text():readme.write_text(readme.read_text()+'\n- '+link+'：包含独立进程 ABBA、100 对同进程复核及输出一致性；候选保持实验入口。\n')
with tarfile.open(root/'review_final.tar.gz','w:gz') as t:
 for p in root.rglob('*'):
  if p.is_file() and p.suffix in ('.json','.npz','.log','.md'):t.add(p,arcname=str(p))
 for p in Path('development/legacy_experiments/experiments_naxkv_20261104').glob('*'):
  if p.is_file():t.add(p,arcname=str(root/'source'/p.name))
print({k:v for k,v in r.items() if k!='pairs'})
