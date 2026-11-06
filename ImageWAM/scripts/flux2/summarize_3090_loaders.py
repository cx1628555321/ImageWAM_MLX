"""Summarize completed native/action-only benchmarks without rerunning inference."""
import csv
import json
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1]).resolve()
reports = [json.loads((root / mode / 'benchmark.json').read_text()) for mode in ('native', 'action_only')]
assert all(r['status'] == 'complete' for r in reports)
a, b = reports
assert a['input'] == b['input']
assert b['execution']['loading_validation']['intermediate_hashes_equal_to_native']
assert all(b['loader']['metadata']['shared_module_checks'].values())
lines = ['# Eager 加载器对比', '', '6 次预热，20 次无钩子稳态推理，10 次模块计时。加载时间受页缓存影响，不是受控冷启动。', '', '| 指标 | native | action_only |', '|---|---:|---:|']
def row(name, values):
    lines.append('| ' + name + ' | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |')
row('模型加载 / s', [next(x['wall_ms'] for x in r['records'] if x['name']=='model_load_total')/1000 for r in reports])
row('预热总时间 / s', [r['warmup']['total_wall_ms']/1000 for r in reports])
row('E2E mean / ms', [r['steady']['latency_ms']['mean'] for r in reports])
row('E2E p95 / ms', [r['steady']['latency_ms']['p95'] for r in reports])
for key in ('max_allocated_bytes', 'max_reserved_bytes'):
    row('稳态 '+key+' / GiB', [r['steady'][key]/2**30 for r in reports])
row('驻留参数 / billion', [r['loader']['resident_parameter_elements']/1e9 for r in reports])
lines += ['', 'CUDA event 模块时间按整次推理累计，父子项不能相加。CPU 提交间隙可能包含在 event 时间内。', '', '| 模块 | native ms | action_only ms |', '|---|---:|---:|']
for module in a['modules']:
    other = next(x for x in b['modules'] if x['name']==module['name'])
    vals = [x['cuda_ms_per_inference']['mean'] for x in (module,other)]
    if all(v is not None for v in vals): row(module['name'], vals)
lines += ['', '## 加载阶段', '', '以下为各自加载计时原始记录；包含父子阶段，不能相加。', '']
for mode, r in zip(('native','action_only'),reports):
    lines += [f'### {mode}', '', '| 阶段 | wall ms |', '|---|---:|']
    lines += [f"| {x['name']} | {x['wall_ms']:.3f} |" for x in r['records'] if x['phase']=='load']
lines += ['', '## 精度与动作', '', 'A / B / C_text / A_again 的五类中间结果哈希一致，最终动作逐位一致。动作各维度完整记录于 actions_comparison.csv。', '']
with (root/'actions_comparison.csv').open('w') as f:
    writer=csv.writer(f); writer.writerow(['input','step','dimension','native','action_only','difference'])
    for label in ('A','B','C_text','A_again'):
        x,y=[np.load(root/m/f'actions_{label}.npy') for m in ('native','action_only')]
        assert np.array_equal(x,y), label
        for index in np.ndindex(x.shape): writer.writerow([label,*index,float(x[index]),float(y[index]),float(y[index]-x[index])])
        lines += [f'- {label}: shape={list(x.shape)}, max_abs_diff=0。']
lines += ['', 'A 的首个动作（其余见 CSV / NPY）：', '', '```json', json.dumps(np.load(root/'action_only/actions_A.npy')[0].tolist()), '```', '', 'Qwen 的优化边界是 27 层 Qwen3Model，无 final norm 和 LM head；原始 benchmark.json 的通用 module_boundaries 文案描述的是 native CausalLM，具体实现以 loader.metadata 为准。', '']
(root/'comparison.md').write_text('\n'.join(lines))
print(root/'comparison.md')
