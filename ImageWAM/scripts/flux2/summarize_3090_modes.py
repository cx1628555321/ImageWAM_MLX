#!/usr/bin/env python3
"""Summarize finished/failed mode experiments; never treats missing data as zero."""
import argparse
import csv
import json
from pathlib import Path


MODE_ORDER = ('eager', 'inference_mode', 'compile', 'torchscript', 'cudagraph')
MODE_SCOPES = {
    'eager': 'Native eager operators with the original no_grad boundary; no compilation or capture.',
    'inference_mode': 'Native eager operators with torch.inference_mode() around the complete predict call; no compilation or capture.',
    'compile': 'Selected GPU module boundaries execute through torch.compile; Python orchestration remains eager.',
    'torchscript': 'Selected GPU module boundaries execute through traced TorchScript modules; Python orchestration remains eager.',
    'cudagraph': 'Selected GPU module boundaries execute through CUDA Graph replay; Python orchestration remains eager.',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    reports = {}
    for mode in MODE_ORDER:
        path = directory / mode / 'benchmark.json'
        if path.exists():
            reports[mode] = json.loads(path.read_text())
            actual_mode = reports[mode].get('execution', {}).get('mode')
            if actual_mode != mode:
                raise ValueError(f'Mode mismatch: directory={mode}, report={actual_mode}')
        elif (directory / mode).exists():
            reports[mode] = {'status': 'missing_report', 'error': 'See run.log and exit_codes.tsv; preparation may have been interrupted'}
    rows = []
    eager = reports.get('eager', {})
    baseline = eager.get('steady', {}).get('latency_ms', {}).get('mean') if eager.get('status') == 'complete' else None
    for mode, report in reports.items():
        steady = report.get('steady', {})
        stats = steady.get('latency_ms', {})
        validation = report.get('execution', {}).get('validation', [])
        rows.append({'mode': mode, 'execution_scope': MODE_SCOPES[mode],
                     'status': report['status'], 'count': stats.get('count'),
                     'mean_ms': stats.get('mean'), 'p95_ms': stats.get('p95'),
                     'speedup_vs_eager': baseline / stats['mean'] if baseline and stats.get('mean') and report['status'] == 'complete' else None,
                     'preparation_ms': report.get('execution', {}).get('preparation_and_validation_ms'),
                     'warmup_ms': report.get('warmup', {}).get('total_wall_ms'),
                     'max_allocated_gib': steady['max_allocated_bytes'] / 2**30 if 'max_allocated_bytes' in steady else None,
                     'max_reserved_gib': steady['max_reserved_bytes'] / 2**30 if 'max_reserved_bytes' in steady else None,
                     'max_action_difference_vs_eager': max((x['max_abs_difference'] for x in validation), default=None),
                     'error': report.get('error')})
    modules = []
    names = list(dict.fromkeys(row['name'] for report in reports.values() for row in report.get('modules', [])))
    for name in names:
        row = {'module': name}
        for mode, report in reports.items():
            values = {entry['name']: entry for entry in report.get('modules', [])}
            entry = values.get(name)
            row[mode + '_cuda_ms'] = entry['cuda_ms_per_inference']['mean'] if entry else None
            row[mode + '_host_wall_ms'] = entry['wall_ms_per_inference']['mean'] if entry else None
        modules.append(row)
    payload = {'directory': str(directory), 'modes': rows, 'modules': modules,
               'note': 'eager and inference_mode use eager operators. inference_mode wraps the complete predict call without compilation/capture. compile, torchscript and cudagraph adapt selected GPU module boundaries while Python orchestration remains eager. Parent/child module times overlap. Preparation excludes model loading.'}
    (directory / 'comparison.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    for filename, data in [('comparison.csv', rows), ('module_comparison.csv', modules)]:
        with (directory / filename).open('w', newline='') as stream:
            if data:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader(); writer.writerows(data)
    fmt = lambda x: '—' if x is None else f'{x:.3f}' if isinstance(x, float) else str(x)
    lines = ['# 3090 执行方式比较', '', '| 模式 | 状态 | E2E平均 ms | p95 ms | 相对eager加速比 | 准备 ms | 预热 ms | allocated GiB | reserved GiB |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| ' + ' | '.join(fmt(row[key]) for key in ('mode', 'status', 'mean_ms', 'p95_ms', 'speedup_vs_eager', 'preparation_ms', 'warmup_ms', 'max_allocated_gib', 'max_reserved_gib')) + ' |')
    if 'inference_mode' in reports:
        lines += ['', '`inference_mode` 仍使用原生 eager 算子执行，仅在完整 `predict` 调用外围启用 `torch.inference_mode()`；不进行编译或 CUDA Graph 捕获。']
    lines += ['', '模块为 CUDA event 每次推理均值；纯 CPU 模块见 CSV host wall。父子行不可相加。', '', '| 模块 | ' + ' | '.join(reports) + ' |', '|---|' + '---:|' * len(reports)]
    for row in modules:
        lines.append('| ' + row['module'] + ' | ' + ' | '.join(fmt(row.get(mode + '_cuda_ms')) for mode in reports) + ' |')
    for row in rows:
        if row['error']:
            lines += ['', f"## {row['mode']} 未完成原因", '', '```text', str(row['error']), '```']
    lines += ['', '逐模式配置、输出一致性检查、编译/捕获边界与原始样本见各子目录 benchmark.json / summary.md / run.log。', '']
    (directory / 'comparison.md').write_text('\n'.join(lines))
    print(directory / 'comparison.md')


if __name__ == '__main__':
    main()
