"""Serial default-allocator study; preserve each process exit code and audit."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
OUT = ROOT / 'records/visionzip_default_20261109'
PRESETS = ['merge252', 'late_align32', 'prune252', 'prune216']


def main():
    assert json.loads((OUT / 'smoke.job.json').read_text())['status'] == 'complete'
    state = {'status': 'running', 'started_unix': time.time(), 'stages': []}
    path = OUT / 'suite.json'

    def save():
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(path)

    def run(name, args):
        rec = {'name': name, 'args': list(map(str, args)), 'started_unix': time.time(), 'status': 'running'}
        state['stages'].append(rec)
        save()
        with (OUT / (name + '.log')).open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', *map(str, args)], cwd=ROOT,
                                     stdout=log, stderr=subprocess.STDOUT)
            rec['pid'] = child.pid
            save()
            code = child.wait()
        rec.update(returncode=code, elapsed_s=time.time()-rec['started_unix'],
                   status='complete' if code == 0 else 'failed')
        if code:
            state['status'] = 'failed'
        save()
        print(name, rec['status'], round(rec['elapsed_s'], 2), flush=True)
        if code:
            raise SystemExit(1)

    def study(name, runner, extra, mode=None):
        run(name, [SCRIPTS/'supervise.py', runner, '--output', OUT/name, *extra])
        args = [SCRIPTS/'audit.py', OUT/name]
        if mode:
            args.append('--' + mode)
        if mode == 'repeatability':
            args.extend(['--validation', OUT/'validation'])
        run(name + '_audit', args)

    study('validation', 'validate.py', [])
    shared = ['--validation', OUT/'validation', '--benchmark', OUT/'benchmark', '--presets', *PRESETS]
    study('benchmark', 'benchmark.py', ['--validation', OUT/'validation', '--pairs', '50', '--presets', *PRESETS], 'benchmark')
    study('modules_direct', 'modules_direct.py', shared, 'direct')
    study('zip_cost', 'zip_cost.py', [*shared, '--pairs', '100'], 'zip')
    for index in (1, 2):
        study('repeatability_' + str(index), 'repeatability.py', [*shared, '--passes', '3'], 'repeatability')
    run('stability', [SCRIPTS/'stability.py', OUT])
    state.update(status='complete', elapsed_s=time.time()-state['started_unix'])
    save()


if __name__ == '__main__':
    main()
