"""One bounded CPU fixture construction; no kernel/device execution."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LIMIT = 3 * 1024**3


def child():
    sys.path.insert(0, str(ROOT))
    def guard():
        while True:
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > LIMIT:
                os._exit(75)
            time.sleep(.05)
    threading.Thread(target=guard, daemon=True).start()
    start = time.monotonic()
    import torch
    from tools import probe_c16_l1_reuse as probe
    torch.set_num_threads(2)
    spec = next(x for x in probe.case_plan() if x.contexts == (65536,))
    before = time.monotonic()
    fixture = probe.make_fixture(torch, spec)
    assert len(fixture['expected']) == 672
    assert fixture['reference_anchor_check']['status'] == 'bitwise_passed'
    data = {'status': 'constructed', 'context': 65536, 'queries': 672,
            'fixture_seconds': time.monotonic()-before, 'total_seconds': time.monotonic()-start,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'blocks': fixture['blocks'], 'virtual_columns': fixture['cpu']['table'].shape[1],
            'reference_anchor_check': fixture['reference_anchor_check'],
            'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                ('tools/probe_c16_l1_reuse.py', 'tools/c16_probe_oracle.py',
                 'tools/probe_history_reuse.py', 'oscar_ascend/ops/reference.py')},
            'NPU': 'not_run', 'device_timing': 'not_measured'}
    (HERE/'fixture.json').write_text(json.dumps(data, indent=2)+'\n')


if __name__ == '__main__':
    if '--child' in sys.argv:
        child()
    else:
        started = time.monotonic()
        with (HERE/'child.log').open('w') as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--child'],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            reason = None
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                reason = 'deadline_60s'; proc.kill(); proc.wait()
        if proc.returncode == 75: reason = 'rss_3GiB'
        data = {'status': 'bounded_host_construction_passed' if proc.returncode == 0 else 'failed',
                'limits': {'seconds': 60, 'rss_bytes': LIMIT}, 'child_returncode': proc.returncode,
                'reason': reason, 'wall_seconds': time.monotonic()-started, 'cleanup_complete': True,
                'scope': 'one 64K full672-query CPU fixture; no NPU or model performance'}
        if proc.returncode == 0: data['fixture'] = json.loads((HERE/'fixture.json').read_text())
        (HERE/'result.json').write_text(json.dumps(data, indent=2)+'\n')
        print(json.dumps(data))
        raise SystemExit(proc.returncode)
