"""Archive #94/#129/#145/#154: one logging-repaired C65 three-arm attempt.

Kernel/library/fixture/deadline and original complete numeric/profile gates
are unchanged. Only the replayer's entire unsafe polling daemon is removed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c16_l1_reuse_20261015 import generate,small_three_arm

original_bound=small_three_arm.bound


def safe_bound(root,output,label,seconds,command,env=None):
    runner=generate.HERE/'cv_replay_small.py'
    if command[:2]!=[sys.executable,str(runner)] or seconds!=600:
        raise ValueError('only the declared complete C16 runner and original600s deadline may be adapted')
    expected=hashlib.sha256(runner.read_bytes()).hexdigest()
    wrapped=[command[0],str(generate.HERE/'safe_logging_runner.py'),
        '--runner',str(runner),'--expected-sha256',expected,'--',*command[2:]]
    return original_bound(root,output,label,seconds,wrapped,env)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('output','fixture','stable-cpu','small-cpu','library','env'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.context=65
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.resolve())
    changes=json.loads(a.env.read_text());config=Path(changes['CAMODEL_CONFIG_PATH'])
    if json.loads((config/'config.json').read_text())['LOG']['core_enable_mask']!=['0x0']:
        raise ValueError('official zero-mask logging environment required')
    small_three_arm.bound=safe_bound
    raise SystemExit(small_three_arm.run(a))
