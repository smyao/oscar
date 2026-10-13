"""Archive #129/#145/#148/#154-157: separate explicit backend contract.

Original strict replay remains byte-unchanged. Only its CPU prerequisite is
replaced; CAModel launches, all outputs/guards/oracle/actor checks are identical.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from csrc.experiments.c4_composed_long import generate


def generate_replay():
    original = (generate.HERE / 'cv_replay.py').read_text()
    expected = generate.artifacts()['cv_replay.py'][0]
    if original != expected: raise ValueError('strict replay changed')
    begin = original.index("    context=meta['context'];bad=meta['bad_position'];receipt=")
    end = original.index("    inputs={name:", begin)
    replacement = "    context=meta['context'];bad=meta['bad_position']\n    from csrc.experiments.c4_composed_long.backend_split_contract import validate\n    qualification=validate(a,meta,attrs)\n"
    edits = [(original[begin:end], replacement, 1),
        ("'scope':meta['scope'],'kernel':identity['symbol']", "'qualification':qualification,'scope':meta['scope'],'kernel':identity['symbol']", 1),
        ("    raise SystemExit(run(p.parse_args()))", "    p.add_argument('--small-cpu-evidence',type=Path,required=True)\n    p.add_argument('--cpu-timeout-control',type=Path,required=True)\n    raise SystemExit(run(p.parse_args()))", 1)]
    changed = generate.transform(original, edits)
    # Full device portion must stay exactly identical, rather than reimplement
    # or remove a failing numerical/actor gate after the CPU timeout.
    suffix = "    save_json(out/'inputs.json',report);stop=threading.Event()"
    last = "    return 2 if failures else 0"
    if original[original.index(suffix):original.index(last)] != changed[changed.index(suffix):changed.index(last)]:
        raise ValueError('full CAModel execution/oracle gates changed')
    return ('# Explicit backend split after preserved CPU1341 timeout; never a CPU1341 pass.\n' + changed)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--check', action='store_true'); a = p.parse_args()
    text = generate_replay(); path = generate.HERE / 'cv_replay_backend_split.py'
    if a.check:
        if path.read_text() != text: raise ValueError('generated backend split replay drift')
    else: path.write_text(text)
    print(json.dumps({'status': 'checked' if a.check else 'generated',
        'strict_replay_unchanged': True, 'complete_CAModel_body_byte_unchanged': True,
        'new_replay_sha256': hashlib.sha256(text.encode()).hexdigest()}))
