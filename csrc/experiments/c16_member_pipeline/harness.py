"""Archive #94/#126/#129/#145/#154-157: adapt complete C4 checker to C16.

Q672 preserves recent256 and exposes one mature C16 cluster plus C1/window;
all2016 prepared rows, independent original-space oracle and guards remain.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from csrc.experiments.c16_member_pipeline.generate import sha


def artifacts():
    result = {}
    old = ROOT / 'csrc/experiments/c4_member_pipeline'
    for name in ('cv_probe.cpp', 'cv_replay.py', 'cv_compare.py'):
        source = (old / name).read_text(); text = source
        text = text.replace('336', '672').replace('1008', '2016')
        text = text.replace('C4', 'C16').replace('c4_', 'c16_')
        text = text.replace('2494464', '5652480')
        if name == 'cv_probe.cpp':
            text = text.replace('bad!=672', 'bad!=336')
            text = text.replace('context!=113&&context!=337', 'context!=337&&context!=1341')
            text = text.replace('leaders!=4||skips!=4', 'leaders!=16||skips!=16')
        elif name == 'cv_replay.py':
            text = text.replace('(-1,80,96,672)', '(-1,80,96,336)')
            text = text.replace('(113,337)', '(337,1341)')
            text = text.replace("'<2016i'", "'<4032i'")
            text = text.replace('[1,4,4]', '[1,16,16]')
        else:
            text = text.replace('(113,337)', '(337,1341)')
        result[name] = (text, {'template': str((old / name).relative_to(ROOT)),
            'template_sha256': sha(source), 'generated_sha256': sha(text)})
    small=result['cv_probe.cpp'][0].replace('672','336').replace('2016','1008')
    small=small.replace('a[14]!=256','a[14]!=1')
    small=small.replace('context!=337&&context!=1341','context!=65&&context!=337')
    small=small.replace('int64_t{64},int64_t{256},int64_t{1},true','int64_t{64},int64_t{1},int64_t{1},true')
    small='// Explicit small semantic gate: Q336/recent1, not production Q672/recent256.\n'+small
    result['cv_probe_small.cpp']=(small,{'template':'generated cv_probe.cpp','generated_sha256':sha(small)})
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--check', action='store_true'); a = p.parse_args()
    manifest = {}
    for name, (text, identity) in artifacts().items():
        path = HERE / name
        if a.check:
            if path.read_text() != text: raise ValueError('harness drift ' + name)
        else: path.write_text(text)
        manifest[name] = identity
    content = json.dumps(manifest, indent=2) + '\n'
    if a.check:
        if (HERE / 'harness.json').read_text() != content: raise ValueError('harness manifest drift')
    else: (HERE / 'harness.json').write_text(content)
    print(json.dumps({'status': 'checked' if a.check else 'generated'}))


if __name__ == '__main__': main()
