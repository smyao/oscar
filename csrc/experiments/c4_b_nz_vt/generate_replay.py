"""Archive #126/#140/#145/#148/#154, D.4: retain full B-NZ research gates.

Only source/library/publisher bindings change. Existing complete numerical,
task, guard and actor gates remain; incomplete large CPU evidence stays so.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_b_nz.generate_publisher_replay import generated as parent


def generated():
    original,identity=parent()
    if original!=(HERE.parent/'c4_b_nz/cv_replay_publisher.py').read_text():
        raise ValueError('parent full numerical runner drift')
    edits=(
        ('from csrc.experiments.c4_b_nz import generate as generate_c4',
         'from csrc.experiments.c4_b_nz_vt import generate as generate_c4'),
        ('from csrc.experiments.c4_b_nz import publisher_probe',
         'from csrc.experiments.c4_b_nz_vt import publisher_probe'),
        ("'oscar_c4_b_nz_publisher_cpu'","'oscar_c4_b_nz_vt_publisher_cpu'"),
        ("'liboscar_c4_b_nz.so'","'liboscar_c4_b_nz_vt.so'"),
        ('explicit B-NZ research library required','explicit Vt-NZ research library required'),
        ("'not_complete_120s_timeout_preserved'",
         "'not_run_for_Vt_full_C4_B_only_120s_timeout_preserved'"),
    )
    text=original
    for old,new in edits:
        if text.count(old)!=1:raise ValueError('replay edit identity mismatch '+old)
        text=text.replace(old,new)
    restored=text
    for old,new in reversed(edits):
        if restored.count(new)!=1:raise ValueError('inverse edit ambiguity')
        restored=restored.replace(new,old)
    if restored!=original:raise ValueError('full replay restoration failed')
    return text,{'parent':identity,'parent_sha256':hashlib.sha256(original.encode()).hexdigest(),
                 'generated_sha256':hashlib.sha256(text.encode()).hexdigest(),
                 'full_parent_restore':True,'numerical_and_completion_gates':'unchanged'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    text,identity=generated();target=HERE/'cv_replay.py';manifest=HERE/'replay_identity.json'
    encoded=json.dumps(identity,indent=2)+'\n'
    if a.check:
        if target.read_text()!=text or manifest.read_text()!=encoded:raise ValueError('Vt replay drift')
    else:target.write_text(text);manifest.write_text(encoded)
    print(json.dumps({'status':'checked' if a.check else 'generated','full_parent_restore':True}))
