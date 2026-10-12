"""Archive #94/#126/#145/#148/#154 and D.4: independent research prerequisites.

The full candidate CPU run retains its 120s timeout result. This alternative
research runner requires completed baseline CPU math/prepare plus the exact
publisher CPU byte gate. Complete CAModel output/oracle/status/actor gates
remain unchanged and precede adoption of any timing. No production gate edits.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_b_nz.generate_entrypoints import artifacts

PREREQUISITES='''
def research_prerequisites(a):
    from csrc.experiments.c4_b_nz.cv_compare import load
    from csrc.experiments.c4_b_nz import publisher_probe
    baseline,control,hashes=load(a.prepared,a.cpu_control,'baseline')
    if 'outputs' in baseline:raise ValueError('real baseline CPU completion required')
    fixture_path=str(a.fixture.resolve())
    if baseline['_cpu_provenance']['fixture']!=fixture_path:
        raise ValueError('completed CPU baseline used another fixture')
    source,identity=publisher_probe.generated()
    if (Path(__file__).parent/'publisher_probe.cpp').read_text()!=source:
        raise ValueError('publisher CPU source changed')
    pc=json.loads(a.publisher_control.read_text())
    if (pc.get('child_returncode')!=0 or pc.get('supervisor_returncode')!=0 or
        pc.get('timed_out') is not False or pc.get('interrupt_signal') is not None or
        pc.get('cleanup_complete') is not True or pc.get('timeout_seconds')!=60):
        raise ValueError('completed bounded publisher CPU run required')
    command=pc['command']
    if (len(command)!=2 or Path(command[0]).name!='oscar_c4_b_nz_publisher_cpu' or
        (Path(pc['cwd'])/command[1]).resolve()!=a.publisher_cpu.resolve()):
        raise ValueError('publisher output/command identity mismatch')
    proof=publisher_probe.verify(a.publisher_cpu,identity)
    return {'baseline_CPU':'complete_independent_oracle_and_prepare',
            'baseline_control_sha256':sha(a.cpu_control.read_bytes()),
            'baseline_output_sha256':hashes,
            'publisher_CPU':proof,
            'publisher_control_sha256':sha(a.publisher_control.read_bytes()),
            'full_candidate_CPU':'not_complete_120s_timeout_preserved',
            'scope':'research_only_full_CAModel_numerics_still_required'}

'''


def generated():
    original=(HERE/'cv_replay.py').read_text()
    if original!=artifacts()['cv_replay.py'][0]:raise ValueError('parent replay drift')
    edits=[
        ('Requires real CPU prepare/status/oracle completion.',
         'Requires complete baseline CPU and an independent publisher CPU byte gate.\nFull candidate CPU is incomplete; full CAModel math must independently pass.',1),
        ('def run(a):\n',PREREQUISITES+'def run(a):\n    prerequisites=research_prerequisites(a)\n',1),
        ("receipt.get('arm')!=a.variant", "receipt.get('arm')!='baseline'",1),
        ("receipt.get('rv_layout')!=('NZ_C0_8' if a.variant=='candidate' else 'ND')",
         "receipt.get('rv_layout')!='ND'",1),
        ('real complete Q336/C113 CPU metadata/evidence required',
         'real complete Q336/C337 baseline CPU metadata/evidence required',1),
        ("(a.prepared/'rv_input.bin').read_bytes()!=inputs['rv']",
         "(a.prepared/'rv_input.bin').read_bytes()!=natural_rv",1),
        ("'simulator_config':configuration,'NPU':'not_run'",
         "'CPU_prerequisites':prerequisites,'simulator_config':configuration,'NPU':'not_run'",1),
        ("    raise SystemExit(run(p.parse_args()))",
         "    for flag in ('cpu-control','publisher-cpu','publisher-control'):p.add_argument('--'+flag,type=Path,required=True)\n    raise SystemExit(run(p.parse_args()))",1),
    ]
    text=original
    for old,new,count in edits:
        if text.count(old)!=count:raise ValueError(('parent guard identity',old[:80]))
        text=text.replace(old,new)
    restored=text
    for old,new,count in reversed(edits):
        if restored.count(new)!=count:raise ValueError('inverse replay edit ambiguity')
        restored=restored.replace(new,old)
    if restored!=original:raise ValueError('full source restoration failed')
    identity={'parent_sha256':hashlib.sha256(original.encode()).hexdigest(),
              'generated_sha256':hashlib.sha256(text.encode()).hexdigest(),
              'full_parent_restore':True,
              'full_candidate_CPU':'timeout_not_relabelled_as_passed',
              'unchanged':'full CAModel inputs, outputs, independent oracle, task/status/guard/profile gates'}
    return text,identity


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    text,identity=generated();dest=HERE/'cv_replay_publisher.py'
    manifest=HERE/'publisher_replay_identity.json'
    encoded=json.dumps(identity,indent=2)+'\n'
    if a.check:
        if dest.read_text()!=text or manifest.read_text()!=encoded:raise ValueError('research replay drift')
    else:dest.write_text(text);manifest.write_text(encoded)
    print(json.dumps(identity))
