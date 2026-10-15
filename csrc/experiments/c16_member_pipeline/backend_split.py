"""Archive #129/#145/#154: Q672 CPU timeout is not numerical completion.

Require complete new Q336/recent1 CPU pair and real full prepare checkpoint.
Full Q672 numerical acceptance still runs both arms against the dense oracle.
"""
import json
import struct
from pathlib import Path
from csrc.experiments.c16_member_pipeline import generate


def validate(a,meta,attrs):
    small=json.loads(a.small_cpu.read_text());control=json.loads(a.full_cpu_control.read_text())
    if small['status']!='complete_small_CPU_pair' or small['tokens']!=336 or small['recent']!=1:
        raise ValueError('small CPU semantic qualification missing')
    if small['manifest_sha256']!=generate.sha((generate.HERE/'manifest.json').read_bytes()):
        raise ValueError('small CPU source binding changed')
    for case,context in [('small65',65),('small337',337)]:
        row=small['cases'][case]
        for arm in ('baseline','candidate'):
            host=json.loads(Path(row['controls'][arm]).read_text());report=row['reports'][arm]
            if host['child_returncode']!=0 or host['timed_out'] or host['cleanup_complete'] is not True:
                raise ValueError('small CPU completion missing')
            if report['context']!=context or report['clusters']!=1 or report['oracle']!='numeric':
                raise ValueError('small CPU oracle/domain missing')
            completed=Path(host['command'][2])
            for name,digest in row['equal_hashes'].items():
                if generate.sha((completed/(name+'.bin')).read_bytes())!=digest:
                    raise ValueError('small CPU saved bytes changed')
    if control['timed_out'] is not True or control['timeout_seconds']!=120 or control['cleanup_complete'] is not True:
        raise ValueError('preserved original full CPU timeout required')
    if attrs!=meta['attrs'] or attrs[:5]!=[672,6,1,256,1] or attrs[6]!=2016 or attrs[13:]!=[64,256,3,1]:
        raise ValueError('full C16 shape changed')
    receipt=json.loads((a.prepared/'metadata.json').read_text())
    expected={'status':'metadata_only','tokens':672,'context':337,'task_rows':2016,
        'positions_checked':True,'guards':True,'input_immutable':True,'source2_suppressed':True}
    if any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('full prepare receipt invalid')
    if meta['context']!=337 or meta['bad_position']!=-1:raise ValueError('undeclared full fixture')
    positions=(a.prepared/'positions.bin').read_bytes()
    if struct.unpack('<672q',positions)!=tuple(range(337,1009)):raise ValueError('full position oracle changed')
    if Path(control['command'][2])/'prepared'!=a.prepared.resolve():raise ValueError('metadata ownership mismatch')
    return {'small_CPU':'complete_Q336_recent1_C65_and_C337',
        'full_CPU':'120s_timeout_preserved_metadata_only','full_CAModel_oracle':'mandatory'}
