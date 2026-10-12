"""Archive #126/#145/#154-156, D.4: complete matched query-rotation arms.

All180s control/profile/byte gates are required. Host NaN oracle payloads may
differ, but actual old/init/batch8 output and status must match every bit.
No primitive ratio is a writer/CV/service performance claim.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.hadamard_batch8 import generate
from csrc.experiments.hadamard_batch8.rotate_replay import ARMS,PLAN,SCOPE,oracle_check


def compare(directory):
    generate.check();binding=generate.source_binding();arms={}
    for name,(symbol,source) in ARMS.items():
        path=directory/name
        r=json.loads((path/'report.json').read_text())
        complete=json.loads((path/'completion.json').read_text())
        control=json.loads((directory/(name+'-control.json')).read_text())
        if (r['scope']!=SCOPE or r['arm']!=name or r['status']!='passed' or r['failures'] or
                complete['failures'] or complete['status']!='device_complete_outputs_saved' or
                r['outputs']!=complete['outputs'] or control['child_returncode']!=0 or
                control['supervisor_returncode']!=0 or control['timed_out'] or
                control['interrupt_signal'] is not None or control['cleanup_complete'] is not True or
                control['timeout_seconds']!=180):
            raise ValueError(name+' lacks complete180s numeric/cleanup evidence')
        geometry=r['geometry'];shape=tuple(geometry[k] for k in ('rows','dim','dtype','heads'))
        if PLAN.get(r['case'])!=shape or geometry['hadamard'] is not True or r['grid']!=min(32,(shape[0]+7)//8):
            raise ValueError('undeclared shape/grid')
        if r['source_binding']!=binding or r['kernel_sha256']!=binding['sources'][source] or r['kernel']!=symbol:
            raise ValueError('source/symbol provenance mismatch')
        output_names={'output','status'};input_names={'source_after','rotation_after','slots_after'}
        if set(r['outputs'])!=output_names|input_names:raise ValueError('incomplete GM evidence inventory')
        outputs={}
        for field,record in r['outputs'].items():
            data=(path/record['path']).read_bytes();guarded=(path/(field+'.guarded.bin')).read_bytes()
            if (len(data)!=record['bytes'] or generate.sha(data)!=record['sha256'] or
                    len(guarded)!=len(data)+128 or guarded[64:-64]!=data or
                    generate.sha(guarded)!=record['guarded_sha256'] or
                    guarded[:64]!=b'\xa5'*64 or guarded[-64:]!=b'\xa5'*64):
                raise ValueError('saved GM payload/guard differs: '+field)
            if field in input_names and generate.sha(data)!=r['input_sha256'][field[:-6]]:
                raise ValueError('input changed: '+field)
            if field in output_names:outputs[field]=data
        oracle={key:(path/('expected_'+key+'.bin')).read_bytes() for key in output_names}
        if any(generate.sha(data)!=r['oracle_sha256'][key] for key,data in oracle.items()):
            raise ValueError('saved independent oracle differs')
        tolerance=json.loads((ROOT/'configs/acceptance.json').read_text())['store_dequant']
        if r['oracle_tolerance']!=tolerance:raise ValueError('frozen oracle tolerance differs')
        actual_oracle=oracle_check(outputs,oracle,tolerance)
        if actual_oracle!=r['oracle_result'] or actual_oracle['status']!='passed':
            raise ValueError('independent oracle not passed')
        task=r['profiles']['task'];actors=r['profiles']['aiv']
        if (len(task)!=1 or r['task_ticks']!=task[0]['duration'] or r['task_ticks']<=0 or
                task[0]['duration']!=task[0]['end_cycle']-task[0]['start_cycle']):
            raise ValueError('incomplete task profile')
        if len(actors)!=r['grid'] or {a['core_id'] for a in actors}!=set(range(r['grid'])) or any(
                a['duration']<=0 or a['duration']!=a['end_cycle']-a['start_cycle'] or
                (a['task_id'],a['stream_id'])!=(task[0]['task_id'],task[0]['stream_id']) for a in actors):
            raise ValueError('incomplete actor profile')
        arms[name]=r
    for name,r in arms.items():
        for field in ('case','geometry','grid','library_sha256','helpers','fixture_sha256','fixture_sources',
                      'input_sha256','initial_sha256','oracle_sha256','oracle_tolerance','outputs',
                      'simulator_configuration','source_binding','build_manifest_sha256'):
            if r[field]!=arms['old'][field]:raise ValueError('unpaired or bit-different '+name+': '+field)
    ticks={name:r['task_ticks'] for name,r in arms.items()}
    return {'status':'complete_matched_query_rotation','case':arms['old']['case'],'ticks':ticks,
            'old_over_init':ticks['old']/ticks['init'],'old_over_batch8':ticks['old']/ticks['batch8'],
            'init_over_batch8':ticks['init']/ticks['batch8'],'cross_arm_output_status_bits':'identical',
            'scope':SCOPE,'NPU':'not_run','graph':'not_run','whole_service':'not_run','promotion_allowed':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=compare(a.directory)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
