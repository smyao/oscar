"""Complete old/current/batch8 writers; no local or partial timing promotion."""
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.hadamard_batch8 import generate
from csrc.experiments.hadamard_batch8.replay_writer import ARMS


def compare(directory):
    generate.check();binding=generate.source_binding();arms={}
    for name,(symbol,source) in ARMS.items():
        if name=='old' and not (directory/name).exists():continue
        path=directory/name;r=json.loads((path/'report.json').read_text())
        complete=json.loads((path/'completion.json').read_text())
        control=json.loads((directory/(name+'-control.json')).read_text())
        if (r['arm']!=name or r['status']!='passed' or r['failures'] or complete['failures'] or
            complete['status']!='device_complete_outputs_saved' or r['outputs']!=complete['outputs'] or
            control['child_returncode']!=0 or control['supervisor_returncode']!=0 or control['timed_out'] or
            control['interrupt_signal'] is not None or control['cleanup_complete'] is not True or control['timeout_seconds']!=180):
            raise ValueError(name+' lacks complete180s numeric/cleanup evidence')
        if r['source_binding']!=binding or r['kernel_sha256']!=binding['sources'][source] or r['kernel']!=symbol:
            raise ValueError('source/symbol provenance mismatch')
        for field,record in r['outputs'].items():
            data=(path/record['path']).read_bytes();guarded=(path/(field+'.guarded.bin')).read_bytes()
            expected=r['input_sha256'][field[:-6]] if field.endswith('_after') else r['oracle_sha256'][field]
            if (len(data)!=record['bytes'] or generate.sha(data)!=record['sha256'] or generate.sha(data)!=expected or
                generate.sha(guarded)!=record['guarded_sha256'] or guarded[:64]!=b'\xa5'*64 or guarded[-64:]!=b'\xa5'*64):
                raise ValueError('saved input/output/oracle/guard differs')
        task=r['profiles']['task'];actors=r['profiles']['aiv']
        if len(task)!=1 or r['task_ticks']!=task[0]['duration'] or r['task_ticks']<=0 or task[0]['duration']!=task[0]['end_cycle']-task[0]['start_cycle']:
            raise ValueError('incomplete task profile')
        if len(actors)!=r['grid'] or {x['core_id'] for x in actors}!=set(range(r['grid'])) or any(
            x['duration']<=0 or x['duration']!=x['end_cycle']-x['start_cycle'] or
            (x['task_id'],x['stream_id'])!=(task[0]['task_id'],task[0]['stream_id']) for x in actors):
            raise ValueError('incomplete actor profile')
        arms[name]=r
    for r in arms.values():
        for field in ('case','geometry','grid','library_sha256','helpers','fixture_sha256','input_sha256','initial_sha256','oracle_sha256','outputs','simulator_configuration','source_binding','build_manifest_sha256'):
            if r[field]!=arms['current'][field]:raise ValueError('unpaired '+field)
    ticks={k:v['task_ticks'] for k,v in arms.items()}
    return {'status':'complete_matched_writer_current_batch8','ticks':ticks,
        'current_over_batch8':ticks['current']/ticks['batch8'],
        'old_over_batch8':ticks['old']/ticks['batch8'] if 'old' in ticks else None,
        'old_not_rerun':'old baseline evidence retained separately; no cross-library speed ratio' if 'old' not in ticks else None,
        'scope':'full writer simulated ticks only','NPU':'not_run','graph':'not_run','whole_service':'not_run','promotion_allowed':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=compare(a.directory);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
