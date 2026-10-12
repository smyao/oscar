"""Archive #140/#151/#154-156, D.4: matched complete writer arms only.

No partial profile, CPU timing, local pack speed or model-level extrapolation.
"""
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.window_batch_decode.replay_camodel import sha,save_json
from csrc.experiments.striped_store_simd_pack import generate


def compare(directory):
    generate.check();binding=generate.source_binding()
    arms={}
    for name in ('old','vector_pack','vector_pack_and_init'):
        path=directory/name;report=json.loads((path/'report.json').read_text())
        completion=json.loads((path/'completion.json').read_text());control=json.loads((directory/(name+'-control.json')).read_text())
        if report['arm']!=name:raise ValueError('arm directory/report identity differs')
        if (report['status']!='passed' or report['failures'] or completion['failures'] or
            completion['status']!='device_complete_outputs_saved' or report['outputs']!=completion['outputs'] or
            control['child_returncode']!=0 or control['supervisor_returncode']!=0 or control['timed_out'] or
            control['interrupt_signal'] is not None or control['cleanup_complete'] is not True or control['timeout_seconds']!=180):
            raise ValueError(name+' lacks complete 180s numeric/cleanup evidence')
        for field,record in report['outputs'].items():
            data=(path/record['path']).read_bytes()
            if len(data)!=record['bytes'] or sha(data)!=record['sha256']:raise ValueError('saved output changed')
            guarded=(path/(field+'.guarded.bin')).read_bytes()
            if sha(guarded)!=record['guarded_sha256'] or guarded[:64]!=b'\xa5'*64 or guarded[-64:]!=b'\xa5'*64:
                raise ValueError('guard evidence differs')
            expected=(report['input_sha256'][field[:-6]] if field.endswith('_after') else report['oracle_sha256'][field])
            if sha(data)!=expected:raise ValueError('oracle/input changed')
        arms[name]=report
    baseline=arms['old']
    # Existing measured reports bind A to the original production SHA; new
    # replays compile the frozen include-relocated A. Never rename a new
    # production writer into "old" or mix legacy and new source bindings.
    frozen_reports='source_binding' in baseline
    source_names={'vector_pack':'writer_pack.cpp','vector_pack_and_init':'writer_pack_init.cpp'}
    for report in arms.values():
        if ('source_binding' in report)!=frozen_reports:
            raise ValueError('mixed legacy and frozen build evidence')
        if frozen_reports and (report['source_binding']!=binding or not report.get('build_manifest_sha256')):
            raise ValueError('frozen source identity differs')
        wanted=(generate.FROZEN_SHA if frozen_reports else generate.SOURCE_SHA) if report['arm']=='old' else binding['research_sources'][source_names[report['arm']]]
        if report['kernel_sha256']!=wanted:raise ValueError('arm source differs from measured frozen writer')
        if set(report['helpers'])!={'csrc/experiments/striped_store_simd_pack/pack.h',
                                    'csrc/experiments/hadamard_indices/vector_indices.h'}:
            raise ValueError('measured helper evidence is incomplete')
        for path,digest in report['helpers'].items():
            expected=(binding['research_sources']['pack.h'] if path=='csrc/experiments/striped_store_simd_pack/pack.h' else binding['dependencies'].get(path))
            if digest!=expected:raise ValueError('measured helper differs: '+path)
        for key in ('case','geometry','grid','library_sha256','helpers','fixture_sha256','input_sha256','initial_sha256','oracle_sha256','outputs','simulator_configuration'):
            if report[key]!=baseline[key]:raise ValueError('unpaired '+key)
        if frozen_reports and report['build_manifest_sha256']!=baseline['build_manifest_sha256']:
            raise ValueError('unpaired build source manifest')
        task=report['profiles']['task']
        if (len(task)!=1 or report['task_ticks']!=task[0]['duration'] or report['task_ticks']<=0 or
                task[0]['duration']!=task[0]['end_cycle']-task[0]['start_cycle']):raise ValueError('invalid task ticks')
        actors=report['profiles']['aiv'];identity=(task[0]['task_id'],task[0]['stream_id'])
        if len(actors)!=report['grid'] or {r['core_id'] for r in actors}!=set(range(report['grid'])) or any(
            (r['task_id'],r['stream_id'])!=identity or r['duration']<=0 or r['duration']!=r['end_cycle']-r['start_cycle'] for r in actors):
            raise ValueError('incomplete actor profile')
    return {'status':'complete_matched_writer_A_B_C','case':baseline['case'],
            'source_evidence':'frozen_build_manifest' if frozen_reports else 'original_measured_source_and_helper_hashes',
            'frozen_source_binding':binding,
            'ticks':{k:v['task_ticks'] for k,v in arms.items()},
            'speedup_vs_old':{k:baseline['task_ticks']/v['task_ticks'] for k,v in arms.items()},
            'scope':'full writer only; simulated ticks are not NPU ms or whole-service proof',
            'NPU':'not_run','graph':'not_run','whole_service':'not_run','promotion_allowed':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=compare(a.directory);save_json(a.output,result);print(json.dumps(result,indent=2))
