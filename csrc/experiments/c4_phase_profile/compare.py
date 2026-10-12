"""Archive #129/#132-135/#145/#148/#154/#155: completed C4 arm comparison.

Independent schema3 gate. Never accept a timeout, missing output, schema2
decode report, different source/inputs/config or omitted current suppression.
"""
import argparse
import json
import struct
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_phase_profile import fixture,generate,markers
from csrc.experiments.window_batch_decode.replay_camodel import sha,save_json


def checked_arm(directory,control_path,arm):
    control=json.loads(control_path.read_text())
    if (control.get('status')!='completed' or control.get('child_returncode')!=0 or
            control.get('supervisor_returncode')!=0 or control.get('timed_out') is not False or
            control.get('interrupt_signal') is not None or control.get('cleanup_complete') is not True or
            control.get('timeout_seconds')!=900):
        raise ValueError(arm+' durable supervisor/deadline gate failed')
    owner=json.loads((directory/'owner.json').read_text());command=control.get('command',[])
    if (owner.get('pid')!=control.get('child_pid') or owner.get('pgid')!=control.get('child_pgid') or
            '--arm' not in command or command[command.index('--arm')+1]!=arm or
            '--output' not in command or command[command.index('--output')+1]!=owner.get('output')):
        raise ValueError(arm+' supervisor belongs to a different output/arm/process')
    report=json.loads((directory/'report.json').read_text())
    n=fixture.declared_tokens(report['attrs'][0])
    completed=json.loads((directory/'completion.json').read_text())
    if (report.get('status')!='passed' or report.get('failures')!=[] or
            completed.get('status')!='device_complete_outputs_saved' or completed.get('failures')!=[]):
        raise ValueError(arm+' completion/numeric/profile gate failed')
    if (report.get('experiment')!='c4_phase_profile' or report.get('arm')!=arm or
            report.get('case')!=fixture.case_name(n) or report.get('kernel')!='oscar_c4_'+arm+'_kernel' or
            report.get('profile_schema')!=(3 if arm=='profile' else None) or
            report.get('instrumentation_revision')!='schema3_finite_reduce_v1' or
            report.get('generated_source_sha256')!=generate.sha(generate.generate(arm=='profile')[0]) or
            report.get('source_sha256')!=generate.PINS[generate.SOURCE] or
            report.get('dependency_sha256')!=generate.PINS):
        raise ValueError(arm+' exact C4 source/schema/symbol differs')
    if (report.get('cores')!=20 or report.get('all_prepared_tasks_retained') is not True or
            len(report['attrs'])!=17 or report['attrs'][:5]!=[n,6,1,256,1] or
            report['attrs'][6]!=3*n or report['attrs'][13:]!=[64,256,3,1] or
            report.get('task_filtering') is not False or report.get('source2_suppressed') is not True or
            report.get('oracle_scope')!=fixture.plan(n)['oracle_scope']):
        raise ValueError(arm+' partition scope differs')
    expected={'partial','lse','status','workspace','cluster_stats'}|({'profile'} if arm=='profile' else set())
    if set(report['outputs'])!=expected:raise ValueError(arm+' output set incomplete')
    for name,meta in report['outputs'].items():
        data=(directory/meta['path']).read_bytes()
        if len(data)!=meta['bytes'] or sha(data)!=meta['sha256'] or completed['outputs'][name]!=meta:
            raise ValueError(arm+' output identity differs: '+name)
    if (report['oracle']['bad_output'] or report['oracle']['bad_lse'] or
            (report['oracle']['atol'],report['oracle']['rtol'])!=(.005,.005) or
            type(report['task_ticks']) is not int or report['task_ticks']<=0):
        raise ValueError(arm+' frozen oracle/complete task cycles failed')
    tasks=(directory/'task_witness.bin').read_bytes()
    if sha(tasks)!=report['input_sha256']['tasks']:raise ValueError('saved task witness differs')
    stats=[list(row) for row in struct.iter_unpack('<8q',(directory/'cluster_stats.bin').read_bytes())]
    if stats!=markers.expected_cluster_stats(tasks):raise ValueError('real C4 cluster count/ownership differs')
    if arm=='profile':
        expected_markers=markers.expected_counts(report,tasks,(directory/'profile.bin').read_bytes())
        if report['marker_collection_requested']:
            files=report['phase_marker_files']
            if len(files)!=60 or {row['source'] for row in files}!=set(expected_markers):
                raise ValueError('complete C4 actor set missing')
            for record in files:
                item=json.loads((directory/record['path']).read_text())
                if not item['complete_for_actor']:raise ValueError('incomplete C4 actor')
                spans=markers.Intervals()
                for event in item['events']:
                    if event['field'] not in markers.MARKED_FIELDS:raise ValueError('unknown C4 marker field')
                    spans.accept(event)
                spans.finish()
                if {key:row['count'] for key,row in spans.groups.items()}!=dict(expected_markers[record['source']]):
                    raise ValueError('C4 marker closure/counts changed')
    return report,control


def compare(normal_directory,profile_directory,normal_control,profile_control):
    normal,nc=checked_arm(normal_directory,normal_control,'normal')
    profile,pc=checked_arm(profile_directory,profile_control,'profile')
    for field in ('case','cores','attrs','scale','source_sha256','dependency_sha256','library_sha256',
                  'input_sha256','oracle_sha256','acceptance_sha256','task_ledger','simulator_configuration',
                  'workspace_bytes_per_core','oracle_scope','source2_suppressed'):
        if normal[field]!=profile[field]:raise ValueError('normal/profile identity differs: '+field)
    for field in ('partial','lse','status','cluster_stats'):
        if normal['outputs'][field]['sha256']!=profile['outputs'][field]['sha256']:
            raise ValueError('normal/profile bitwise mismatch: '+field)
    old,new=normal['task_ticks'],profile['task_ticks'];delta=(new-old)/old
    return {'status':'completed_C4_instrumentation_comparison','case':normal['case'],
        'normal_ticks':old,'profile_ticks':new,'profile_over_normal':new/old,
        'perturbation_fraction':delta,'max_absolute_perturbation':.05,
        'perturbation_gate':'passed' if abs(delta)<=.05 else 'failed',
        'partial_lse_status_cluster_stats_bitwise':True,'independent_oracle':'passed_both_arms',
        'oracle_scope':normal['oracle_scope'],'source2_suppressed':True,
        'hard_deadline_seconds':900,'source_sha256':normal['source_sha256'],
        'library_sha256':normal['library_sha256'],'input_sha256':normal['input_sha256'],
        'scope':'same declared R1 prefill CV partition; excludes native current FIA; no K32/Q2048 saturation/E2E inference',
        'profile_schema':3,'npu':'not_run','graph':'not_run','whole_service':'not_run'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--normal',type=Path,required=True);p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--normal-control',type=Path,required=True);p.add_argument('--profile-control',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:result=compare(a.normal,a.profile,a.normal_control,a.profile_control)
    except (KeyError,OSError,ValueError) as error:
        save_json(a.output,{'status':'failed','reason':str(error),'profile_over_normal':None});raise
    save_json(a.output,result);print(json.dumps(result))
    if result['perturbation_gate']!='passed':raise SystemExit(2)
