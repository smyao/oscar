"""Archive #129/#140/#148/#155/D.4: compare only two complete matched arms.

No ratio after timeout, failed download, wrong inputs, incomplete profile or
missing output bytes. All-source short-context results remain operator-only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def checked_arm(directory,case,variant):
    prefix=case+'-'+variant
    host=json.loads((directory/(prefix+'-host.json')).read_text())
    if host['process_rc']!=0 or host['timed_out'] or host.get('interrupted') or host['fetch_rc']!=0 or host['extract_rc']!=0:
        raise ValueError(variant+' host/deadline/download gate failed')
    arm=directory/prefix
    report=json.loads((arm/'report.json').read_text())
    complete=json.loads((arm/'completion.json').read_text())
    if report['status']!='passed' or report['failures'] or complete['status']!='device_complete_outputs_saved':
        raise ValueError(variant+' completion/numeric/profile gate failed')
    if not report['all_sources_and_tasks'] or report['task_filtering']:
        raise ValueError(variant+' filtered task/source scope')
    for name,meta in report['outputs'].items():
        data=(arm/meta['path']).read_bytes()
        if len(data)!=meta['bytes'] or hashlib.sha256(data).hexdigest()!=meta['sha256'] or complete['outputs'][name]!=meta:
            raise ValueError(variant+' saved '+name+' identity mismatch')
    if report['oracle']['bad_output'] or report['oracle']['bad_lse'] or not isinstance(report['task_ticks'],int) or report['task_ticks']<=0:
        raise ValueError(variant+' oracle/timing gate failed')
    return report,host


def compare(directory,case,baseline_directory=None):
    baseline_directory=directory if baseline_directory is None else baseline_directory
    baseline,bh=checked_arm(baseline_directory,case,'baseline')
    candidate,ch=checked_arm(directory,case,'candidate')
    for key in ('case','mode','cores','attrs','scale','input_sha256','oracle_sha256','cpu_baseline_sha256','library_sha256'):
        if baseline[key]!=candidate[key]:raise ValueError('A/B '+key+' differs')
    for key in ('profile_counters','phase_profile_schema','simulator_configuration'):
        if baseline.get(key)!=candidate.get(key):raise ValueError('A/B '+key+' differs')
    if baseline.get('cpu_reference','available_checked')!=candidate.get('cpu_reference','available_checked'):
        raise ValueError('A/B CPU reference availability differs')
    if bh['cores']!=baseline['cores'] or ch['cores']!=candidate['cores']:
        raise ValueError('host/kernel core budget differs')
    if bh['hard_limit_seconds']!=ch['hard_limit_seconds']:
        raise ValueError('A/B predeclared host simulation budgets differ')
    for name in ('partial','lse','status'):
        if baseline['outputs'][name]['sha256']!=candidate['outputs'][name]['sha256']:
            raise ValueError('A/B '+name+' bits differ')
    old,new=baseline['task_ticks'],candidate['task_ticks']
    return {'status':'completed_matched_comparison','case':case,'cores':baseline['cores'],
        'baseline':baseline['kernel'],'candidate':candidate['kernel'],'all_sources_and_tasks':True,
        'baseline_directory':str(baseline_directory),'candidate_directory':str(directory),
        'task_ticks':{'baseline':old,'candidate':new},'baseline_over_candidate':old/new,
        'candidate_tick_reduction':(old-new)/old,'candidate_faster':new<old,
        'partial_lse_status_bitwise':True,'independent_oracle':'passed_both_arms',
        'cpu_reference':baseline.get('cpu_reference','available_checked'),
        'input_sha256':baseline['input_sha256'],'library_sha256':baseline['library_sha256'],
        'simulator_configuration':baseline.get('simulator_configuration'),
        'host_seconds':{'baseline':bh['host_monotonic_seconds'],'candidate':ch['host_monotonic_seconds']},
        'scope':'complete operator on this exact synthetic fixture; no other-shape or whole-model extrapolation',
        'npu':'not_run','graph':'not_run','model_quality':'not_run','whole_service':'not_run'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True)
    parser.add_argument('--directory',type=Path,default=Path('reports/window_batch_decode_camodel_20261006'))
    parser.add_argument('--baseline-directory',type=Path,
        help='reuse an actually completed arm; all original library/input/profile/config gates still apply')
    args=parser.parse_args();destination=args.directory/(args.case+'-comparison.json')
    try:
        result=compare(args.directory,args.case,args.baseline_directory)
    except Exception as error:
        result={'status':'failed','case':args.case,'baseline_over_candidate':None,'reason':str(error)}
        destination.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));raise SystemExit(2)
    destination.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
