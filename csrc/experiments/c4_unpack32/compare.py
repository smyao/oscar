"""Archive #13-22/#94/#126/#154/#155: completed exact helper pair only."""
import argparse
import hashlib
import json
from pathlib import Path

def load(path,control,variant):
    host=json.loads(control.read_text());report=json.loads((path/'report.json').read_text())
    complete=json.loads((path/'completion.json').read_text())
    if (host['child_returncode']!=0 or host['timed_out'] or host['interrupt_signal'] is not None or
            host['cleanup_complete'] is not True or report['status']!='passed' or report['failures'] or
            report['variant']!=variant or complete['failures']):raise ValueError('incomplete helper arm')
    for name,info in report['outputs'].items():
        data=(path/info['path']).read_bytes()
        if len(data)!=info['bytes'] or hashlib.sha256(data).hexdigest()!=info['sha256'] or info!=complete['outputs'][name]:
            raise ValueError('output identity '+name)
    return report,host

def compare(old_dir,new_dir,old_control,new_control):
    old,oc=load(old_dir,old_control,'16');new,nc=load(new_dir,new_control,'32')
    for field in ('cases','library_sha256','fixture_sha256','source_pins','helper_sha256','simulator_config'):
        if old[field]!=new[field]:raise ValueError('pair differs '+field)
    if oc['timeout_seconds']!=nc['timeout_seconds']:raise ValueError('deadline differs')
    for field in ('output','status'):
        if old['outputs'][field]['sha256']!=new['outputs'][field]['sha256']:raise ValueError('A/B bitwise '+field)
    return {'status':'completed_bitwise_helper_pair','old_ticks':old['task_ticks'],'new_ticks':new['task_ticks'],
        'old_over_new':old['task_ticks']/new['task_ticks'],'cases':old['cases'],
        'library_sha256':old['library_sha256'],'fixture_sha256':old['fixture_sha256'],
        'scope':'helper setup/loads/reconstruction/output only; not C4, NPU or end-to-end',
        'full_C4':'not_run','NPU':'not_run'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True)
    p.add_argument('--old-control',type=Path,required=True);p.add_argument('--new-control',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:r=compare(a.old,a.new,a.old_control,a.new_control)
    except (OSError,KeyError,ValueError) as error:
        a.output.write_text(json.dumps({'status':'failed','reason':str(error),'old_over_new':None},indent=2)+'\n');raise
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
