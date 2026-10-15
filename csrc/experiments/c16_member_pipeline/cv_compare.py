"""Archive #13-22/#94/#126/#145/#154/#155: complete same-input C16 parity gate."""
import argparse
import hashlib
import json
from pathlib import Path
import posixpath
import shlex
import struct

def sha(data):return hashlib.sha256(data).hexdigest()

def cpu_command(host):
    """Parse recorded argv only; never execute the recorded shell program."""
    command=host['command']
    if not isinstance(command,list) or not all(isinstance(x,str) and x and '\x00' not in x for x in command):
        raise ValueError('invalid recorded CPU command')
    if len(command)==3 and posixpath.basename(command[0])=='bash' and command[1]=='-lc':
        script=command[2]
        if any(x in script for x in ('$','`','\n','\r')):
            raise ValueError('CPU shell substitution/newline is not allowed')
        lexer=shlex.shlex(script,posix=True,punctuation_chars=';&|()<>')
        lexer.whitespace_split=True;lexer.commenters=''
        parts=[[]]
        for token in lexer:
            if token==';':parts.append([])
            elif any(char in token for char in '&|()<>'):
                raise ValueError('unsupported recorded CPU shell operation')
            else:parts[-1].append(token)
        for prefix in parts[:-1]:
            allowed=(prefix==['set','-e'] or prefix==['unset','LD_PRELOAD'] or
                (len(prefix)==2 and prefix[0]=='source' and prefix[1].startswith('/') and
                 posixpath.basename(prefix[1])=='set_env.sh'))
            if not allowed:raise ValueError('unrecognized CPU shell preamble')
        if len(parts[-1])!=5 or parts[-1][0]!='exec':
            raise ValueError('recorded CPU shell must end with exec EXE FIXTURE OUTPUT ARM')
        command=parts[-1][1:]
    if len(command)!=4 or not posixpath.basename(command[0]).startswith('oscar_c16_') or not command[0].endswith('_cpu'):
        raise ValueError('recorded CPU argv must be EXE FIXTURE OUTPUT ARM')
    cwd=host['cwd']
    if not isinstance(cwd,str) or not cwd.startswith('/'):
        raise ValueError('recorded CPU cwd must be absolute')
    normalize=lambda value:posixpath.normpath(posixpath.join(cwd,value))
    return {'executable':normalize(command[0]),'fixture':normalize(command[1]),
        'output':normalize(command[2]),'arm':command[3]}


def cpu_provenance(path,host,report,arm):
    prepared=path/'prepared';metadata=json.loads((prepared/'metadata.json').read_text())
    expected={'status':'metadata_only','backend':'real_ascendc_cpu_prepare',
        'tokens':672,'context':report['context'],'task_rows':2016,
        'source2_suppressed':True,'positions_checked':True,'guards':True,
        'input_immutable':True,'attention_execution':'not_established'}
    if any(type(metadata.get(k)) is not type(v) or metadata[k]!=v for k,v in expected.items()):
        raise ValueError('CPU prepared metadata receipt mismatch')
    tasks=(path/'tasks.bin').read_bytes();prepared_tasks=(prepared/'tasks.bin').read_bytes()
    positions=(prepared/'positions.bin').read_bytes()
    if len(tasks)!=2016*128 or prepared_tasks!=tasks:
        raise ValueError('CPU prepared tasks differ from completed tasks')
    if len(positions)!=672*8 or struct.unpack('<672q',positions)!=tuple(range(report['context'],report['context']+672)):
        raise ValueError('CPU prepared position oracle mismatch')
    recorded=cpu_command(host)
    if recorded['arm']!=arm or posixpath.basename(recorded['output'])!=path.name:
        raise ValueError('CPU recorded arm/output mismatch')
    # These owned runs place logs beside OUTPUT as OUTPUT-logs. This binds
    # the parsed output to the supervisor receipt, not a copied path guess.
    if any(posixpath.dirname(host[field])!=recorded['output']+'-logs' for field in ('stdout_path','stderr_path')):
        raise ValueError('CPU recorded output/log ownership mismatch')
    # Historical receipts lack fixture/bad fields: source them from the
    # actual recorded exec argv and completed report, never amend old JSON.
    if 'fixture' in metadata:
        value=metadata['fixture']
        if not isinstance(value,str) or posixpath.normpath(posixpath.join(host['cwd'],value))!=recorded['fixture']:
            raise ValueError('CPU metadata/command fixture mismatch')
    if 'bad_position' in metadata and metadata['bad_position']!=report['bad_position']:
        raise ValueError('CPU metadata fault mismatch')
    if report['oracle']!=('numeric' if report['bad_position']<0 else 'invalid_metadata_status'):
        raise ValueError('CPU completed oracle scope mismatch')
    return recorded,{'tasks':sha(tasks),'prepared_positions':sha(positions)}


def load(path,control,arm):
    host=json.loads(control.read_text());report=json.loads((path/'report.json').read_text())
    if (host['child_returncode']!=0 or host['timed_out'] or host['interrupt_signal'] is not None or
            host['cleanup_complete'] is not True or report['status']!='passed' or
            report.get('variant',report.get('arm'))!=arm or report['context'] not in (337,1341)):raise ValueError('C16 arm did not complete')
    hashes={}
    for name in ('partial','lse','status','cluster_stats'):
        data=(path/(name+'.bin')).read_bytes();hashes[name]=sha(data)
        expected={'partial':672*6*3*256*4,'lse':672*6*3*4,'status':2016*2*4,'cluster_stats':20*64}[name]
        if len(data)!=expected:raise ValueError('incomplete '+name)
        if 'outputs' in report and report['outputs'][name]['sha256']!=hashes[name]:raise ValueError('saved output changed')
    if 'outputs' in report:
        complete=json.loads((path/'completion.json').read_text())
        if report['failures'] or complete['status']!='device_complete_outputs_saved' or complete['failures']:raise ValueError('C16 device/numeric failed')
        if any(complete['outputs'][name]!=report['outputs'][name] for name in report['outputs']):raise ValueError('completion output changed')
    else:
        if not all(report[name] is True for name in ('guards','input_immutable','all_status_lanes')):raise ValueError('CPU ownership/status evidence missing')
        report['_cpu_provenance'],prepared_hashes=cpu_provenance(path,host,report,arm)
        hashes.update(prepared_hashes)
    return report,host,hashes

def compare(old_dir,new_dir,old_control,new_control):
    old,oc,oh=load(old_dir,old_control,'baseline');new,nc,nh=load(new_dir,new_control,'candidate')
    if old['context']!=new['context'] or old['bad_position']!=new['bad_position'] or oc['timeout_seconds']!=nc['timeout_seconds']:raise ValueError('fixture/deadline mismatch')
    if oh!=nh:raise ValueError('C16 partial/LSE/status/counters/tasks bit mismatch')
    if ('outputs' in old)!=('outputs' in new):raise ValueError('CPU/CAModel evidence cannot be mixed')
    result={'status':'completed_full_C16_bitwise_pair','context':old['context'],'bad_position':old['bad_position'],
        'output_sha256':oh,'scope':f"Q672/R1/C{old['context']}/S1/20core source0+1, all2016tasks; not production/NPU/graph"}
    if 'outputs' in old:
        for field in ('attrs','cores','input_sha256','oracle_sha256','library_sha256','simulator_config','source_sha256','H32_source_sha256'):
            if old[field]!=new[field]:raise ValueError('C16 pair identity mismatch '+field)
        result.update(old_ticks=old['task_ticks'],new_ticks=new['task_ticks'],old_over_new=old['task_ticks']/new['task_ticks'])
    else:
        # Equal normalized VM fixture paths, with per-arm owned output and
        # prepare receipts already checked. Direct argv and controlled bash
        # wrappers are accepted without assuming command[1] is a fixture.
        if old['_cpu_provenance']['fixture']!=new['_cpu_provenance']['fixture']:raise ValueError('CPU fixture path mismatch')
        result['CPU_provenance']={'fixture':old['_cpu_provenance']['fixture'],
            'prepared_metadata_checked':True,'prepared_tasks_equal_final':True,
            'prepared_positions_checked':True,'arm_output_log_ownership_checked':True}
        result['timing']='CPU elapsed is not a device speed metric'
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True)
    p.add_argument('--old-control',type=Path,required=True);p.add_argument('--new-control',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:r=compare(a.old,a.new,a.old_control,a.new_control)
    except (OSError,KeyError,ValueError) as error:
        a.output.write_text(json.dumps({'status':'failed','reason':str(error),'old_over_new':None},indent=2)+'\n');raise
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
