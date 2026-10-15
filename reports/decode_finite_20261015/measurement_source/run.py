"""Archive #94/#129/#140/#145/#148/#154: bound complete family comparison.

Reuse the complete checked replay; remove its unsafe FD thread by the existing
strict inverse logging adapter and require official zero-dump LOG mask. Keep
full outputs, real prepare task ABI, independent oracle, guards and all actors.
No incomplete trace is interpreted; CPU small and CAModel long are separate.
"""
from __future__ import annotations
import argparse,json,os,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.decode_finite_20261015 import generate
from csrc.experiments.c16_l1_reuse_20261015.safe_logging_runner import transform
from csrc.experiments.decode_task_overlap.run_pair import checked as numeric_checked
sha=generate.sha;digest=generate.digest
TEMPLATE=ROOT/'csrc/experiments/window_batch_decode/replay_camodel.py'
TEMPLATE_SHA='2b435e186ad6aa8112d620fa83299cacc7c9b7a60d02f77b59a1b43e9447883c'
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')

def source_check(library):
    for family in ('q4','q1'):
        for arm in ('baseline','candidate'):
            text,_=generate.generate(family,arm)
            if text!=(HERE/f'{family}_{arm}.cpp').read_text():raise ValueError('source changed')
    binding=json.loads((library.parent.parent/'source_binding.json').read_text())
    if binding['manifest_sha256']!=digest(HERE/'manifest.json') or binding['generator_sha256']!=digest(generate.__file__):raise ValueError('compiled source binding stale')
    for name,pin in binding['sources'].items():
        if digest(HERE/name)!=pin:raise ValueError('compiled source drift '+name)
    return binding

def adapted():
    original=TEMPLATE.read_text()
    if sha(original)!=TEMPLATE_SHA:raise ValueError('qualified replay changed')
    text=transform(original)
    old="'rv':'rv.bin'";new="'rv':('rv_nz.bin' if mode=='q4' else 'rv.bin')"
    if text.count(old)!=1:raise ValueError('replay Rv anchor')
    text=text.replace(old,new)
    if text.replace(new,old)!=transform(original):raise ValueError('Rv adapter inverse')
    if 'os.dup2' in text or '/proc/self/fd' in text:raise ValueError('unsafe FD mutation')
    return text

def geometry(r,tasks):
    family=r['mode'];q=4 if family=='q4' else 1;a=r['attrs']
    if r['case']!=family+'_finite_long' or r['cores']!=1 or a[:5]!=[8,6,1,256,2] or a[6]!=24 or a[-1]!=1:
        raise ValueError('frozen full geometry changed')
    if len(tasks)!=24*128 or sha(tasks)!=r['input_sha256']['tasks']:raise ValueError('task witness drift')
    rows=list(struct.iter_unpack('<16q',tasks))
    # Validate native prepare ownership and preserve ALL padding/followers.
    for token in range(8):
        for source in range(3):
            row=rows[token*3+source]
            if row[0]!=token or row[2]!=0 or row[6]!=source or row[7]!=source or row[10]:raise ValueError('task identity/error changed')
            if token>=2*q:
                if row[1]!=-1 or any(row[i] for i in (3,4,5,8,9)):raise ValueError('padding became live')
            elif token%q:
                if row[1]!=0:raise ValueError('follower count changed')
            else:
                request=token//q;ctx=(1344,1376)[request]
                bounds=((64,ctx+q-256),(0,319),(ctx,ctx+q))[source]
                if (row[1],row[3],row[4],row[5],row[8],row[9])!=(q,*bounds,request,ctx,request*q):raise ValueError('full source domain changed')

def run(a):
    a.library=a.library.resolve();a.output=a.output.resolve();binding=source_check(a.library)
    config=a.simulator_config_dir.resolve();detail=json.loads((config/'config.json').read_text())
    if detail['LOG']['core_enable_mask']!=['0x0'] or os.environ.get('CAMODEL_CONFIG_PATH')!=str(config):raise ValueError('official mask0 config required')
    code=adapted();case=a.family+'_finite_long';plan=a.output.with_name(a.output.name+'.plan.json')
    if plan.exists() or a.output.exists():raise ValueError('fresh arm required')
    plan.parent.mkdir(parents=True,exist_ok=True)
    save(plan,{'family':a.family,'arm':a.arm,'source_binding':binding,'runner_sha256':digest(__file__),
        'library_sha256':digest(a.library),'template_sha256':TEMPLATE_SHA,'adapted_sha256':sha(code),
        'deadline_s':600,'descriptor_mutation':False,'CPU_reference':'small complete cases required separately; long prepare-only by predeclared budget',
        'numeric_oracle_guard_profile_gates':'unchanged complete qualified runner'})
    previous=sys.argv
    sys.argv=[str(HERE/'run.py'),'--root',str(ROOT),'--output',str(a.output),'--case',case,
      '--variant',a.arm,'--fixture-root',str(a.fixture_root.resolve()),'--export-root',str(a.export_root.resolve()),
      '--library',str(a.library),'--simulator-config-dir',str(config),
      '--baseline-q4','oscar_decode_finite_q4_baseline_kernel','--candidate-q4','oscar_decode_finite_q4_candidate_kernel',
      '--baseline-q1','oscar_decode_finite_q1_baseline_q1_kernel','--candidate-q1','oscar_decode_finite_q1_candidate_q1_kernel',
      '--baseline-workspace-bytes-per-core','1146880','--candidate-workspace-bytes-per-core','1146880',
      '--cpu-reference','optional','--instruction-budget-bytes','0']
    try:
        ns={'__file__':str(HERE/'run.py'),'__name__':'decode_finite_safe_replay'}
        exec(compile(code,str(TEMPLATE),'exec'),ns);rc=ns['main']()
        if rc:raise ValueError('full replay failed')
        r=json.loads((a.output/'report.json').read_text());tasks=(a.export_root/case/'tasks.bin').read_bytes()
        geometry(r,tasks);(a.output/'task_witness.bin').write_bytes(tasks)
        r['finite_binding']={'plan_sha256':digest(plan),'all_sources_tasks_verified':True}
        save(a.output/'report.json',r)
    finally:sys.argv=previous

def checked(path,control,arm):
    r=numeric_checked(path,control);c=json.loads(control.read_text());p=path.with_name(path.name+'.plan.json');binding=json.loads(p.read_text())
    geometry(r,(path/'task_witness.bin').read_bytes())
    if c['timeout_seconds']!=600 or binding['arm']!=arm or binding['family']!=r['mode'] or binding['library_sha256']!=r['library_sha256'] or r['finite_binding']['plan_sha256']!=digest(p):raise ValueError('arm/control/source binding')
    if r['dump_files_redirected'] or r['profile_counters'] or r['simulator_configuration']['detail']['LOG']['core_enable_mask']!=['0x0']:raise ValueError('profile/logging changed')
    expected='oscar_decode_finite_'+r['mode']+'_'+arm+('_q1' if r['mode']=='q1' else '')+'_kernel'
    if r['kernel']!=expected:raise ValueError('wrong kernel symbol')
    for kind,count in (('task',1),('aic',1),('aiv',2)):
        rows=r['profiles'][kind]
        if len(rows)!=count or any(row['duration']<=0 or row['end_cycle']-row['start_cycle']!=row['duration'] for row in rows):raise ValueError('incomplete profile')
    return r

def compare(a):
    old=checked(a.baseline,a.baseline_control,'baseline');new=checked(a.candidate,a.candidate_control,'candidate')
    for key in ('case','mode','cores','attrs','scale','input_sha256','oracle_sha256','cpu_baseline_sha256','library_sha256','simulator_configuration','cpu_reference','cpu_reference_policy','workspace_bytes_per_core','trace_policy'):
        if old[key]!=new[key]:raise ValueError('unpaired '+key)
    for key in ('partial','lse','status'):
        if old['outputs'][key]['sha256']!=new['outputs'][key]['sha256']:raise ValueError('changed bits '+key)
    ratio=old['task_ticks']/new['task_ticks']
    result={'status':'completed_same_library_pair','family':old['mode'],'task_ticks':{'baseline':old['task_ticks'],'candidate':new['task_ticks']},
      'ratio':ratio,'positive_net_gain':ratio>1,'all_partial_lse_status_bitwise':True,'independent_oracle':'passed_both',
      'all_sources_and_tasks':True,'library_sha256':old['library_sha256'],'profiles':'complete 1 task + 1 AIC + 2 AIV each arm',
      'instruction_trace':'not_requested_official_mask0','NPU':'not_run','graph':'not_run','E2E':'not_established'}
    save(a.output,result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run')
    for name in ('output','fixture-root','export-root','library','simulator-config-dir'):r.add_argument('--'+name,type=Path,required=True)
    r.add_argument('--family',choices=('q4','q1'),required=True);r.add_argument('--arm',choices=('baseline','candidate'),required=True)
    r=sub.add_parser('compare')
    for name in ('baseline','candidate','baseline-control','candidate-control','output'):r.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a) if a.command=='run' else compare(a)
