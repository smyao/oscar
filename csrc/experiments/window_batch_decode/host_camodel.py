"""Archive #126/#129/#140/#145/#148/#155/D.4: mac-host monotonic 600s cap for one complete window CAModel arm.

Only owns the fresh /tmp staging run's process group. Completion data are saved
by replay_camodel before cleanup and pulled after each side, never after the pair.
"""
import argparse,json,os,shlex,subprocess,sys,threading,time,signal
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--variant',choices=('baseline','candidate'),required=True)
p.add_argument('--case',required=True)
p.add_argument('--library')
p.add_argument('--candidate-q4',default='oscar_attention_cv_window_batch_decode_kernel')
p.add_argument('--candidate-q1',default='oscar_attention_cv_window_batch_q1_kernel')
p.add_argument('--candidate-workspace-bytes-per-core',type=int,default=917504)
p.add_argument('--baseline-workspace-bytes-per-core',type=int,default=917504)
p.add_argument('--run-label',default='window_batch_decode_camodel_20261006')
p.add_argument('--instruction-budget-bytes',type=int,default=0)
p.add_argument('--fixture-root',type=Path,default=Path('artifacts/window_batch_decode'),
               help='project-relative fixture path shared by host and VM staging tree')
p.add_argument('--export-root',type=Path,default=Path('artifacts/window_batch_decode_cpu_export'),
               help='fixture export path relative to --vm-root, or absolute in VM')
p.add_argument('--vm-root',default='/tmp/window-batch-decode-20261006')
p.add_argument('--profile-counters',action='store_true')
p.add_argument('--require-phase-markers',action='store_true')
p.add_argument('--baseline-q4',default='oscar_attention_cv_striped_decode_simd_kernel')
p.add_argument('--baseline-q1',default='oscar_attention_cv_striped_q1_simd_kernel')
p.add_argument('--cpu-reference',choices=('required','optional'),default='required')
p.add_argument('--simulator-config-dir',help='explicit independent VM CAModel config directory; never changes SDK')
p.add_argument('--timeout-seconds',type=int,default=600,
               help='host simulation budget fixed before this arm; not a device-performance threshold')
a=p.parse_args()
if not 1<=a.timeout_seconds<=21600:raise SystemExit('host simulation budget must be1..21600 seconds')
project_root=Path(__file__).resolve().parents[3]
local_fixture=a.fixture_root if a.fixture_root.is_absolute() else project_root/a.fixture_root
fixture_manifest=json.loads((local_fixture/'cases.json').read_text())
selected=[c for c in fixture_manifest['cases'] if c['name']==a.case]
if len(selected)!=1 or selected[0]['expected_error']:raise SystemExit('requires a valid registered full-source case')
cores=selected[0]['cores']
if Path(a.run_label).name!=a.run_label or a.run_label in ('.','..'):
    raise SystemExit('run-label must be one report directory name')
vmroot=os.path.normpath(a.vm_root)
if not os.path.isabs(vmroot):raise SystemExit('vm-root must be an absolute VM path')
rel=f'reports/{a.run_label}/{a.case}-{a.variant}'
guestout=vmroot+'/'+rel
local=Path(rel);local.parent.mkdir(parents=True,exist_ok=True)
if local.exists():raise SystemExit('Refuse repeat: local side directory already exists')
logpath=local.parent/f'{a.case}-{a.variant}-host.log'
sdk='/usr/local/Ascend/cann-9.1.0';sim=sdk+'/aarch64-linux/simulator/dav_2201/lib'
args=['python3','csrc/experiments/window_batch_decode/replay_camodel.py','--root',vmroot,
      '--variant',a.variant,'--case',a.case,'--output',guestout]
try:fixture_relative=local_fixture.resolve().relative_to(project_root)
except ValueError:raise SystemExit('host fixture-root must be inside this project so its relative VM path is unambiguous')
args+=['--fixture-root',str(fixture_relative),'--export-root',str(a.export_root)]
args+=['--candidate-q4',a.candidate_q4,'--candidate-q1',a.candidate_q1,
       '--candidate-workspace-bytes-per-core',str(a.candidate_workspace_bytes_per_core),
       '--baseline-workspace-bytes-per-core',str(a.baseline_workspace_bytes_per_core),
       '--instruction-budget-bytes',str(a.instruction_budget_bytes)]
if a.library:args+=['--library',a.library]
if a.profile_counters:args+=['--profile-counters']
if a.require_phase_markers:args+=['--require-phase-markers']
args+=['--baseline-q4',a.baseline_q4,'--baseline-q1',a.baseline_q1]
args+=['--cpu-reference',a.cpu_reference]
if a.simulator_config_dir:
    if not os.path.isabs(a.simulator_config_dir):raise SystemExit('simulator-config-dir must be absolute in VM')
    args+=['--simulator-config-dir',a.simulator_config_dir]
config_shell=('export CAMODEL_CONFIG_PATH='+shlex.quote(a.simulator_config_dir)+'; '
              if a.simulator_config_dir else 'unset CAMODEL_CONFIG_PATH; ')
shell=('set -eo pipefail; source '+shlex.quote(sdk+'/set_env.sh')+'; set -u; '
       +config_shell
       +'export LD_LIBRARY_PATH='+shlex.quote(sim)+':"${LD_LIBRARY_PATH}"; '
       +'export LD_PRELOAD='+shlex.quote(sim+'/libruntime_camodel.so')+'; '
       +'exec setsid flock --no-fork --exclusive --nonblock /tmp/oscar-cann-validation.lock '+shlex.join(args))
cmd=['limactl','shell','--workdir',vmroot,'oscar','--','bash','-lc',shell]
interrupted=False
def on_interrupt(signum,frame):
    global interrupted
    interrupted=True
signal.signal(signal.SIGINT,on_interrupt);signal.signal(signal.SIGTERM,on_interrupt)
t0=time.monotonic();deadline=t0+a.timeout_seconds
process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=0)
with logpath.open('wb') as logfile:
    def drain():
        while True:
            chunk=process.stdout.read(65536)
            if not chunk:break
            logfile.write(chunk);logfile.flush();sys.stdout.buffer.write(chunk);sys.stdout.buffer.flush()
    reader=threading.Thread(target=drain,daemon=True);reader.start()
    next_heartbeat=t0+30;timed_out=False;cleanup=None
    while process.poll() is None:
        now=time.monotonic()
        if now>=deadline or interrupted:
            timed_out=not interrupted
            print('HOST_INTERRUPTED' if interrupted else 'HOST_DEADLINE',a.variant,'limit_seconds='+str(a.timeout_seconds),flush=True)
            code='''import json,os,signal,time
from pathlib import Path
root=Path(%r); owner=json.loads((root/'owner.json').read_text())
pid,pgid=owner['pid'],owner['pgid']
cmdline=Path('/proc/%%d/cmdline'%%pid).read_bytes().replace(b'\\0',b' ').decode()
assert 'replay_camodel.py' in cmdline and str(root) in cmdline and pgid==pid and os.getpgid(pid)==pgid
os.killpg(pgid,signal.SIGTERM);time.sleep(2)
try:os.killpg(pgid,signal.SIGKILL)
except ProcessLookupError:pass
print('OWNED_GROUP_TERMINATED',pgid)
'''%guestout
            try:
                result=subprocess.run(['limactl','shell','--workdir',vmroot,'oscar','--','python3','-c',code],
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=12,text=True)
                cleanup={'rc':result.returncode,'output':result.stdout};print(result.stdout,flush=True)
            except subprocess.TimeoutExpired:cleanup={'status':'cleanup_command_timeout'}
            if process.poll() is None:process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            break
        if now>=next_heartbeat:
            print('HOST_HEARTBEAT',a.variant,'elapsed_s',round(now-t0,1),'remaining_s',round(deadline-now,1),flush=True)
            next_heartbeat=now+30
        time.sleep(.25)
    reader.join(timeout=5)
seconds=time.monotonic()-t0
archive=local.parent/f'{a.case}-{a.variant}.tar'
fetch_error=None;fetch_rc=None;extract_rc=None
try:
    with archive.open('wb') as target:
        fetch=subprocess.run(['limactl','shell','--workdir',vmroot,'oscar','--','tar','--exclude=*.dump','-cf','-',rel],
            stdout=target,stderr=subprocess.PIPE,timeout=30)
    fetch_rc=fetch.returncode
    if fetch_rc==0:extract_rc=subprocess.run(['tar','-xf',str(archive)]).returncode
    else:fetch_error=fetch.stderr.decode(errors='replace')
except subprocess.TimeoutExpired:
    fetch_rc=124;fetch_error='artifact fetch exceeded 30 seconds'
status={'variant':a.variant,'case':a.case,'cores':cores,'hard_limit_seconds':a.timeout_seconds,'host_monotonic_seconds':seconds,
    'process_rc':process.returncode,'timed_out':timed_out,'interrupted':interrupted,'guest_cleanup':cleanup,'fetch_rc':fetch_rc,'extract_rc':extract_rc,'fetch_error':fetch_error,
    'host_log':str(logpath),'archive':str(archive)}
evidence_error=None
if not timed_out and not interrupted and process.returncode==0 and fetch_rc==0 and extract_rc==0:
    try:
        report=json.loads((local/'report.json').read_text())
        complete=json.loads((local/'completion.json').read_text())
        if (report.get('status')!='passed' or report.get('failures') or
                complete.get('status')!='device_complete_outputs_saved'):
            raise ValueError('device completion/numerical report did not pass')
    except (OSError,ValueError) as error:
        evidence_error=str(error)
status['evidence_error']=evidence_error
status['evidence_complete']=evidence_error is None and not timed_out and not interrupted and process.returncode==0 and fetch_rc==0 and extract_rc==0
(local.parent/f'{a.case}-{a.variant}-host.json').write_text(json.dumps(status,indent=2)+'\n')
print('HOST_RESULT',json.dumps(status),flush=True)
raise SystemExit(130 if interrupted else 124 if timed_out else (process.returncode or (2 if fetch_rc!=0 or extract_rc!=0 or evidence_error else 0)))
