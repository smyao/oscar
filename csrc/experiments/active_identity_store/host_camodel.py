"""Archive #86/#97/#111/#126/#145/#155/D.4: mac-host monotonic 600s cap for one CAModel store variant.

Only owns the fresh /tmp staging run's process group. Completion data are saved
by replay_math before cleanup and pulled after each side, never after the pair.
"""
import argparse,json,os,shlex,subprocess,sys,threading,time,signal
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--variant',choices=('old','active','identity'),required=True)
p.add_argument('--case',required=True)
p.add_argument('--grid',type=int,choices=(4,16),default=16)
a=p.parse_args()
vmroot='/tmp/active-identity-store-20261006'
rel=f'reports/active_identity_store_camodel_20261006/{a.case}-g{a.grid}-{a.variant}'
guestout=vmroot+'/'+rel
local=Path(rel);local.parent.mkdir(parents=True,exist_ok=True)
if local.exists():raise SystemExit('Refuse repeat: local side directory already exists')
logpath=local.parent/f'{a.case}-g{a.grid}-{a.variant}-host.log'
sdk='/usr/local/Ascend/cann-9.1.0';sim=sdk+'/aarch64-linux/simulator/dav_2201/lib'
args=['python3','csrc/experiments/active_identity_store/replay_store.py','--root',vmroot,
      '--variant',a.variant,'--case',a.case,'--grid',str(a.grid),'--output',guestout]
shell=('set -eo pipefail; source '+shlex.quote(sdk+'/set_env.sh')+'; set -u; '
       +'export LD_LIBRARY_PATH='+shlex.quote(sim)+':"${LD_LIBRARY_PATH}"; '
       +'export LD_PRELOAD='+shlex.quote(sim+'/libruntime_camodel.so')+'; '
       +'exec setsid '+shlex.join(args))
cmd=['limactl','shell','--workdir',vmroot,'oscar','--','bash','-lc',shell]
interrupted=False
def on_interrupt(signum,frame):
    global interrupted
    interrupted=True
signal.signal(signal.SIGINT,on_interrupt);signal.signal(signal.SIGTERM,on_interrupt)
t0=time.monotonic();deadline=t0+600
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
            print('HOST_INTERRUPTED' if interrupted else 'HOST_DEADLINE',a.variant,'limit_seconds=600',flush=True)
            code='''import json,os,signal,time
from pathlib import Path
root=Path(%r); owner=json.loads((root/'owner.json').read_text())
pid,pgid=owner['pid'],owner['pgid']
cmdline=Path('/proc/%%d/cmdline'%%pid).read_bytes().replace(b'\\0',b' ').decode()
assert 'replay_store.py' in cmdline and str(root) in cmdline and pgid==pid and os.getpgid(pid)==pgid
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
archive=local.parent/f'{a.case}-g{a.grid}-{a.variant}.tar'
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
status={'variant':a.variant,'case':a.case,'grid':a.grid,'hard_limit_seconds':600,'host_monotonic_seconds':seconds,
    'process_rc':process.returncode,'timed_out':timed_out,'interrupted':interrupted,'guest_cleanup':cleanup,'fetch_rc':fetch_rc,'extract_rc':extract_rc,'fetch_error':fetch_error,
    'host_log':str(logpath),'archive':str(archive)}
(local.parent/f'{a.case}-g{a.grid}-{a.variant}-host.json').write_text(json.dumps(status,indent=2)+'\n')
print('HOST_RESULT',json.dumps(status),flush=True)
raise SystemExit(130 if interrupted else 124 if timed_out else (process.returncode or (2 if fetch_rc!=0 or extract_rc!=0 else 0)))
