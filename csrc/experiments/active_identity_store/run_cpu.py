"""Archive #111/#126/#145/#155/D.4: bounded official CPU byte-oracle suite.

Every case runs old/active/explicit-I kernels at the production row-derived grid. Math/checks stay enabled;
owned tmpfs instruction traces are snapshotted and cleared after every case.
"""
import argparse,json,os,signal,subprocess,time,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--only-prefix');p.add_argument('--case-list');p.add_argument('--label',default='suite');a=p.parse_args()
root=Path.cwd();manifest=json.loads((root/'artifacts/active_identity_store_20261006/cases.json').read_text())
output=root/'reports/active_identity_store_cpu_final_20261006';output.mkdir(parents=True,exist_ok=True)
selected=manifest['cases']
if a.only_prefix:selected=[c for c in selected if c['name'].startswith(a.only_prefix)]
if a.case_list:
 names=set(a.case_list.split(','));selected=[c for c in selected if c['name'] in names]
results=[]
for case in selected:
 name=case['name'];log=output/(name+'.log');start=time.monotonic();process=None;rc=None;helper_errors=[]
 command=['timeout','--signal=TERM','--kill-after=5s','120s',str(root/'build/cpu-binding/oscar_active_identity_store_cpu'),str(root/case['path'])]
 case_work=Path('/dev/shm/oscar-active-identity-cpu-20261006')/name
 case_work.mkdir(parents=True,exist_ok=False)
 try:
  process=subprocess.Popen(command,cwd=case_work,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                           text=True,bufsize=1,start_new_session=True)
  with log.open('w') as dest:
   for line in process.stdout:print(line,end='',flush=True);dest.write(line);dest.flush()
  rc=process.wait()
 finally:
  if process is not None and process.poll() is None:
   try:os.killpg(process.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:process.wait(timeout=5)
   except subprocess.TimeoutExpired:
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    process.wait()
   rc=process.returncode
  try:
   snapshots=output/(name+'_trace');snapshots.mkdir(exist_ok=True);trace_files=[]
   for trace in case_work.rglob('*'):
    if not trace.is_file():continue
    size=trace.stat().st_size
    with trace.open('rb') as src:
     head=src.read(16384)
     if size>16384:src.seek(max(16384,size-16384));tail=src.read(16384)
     else:tail=b''
    target=snapshots/(str(trace.relative_to(case_work)).replace('/','__')+'.snapshot')
    target.write_bytes(head+b'\n[bounded trace head/tail; middle omitted]\n'+tail)
    trace_files.append({'path':str(trace.relative_to(case_work)),'original_bytes':size,'snapshot':str(target)})
   (snapshots/'manifest.json').write_text(json.dumps(trace_files,indent=2)+'\n')
  except Exception as error:
   helper_errors.append('trace snapshot: '+repr(error));print('STORE_TRACE_ERROR',repr(error),flush=True)
  finally:
   shutil.rmtree(case_work)
 entry={'case':name,'rc':rc,'seconds':time.monotonic()-start,'log':str(log),
        'variants':(2 if case['shape_fields']['hadamard'] else 3) if rc==0 else None,'helper_errors':helper_errors}
 results.append(entry);print('STORE_CASE_RC',json.dumps(entry),flush=True)
 (output/(a.label+'.json')).write_text(json.dumps({'selected_cases':len(selected),'completed':results,'cpu_only':True},indent=2)+'\n')
 if rc or helper_errors:raise SystemExit(rc or 2)
print('STORE_SUITE_PASSED',len(results),'cases',sum(x['variants'] for x in results),'kernel comparisons',flush=True)
