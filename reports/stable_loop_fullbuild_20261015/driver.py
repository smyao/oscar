# Archive #79-85/#94/#95/#117/#153: isolated local Lima source stage and real rc.
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

ROOT=Path('/Users/sunao2000/gpt_new_oscar_kimi')
STAGE='/tmp/oscar-stable-loop-fullbuild-20261015'
PYTHON='/home/sunao2000.linux/gpt_new_oscar/.venv/bin/python'
OUT=ROOT/'reports/stable_loop_fullbuild_20261015'
PREFIX=['limactl','shell','--workdir','/tmp','oscar','--']


def sources():
    found=set()
    for part in ('csrc/kernels','csrc/include'):
        found.update(p for p in (ROOT/part).rglob('*') if p.is_file() and not p.is_symlink()
                     and p.suffix in {'.cpp','.h','.hpp'})
    found.update(p for p in (ROOT/'csrc').iterdir() if p.is_file() and not p.is_symlink()
                 and (p.suffix in {'.cpp','.h','.hpp'} or p.name=='CMakeLists.txt'))
    todo=list(found)
    while todo:
        p=todo.pop()
        for inc in re.findall(r'^\s*#include\s*"([^"]+)"',p.read_text(),re.M):
            dep=(p.parent/inc).resolve()
            if dep.is_file() and dep.is_relative_to(ROOT/'csrc') and dep not in found:
                found.add(dep);todo.append(dep)
    found.update(p for p in (ROOT/'oscar_ascend').rglob('*.py') if not p.is_symlink())
    found.update(ROOT/p for p in ('tools/__init__.py','tools/build_ops.py','tools/environment.py',
                                  'tools/phase.py','tools/experiments/run_camodel_arm.py'))
    return sorted(found)


def manifest():
    selected=sources()
    return {'stage':STAGE,'vm':'oscar','files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in selected},
            'total_bytes':sum(p.stat().st_size for p in selected),'target_accessed':False,
            'compile_child':[PYTHON,'-m','tools.build_ops','--soc','ascend910b4','--jobs','2',
                             '--timeout','300','--log-dir','reports/fullbuild-phases'],
            'supervisor_timeout_seconds':600,'fresh_stage':True,
            'excluded':'No reports, .git, references, build outputs, unrelated experiments or untracked binary/log files.'}


def stage():
    spec=manifest();OUT.mkdir(parents=True,exist_ok=True)
    # Prior C16 diagnostic npuchk files exhausted /tmp. This is an owned
    # build-space check, not a target readiness/environment audit.
    preflight = 'import json,os,pathlib; p=pathlib.Path('+repr(STAGE)+'); s=os.statvfs(p.parent); free=s.f_bavail*s.f_frsize; assert free>=2*1024**3, ("insufficient local build space",free); p.mkdir(); print(json.dumps({"free_bytes_before_stage":free}))'
    space=subprocess.run(PREFIX+['python3','-c',preflight],capture_output=True,text=True,timeout=30)
    if space.returncode:raise RuntimeError(space.stderr.strip())
    spec['VM_space']=json.loads(space.stdout)
    data=io.BytesIO()
    with tarfile.open(fileobj=data,mode='w') as archive:
        for p in sources():archive.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
        encoded=json.dumps(spec,indent=2).encode();info=tarfile.TarInfo('source_stage.json');info.size=len(encoded)
        archive.addfile(info,io.BytesIO(encoded))
    subprocess.run(PREFIX+['tar','-xf','-','-C',STAGE],input=data.getvalue(),check=True,timeout=60)
    (OUT/'source_stage.json').write_text(json.dumps(spec,indent=2)+'\n')
    print(json.dumps({'staged':STAGE,'files':len(spec['files']),'bytes':spec['total_bytes']}))


def start():
    saved=json.loads((OUT/'source_stage.json').read_text())
    if saved['files']!=manifest()['files']:
        raise RuntimeError('source changed since staging; do not certify stale snapshot')
    command=PREFIX+['bash','-lc',f'''set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
unset LD_PRELOAD
cd {STAGE}
mkdir -p reports
{PYTHON} tools/experiments/run_camodel_arm.py --cwd {STAGE} --timeout 600 --output {STAGE}/reports/fullbuild-supervisor-logs --control-path {STAGE}/reports/fullbuild-control.json -- {PYTHON} -m tools.build_ops --soc ascend910b4 --jobs 2 --timeout 300 --log-dir reports/fullbuild-phases
''']
    return subprocess.call(command)


def collect():
    # Gather existing evidence even if configure failed before producing either
    # manifest. Missing records fail qualification; original bytes/rc survive.
    collector = """import hashlib,io,json,pathlib,sys,tarfile
root=pathlib.Path(STAGE_VALUE)
wanted=('reports','source_stage.json','build/ascendc/build_manifest.json','build/ascendc/oscar_build_signature.json')
found=[name for name in wanted if (root/name).exists()]
receipt={'requested':list(wanted),'collected':found,'missing':[name for name in wanted if name not in found]}
expected=json.loads((root/'source_stage.json').read_text())['files']
actual={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in expected if (root/name).is_file()}
verification={'files':actual,'missing':sorted(expected.keys()-actual.keys()),'matches_staged_manifest':actual==expected}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
    for name in found:archive.add(root/name,arcname=name,recursive=True)
    data=json.dumps(receipt,indent=2).encode();info=tarfile.TarInfo('collection_status.json');info.size=len(data)
    archive.addfile(info,io.BytesIO(data))
    data=json.dumps(verification,indent=2).encode();info=tarfile.TarInfo('source_verification.json');info.size=len(data)
    archive.addfile(info,io.BytesIO(data))
""".replace('STAGE_VALUE',repr(STAGE))
    result=subprocess.run(PREFIX+['python3','-c',collector],capture_output=True,timeout=60)
    (OUT/'collection.tar').write_bytes(result.stdout)
    (OUT/'collection_transport.json').write_text(json.dumps({'returncode':result.returncode,
        'stderr':result.stderr.decode(errors='replace'),'bytes':len(result.stdout)},indent=2)+'\n')
    if result.returncode:
        raise RuntimeError('collection failed; raw archive and transport rc/stderr retained')
    destination=OUT/'completed';destination.mkdir(exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(result.stdout),mode='r:') as archive:
        archive.extractall(destination,filter='data')
    load_errors={}
    def read_json(name):
        try:return json.loads((destination/name).read_text())
        except (OSError,ValueError) as error:
            load_errors[name]=f'{type(error).__name__}: {error}';return {}
    collection=read_json('collection_status.json')
    control=read_json('reports/fullbuild-control.json')
    build=read_json('reports/build.json')
    runtime=read_json('build/ascendc/build_manifest.json')
    build_signature=read_json('build/ascendc/oscar_build_signature.json')
    snapshot=read_json('source_stage.json').get('files',{})
    current=manifest()['files']
    verified=read_json('source_verification.json')
    actual=verified.get('files',{})
    import sys
    sys.path.insert(0,str(ROOT))
    from oscar_ascend.ops.contracts import SOURCE_CAPABILITIES
    caps=runtime.get('source_capabilities',[])
    exact_caps=isinstance(caps,list) and len(caps)==len(set(caps))==44 and set(caps)==SOURCE_CAPABILITIES
    passed=(not collection.get('missing') and not load_errors
        and control.get('status')=='completed' and control.get('child_returncode')==0
        and control.get('cleanup_complete') is True and build.get('build')=='passed'
        and build.get('reused') is False and build_signature.get('reused') is False
        and runtime.get('signature')==build.get('signature')==build_signature.get('signature')
        and current==snapshot==actual and verified.get('matches_staged_manifest') is True and exact_caps)
    summary={'status':'complete_production_CANN_and_Torch_build_passed' if passed else 'failed_or_source_drift',
       'scope':'fresh_local_Lima_CANN9.1_ascend910b4_production_sources',
       'stage':STAGE,'signature':runtime.get('signature'),'source_capabilities':len(caps),
       'source_capabilities_names':caps,'exact_current_44_capabilities':exact_caps,
       'reused':build.get('reused'),'collection_missing':collection.get('missing',[]),'collection_read_errors':load_errors,
       'source_matches_current':current==snapshot==actual,'actual_VM_source_hashes_verified':actual==snapshot,
       'verified_source_files':len(actual),'source_verification_missing':verified.get('missing',[]),'source_changes_since_stage':[p for p in current.keys()|snapshot.keys() if current.get(p)!=snapshot.get(p)],
       'build_seconds':control.get('child_elapsed_seconds'),'child_returncode':control.get('child_returncode'),
       'supervisor_returncode':control.get('supervisor_returncode'),'cleanup_complete':control.get('cleanup_complete'),
       'reported_build':build.get('build'),'build_error':build.get('error'),
       'NPU':'not_run','graph':'not_run','model_quality':'not_established','whole_service_performance':'not_established','target_accessed':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
    return 0 if passed else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','stage','start','collect']);a=p.parse_args()
    if a.action=='plan':
        m=manifest();print(json.dumps({k:v for k,v in m.items() if k!='files'}|{'file_count':len(m['files'])},indent=2))
    elif a.action=='stage':stage()
    elif a.action=='start':raise SystemExit(start())
    else:raise SystemExit(collect())
