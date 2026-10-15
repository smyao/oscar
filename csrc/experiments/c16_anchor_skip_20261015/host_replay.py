"""Archive #34/#36/#129/#148/#152/#157: native-reachable traversal proof.

Compile only an ordinary C++ event recorder using the exact extracted Process,
IsClusterMember and TaskValid source. This is not an AscendC numerical run.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,struct,subprocess,tempfile,time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c16_anchor_skip_20261015 import generate as gen

def sha(data):return hashlib.sha256(data).hexdigest()
def run(command,cwd):
    begin=time.monotonic();p=subprocess.run(command,cwd=cwd,capture_output=True,text=True,timeout=120)
    if p.returncode:raise RuntimeError({'command':command,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    return p,time.monotonic()-begin

PREPARE_SOURCE_SHA='1e6c7c4e4d5f71d8c8679573bcc3c4d523361f93ae4e16b2fa3606acb7bc3952'

def prove(output):
    if sha((ROOT/'csrc/kernels/attention_tasks.cpp').read_bytes())!=PREPARE_SOURCE_SHA:
        raise ValueError('native task producer changed; re-audit the transcription and reachability proof')
    if gen.host_source()!=(HERE/'host_replay.cpp').read_text():raise ValueError('host source extraction drift')
    provenance=json.loads((HERE/'fixtures/provenance.json').read_text())
    report={'status':'running','scope':'ordinary host control-flow equivalence; no floating-point kernel execution',
       'basis':{'source':gen.SOURCE,'sha256':gen.SOURCE_SHA,'qualification':'Rv separately qualified before combination; do not multiply old speedups'},
       'source_sha256':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [HERE/'generate.py',HERE/'host_template.cpp',HERE/'host_replay.cpp',ROOT/'csrc/kernels/attention_tasks.cpp']},
       'archive':['#34','#36','#126','#129','#144','#145','#148','#152','#154','#157'],
       'initial_host_harness_compile_failure':'TaskState member name and absent mm.End stub corrected before the recorded passing build; no CANN/VM run',
       'actual_prepare_cases':{},'CANN':'not_run','NPU':'not_run','graph':'not_run','performance':'not_established'}
    with tempfile.TemporaryDirectory(prefix='oscar-c16-anchor-proof-',dir='/tmp') as temp:
        temp=Path(temp);binary=temp/'host-replay'
        built,seconds=run(['c++','-std=c++17','-O2',str(HERE/'host_replay.cpp'),'-o',str(binary)],ROOT)
        report['host_compile']={'returncode':built.returncode,'seconds':seconds,'stderr':built.stderr}
        result,seconds=run([str(binary)],ROOT);report['enumeration']=json.loads(result.stdout);report['enumeration']['host_seconds']=seconds
        for name,witness in provenance.items():
            data=HERE/'fixtures'/name;native=json.loads((data/'native.json').read_text());attrs=native['attrs'];fixture=temp/name;prepared=fixture/'prepared';prepared.mkdir(parents=True)
            for file,kind in [('starts.bin','i'),('lens.bin','i'),('slots.bin','q')]:
                values=native['arrays'][file];raw=struct.pack('<'+str(len(values))+kind,*values)
                if sha(raw)!=witness['native_arrays_sha256'][file]:raise ValueError('native metadata input digest mismatch')
                (fixture/file).write_bytes(raw)
            zipped=(data/'tasks.bin.gz').read_bytes();raw=gzip.decompress(zipped)
            if sha(zipped)!=witness['compressed_task_sha256'] or sha(raw)!=witness['task_sha256']:raise ValueError('actual prepare task digest mismatch')
            (prepared/'tasks.bin').write_bytes(raw)
            command=[str(binary),'--actual',str(fixture),str(prepared),*map(str,[attrs[0],attrs[1],attrs[2],attrs[4],attrs[16],attrs[13],attrs[14]])]
            result,seconds=run(command,ROOT)
            report['actual_prepare_cases'][name]={**json.loads(result.stdout),'host_seconds':seconds,'task_sha256':witness['task_sha256'],'original_prepare_receipt':json.loads((data/'prepare_receipt.json').read_text())}
    report['status']='passed_native_reachable_event_equivalence_only'
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'reports/c16_anchor_skip_20261015/host_proof.json');args=parser.parse_args()
    result=prove(args.output);print(json.dumps({'status':result['status'],'cases':result['enumeration']['cases'],'actors':result['enumeration']['actors'],'actual_prepare_cases':list(result['actual_prepare_cases']),'report':str(args.output)}))
