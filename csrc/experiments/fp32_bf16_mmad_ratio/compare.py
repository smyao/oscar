"""Archive #126/#140/#154, D.4: exact four-run evidence, never production casting."""
import argparse
import gzip
import json
from pathlib import Path
import struct
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.fp32_bf16_mmad_ratio import fixture


def compare(directory):
    source=fixture.check_source();cases=[]
    for case,live,padded,layout in fixture.PLAN:
        arms={};payloads={}
        for arm in ('fp32','bf16'):
            path=directory/case/arm
            r=json.loads((path/'report.json').read_text());c=json.loads((path/'completion.json').read_text())
            ctl=json.loads((directory/case/(arm+'-control.json')).read_text())
            if (r['arm']!=arm or r['case']!=case or r['status']!='passed' or r['failures'] or
                    c['status']!='numeric_passed_pending_profile' or c['failures'] or c['outputs']!=r['outputs'] or
                    ctl['child_returncode']!=0 or ctl['supervisor_returncode']!=0 or ctl['timed_out'] or
                    ctl['interrupt_signal'] is not None or ctl['cleanup_complete'] is not True or ctl['timeout_seconds']!=180):
                raise ValueError('missing complete180s numeric/control evidence')
            if (r['live_rows'],r['padded_rows'],r['layout'],r['grid'],r['N'],r['K'])!=(live,padded,layout,1,256,256):
                raise ValueError('unpaired geometry/grid')
            binding=r['binding']
            expected_binding={'kernel_sha256':fixture.sha((Path(__file__).parent/'kernel.cpp').read_bytes()),
                'common_sha256':source['sha256']['csrc/kernels/oscar_common.h'],
                'baseline_sha256':source['sha256']['csrc/kernels/attention_cv_decode_bundle.cpp'],
                'basic':[64,64,128],'single':[128,256,256],'C_dtype':'float32','hf32':False}
            if binding!=expected_binding:raise ValueError('build source/specification differs')
            data={}
            if set(r['outputs'])!={'a','b','c','status'}:raise ValueError('incomplete GM inventory')
            for name,item in r['outputs'].items():
                raw=(path/(name+'.bin')).read_bytes();guard=(path/(name+'.guarded.bin')).read_bytes()
                if (len(raw)!=item['bytes'] or fixture.sha(raw)!=item['sha256'] or len(guard)!=len(raw)+128 or
                        guard[64:-64]!=raw or guard[:64]!=b'\xa5'*64 or guard[-64:]!=b'\xa5'*64 or
                        fixture.sha(guard)!=item['guarded_sha256']):raise ValueError('saved payload/guard differs')
                if name in ('a','b') and fixture.sha(raw)!=r['input_sha256'][name]:raise ValueError('input changed')
                data[name]=raw
            oracle=(path/'expected_output.bin').read_bytes()
            if fixture.sha(oracle)!=r['oracle_sha256'] or data['c']!=oracle or r['oracle_byte_differences']!=0:
                raise ValueError('exact integer oracle not passed')
            fields=list(struct.unpack('<16i',data['status']));which=0 if arm=='fp32' else 1
            if fields[:11]!=[1,which,live,layout,4 if which==0 else 2,64,64,128,128,256,256] or fields[15]!=0:
                raise ValueError('completion/specification differs')
            for kind in ('task','aic'):
                rows=r['profiles'][kind]
                if len(rows)!=1 or rows[0]['duration']<=0 or rows[0]['duration']!=rows[0]['end_cycle']-rows[0]['start_cycle']:
                    raise ValueError('incomplete profile')
            task,aic=r['profiles']['task'][0],r['profiles']['aic'][0]
            if aic['core_id']!=0 or r['task_ticks']!=task['duration'] or any(aic[k]!=task[k] for k in ('task_id','stream_id')):
                raise ValueError('task/actor identity differs')
            if not r['traces'] or r['trace_cap_per_file_bytes']!=16*1024*1024:raise ValueError('missing bounded trace evidence')
            for name in ('core0.cubecore0.instr_log.dump','core0.cubecore0.instr_popped_log.dump'):
                if name not in r['traces'] or r['traces'][name]['bytes']==0:raise ValueError('missing active instruction trace')
            for name,item in r['traces'].items():
                raw=gzip.decompress((path/'log_ca'/(name+'.gz')).read_bytes())
                if len(raw)!=item['bytes'] or len(raw)>=16*1024*1024 or fixture.sha(raw)!=item['sha256']:
                    raise ValueError('incomplete/changed trace')
            arms[arm]=r;payloads[arm]=data
        for field in ('case','live_rows','padded_rows','layout','grid','N','K','library_sha256','binding','binding_sha256',
                      'fixture_sha256','mother_sha256','oracle_sha256','math_proof','simulator_config'):
            if arms['fp32'][field]!=arms['bf16'][field]:raise ValueError('unpaired '+field)
        for name in ('a','b'):
            f32=struct.unpack('<'+str(len(payloads['fp32'][name])//4)+'I',payloads['fp32'][name])
            bf16=struct.unpack('<'+str(len(payloads['bf16'][name])//2)+'H',payloads['bf16'][name])
            if len(f32)!=len(bf16) or any(x!=y<<16 for x,y in zip(f32,bf16)):raise ValueError('operands are not exactly equal')
        if payloads['fp32']['c']!=payloads['bf16']['c']:raise ValueError('FP32 output bits differ')
        left,right=arms['fp32']['task_ticks'],arms['bf16']['task_ticks']
        cases.append({'case':case,'fp32_ticks':left,'bf16_ticks':right,'R':left/right,
                      'exact_integer_oracle':'passed','cross_dtype_output_bits':'identical'})
    return {'status':'complete_exact_same_shape_ratio','cases':cases,'scope':'complete single Matmul task, not peak Cube arithmetic throughput',
            'R_at_most_three_all_cases':all(row['R']<=3 for row in cases),
            'multi_component_speedup':'not_established; casts/sums/control reuse not measured',
            'production_BF16_cast':'not_qualified','NPU':'not_run','graph':'not_run','whole_service':'not_run'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=compare(a.directory);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
