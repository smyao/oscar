"""Archive #94/#125/#126/#140/#145/#154, D.4: paired K-major qualification.

Require real completion/control/profile, exact paired bytes before the frozen
independent oracle, and each output's K128 order. Global MMAD issue order is
intentionally different. Pure Matmul timing excludes publisher/packing cost.
"""
import argparse
import gzip
import json
from pathlib import Path
import struct
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.fp32_kmajor_matmul import fixture
from csrc.experiments.fp32_kmajor_matmul.replay import instruction_spec, oracle_gate

PLAN=(('qk_m24',24,32,0,True),('pv_m126',126,128,1,True),('cancel_qk_m24',24,32,0,False))


def expected_binding():
    return {'kernel_sha256':fixture.sha((Path(__file__).parent/'kernel.cpp').read_bytes()),
            'common_sha256':fixture.sha((ROOT/'csrc/kernels/oscar_common.h').read_bytes()),
            'baseline_sha256':fixture.sha((ROOT/'csrc/kernels/attention_cv_decode_bundle.cpp').read_bytes()),
            'mm_api_sha256':fixture.sha((ROOT/'artifacts/cann_sdk_cube_public_20261010/include/basic_api/kernel_struct_mm.h').read_bytes()),
            'fix_api_sha256':fixture.sha((ROOT/'artifacts/cann_sdk_cube_public_20261010/include/basic_api/kernel_struct_fixpipe.h').read_bytes()),
            'basic':[64,64,128],'single':[128,256,256],'C_dtype':'float32','hf32':False,
            'A_formats':['NZ','NZ'],'B_formats':['NZ','NZ'],'C0':8,
            'org_m':'physical_padded_rows','single_m':'live_rows','M_blocks':'24_or_64_62',
            'K_order_per_output':[0,128],'loop_order':['SDK','K_N_M']}


def compare(directory):
    source=fixture.check_source();binding=expected_binding();results=[]
    for case,live,padded,layout,exact in PLAN:
        arms={};payloads={};goldens={}
        for which,arm in enumerate(('sdk','kmajor')):
            path=directory/case/arm
            r=json.loads((path/'report.json').read_text());completion=json.loads((path/'completion.json').read_text())
            control=json.loads((directory/case/(arm+'-control.json')).read_text())
            if (r['arm']!=arm or r['case']!=case or r['status']!='passed' or r['failures'] or
                    completion['status']!='numeric_passed_pending_profile' or completion['failures'] or
                    completion['outputs']!=r['outputs'] or control['child_returncode']!=0 or
                    control['supervisor_returncode']!=0 or control['timed_out'] or control['interrupt_signal'] is not None or
                    control['cleanup_complete'] is not True or control['timeout_seconds']!=180):
                raise ValueError('missing successful180s numeric/control evidence: '+case+'/'+arm)
            if (r['live_rows'],r['padded_rows'],r['layout'],r['grid'],r['N'],r['K'])!=(live,padded,layout,1,256,256):
                raise ValueError('geometry/grid differs')
            if r['binding']!=binding or r['fixture_source']!=source or r['exact_integer_oracle'] is not exact:
                raise ValueError('source/spec/oracle identity differs')
            if r['A_format']!='NZ' or r['B_format']!='NZ' or r['org_m']!=padded or r['single_m']!=live or r['prepack_in_timing'] is not False:
                raise ValueError('layout/timing scope changed')
            data={}
            if set(r['outputs'])!={'a','b','c','status'}:raise ValueError('incomplete GM inventory')
            for name,item in r['outputs'].items():
                raw=(path/(name+'.bin')).read_bytes();guarded=(path/(name+'.guarded.bin')).read_bytes()
                if (len(raw)!=item['bytes'] or fixture.sha(raw)!=item['sha256'] or len(guarded)!=len(raw)+128 or
                        guarded[64:-64]!=raw or guarded[:64]!=b'\xa5'*64 or guarded[-64:]!=b'\xa5'*64 or
                        fixture.sha(guarded)!=item['guarded_sha256']):raise ValueError('payload/guard differs')
                if name in ('a','b') and fixture.sha(raw)!=r['input_sha256'][name]:raise ValueError('input changed')
                data[name]=raw
            fields=list(struct.unpack('<32i',data['status']))
            if (fields[:11]!=[1,which,live,layout,4,64,64,128,128,256,256] or fields[15]!=0 or
                    fields[24:30]!=[1,8,1,padded,padded,live] or not 0<fields[11]<=512*1024):
                raise ValueError('completion/specification differs')
            for kind in ('task','aic'):
                rows=r['profiles'][kind]
                if len(rows)!=1 or rows[0]['duration']<=0 or rows[0]['duration']!=rows[0]['end_cycle']-rows[0]['start_cycle']:
                    raise ValueError('incomplete profile')
            task,aic=r['profiles']['task'][0],r['profiles']['aic'][0]
            if aic['core_id']!=0 or r['task_ticks']!=task['duration'] or any(aic[k]!=task[k] for k in ('task_id','stream_id')):
                raise ValueError('task/actor identity differs')
            if r['trace_cap_per_file_bytes']!=16*1024*1024:raise ValueError('trace cap changed')
            for name in ('core0.cubecore0.instr_log.dump','core0.cubecore0.instr_popped_log.dump'):
                if name not in r['traces'] or r['traces'][name]['bytes']==0:raise ValueError('active instruction trace missing')
            for name,item in r['traces'].items():
                raw=gzip.decompress((path/'log_ca'/(name+'.gz')).read_bytes())
                if len(raw)!=item['bytes'] or len(raw)>=16*1024*1024 or fixture.sha(raw)!=item['sha256']:
                    raise ValueError('incomplete/changed trace')
            spec=instruction_spec(path/'log_ca',live,arm)
            if spec!=r['instructions'] or spec['status']!='passed':raise ValueError('per-output MMAD K128 order failed')
            golden=(path/'expected_output.bin').read_bytes()
            if fixture.sha(golden)!=r['oracle_sha256']:raise ValueError('golden changed')
            arms[arm]=r;payloads[arm]=data;goldens[arm]=golden
        for field in ('case','live_rows','padded_rows','layout','grid','N','K','library_sha256','binding','binding_sha256',
                      'org_m','single_m','fixture_sha256','origin_fixture_sha256','oracle_sha256','math_proof',
                      'simulator_config','fixture_source','exact_integer_oracle','input_sha256'):
            if arms['sdk'][field]!=arms['kmajor'][field]:raise ValueError('unpaired '+field)
        if payloads['sdk']['a']!=payloads['kmajor']['a'] or payloads['sdk']['b']!=payloads['kmajor']['b']:
            raise ValueError('same full-NZ operands required')
        # Check all paired output bits first, including live signed-zero bits and
        # padding. Never replace a failed pair with a tolerance-only success.
        if payloads['sdk']['c']!=payloads['kmajor']['c']:raise ValueError('paired FP32 output bits differ: '+case)
        for arm in ('sdk','kmajor'):
            gate=oracle_gate(payloads[arm]['c'],goldens[arm],live,exact)
            if not gate['passed'] or gate!=arms[arm]['oracle']:raise ValueError('frozen independent oracle failed')
        left,right=arms['sdk']['task_ticks'],arms['kmajor']['task_ticks']
        results.append({'case':case,'sdk_ticks':left,'kmajor_ticks':right,'local_sdk_over_kmajor':left/right,
                        'paired_output_bits':'identical','independent_oracle':'passed',
                        'per_output_K128_order':'passed','global_MMAD_order':'intentionally_different'})
    return {'status':'complete_paired_kmajor_qualification','cases':results,
            'scope':'bounded prepacked Matmul only; publisher, packing, complete attention and service cost excluded',
            'precision_domain':'three finite cases only; NaN/Inf/error-domain and full-attention gates still pending',
            'NPU':'not_run','graph':'not_run','whole_service':'not_run'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=compare(a.directory);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
