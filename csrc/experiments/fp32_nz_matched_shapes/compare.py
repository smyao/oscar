"""Archive #126/#140/#154, D.4: exact six-run matched-shape ND/B-NZ/AB-NZ upper bound; never free publisher conversion."""
import argparse
import gzip
import json
from pathlib import Path
import struct
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.fp32_nz_matched_shapes import fixture
from csrc.experiments.fp32_nz_matched_shapes.replay import instruction_spec


def compare(directory):
    source=fixture.check_source();cases=[]
    for case,live,padded,layout in (('qk_m126',126,128,0),('pv_m24',24,32,1)):
        arms={};payloads={}
        for arm in ('nd','b_nz','ab_nz'):
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
                'common_sha256':source['mother_source']['sha256']['csrc/kernels/oscar_common.h'],
                'baseline_sha256':source['mother_source']['sha256']['csrc/kernels/attention_cv_decode_bundle.cpp'],
                'basic':[64,64,128],'single':[128,256,256],'C_dtype':'float32','hf32':False,'A_formats':['ND','ND','NZ'],'B_formats':['ND','NZ','NZ'],'C0':8,
                'org_m':'a_nd_live_a_nz_physical_padded_rows','single_m':'live_rows'}
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
            fields=list(struct.unpack('<32i',data['status']));which={'nd':0,'b_nz':1,'ab_nz':2}[arm]
            if fields[:11]!=[1,which,live,layout,4,64,64,128,128,256,256] or fields[15]!=0 or fields[25]!=8 or fields[24]!=int(which!=0) or fields[26]!=int(which==2) or fields[27:30]!=[padded,padded if which==2 else live,live]:
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
            spec=instruction_spec(path/'log_ca',live)
            if spec!=r['instructions'] or spec['status']!='passed':raise ValueError('MMAD K128 sequence not passed')
            arms[arm]=r;payloads[arm]=data
        for candidate in ('b_nz','ab_nz'):
            for field in ('case','live_rows','padded_rows','layout','grid','N','K','library_sha256','binding','binding_sha256',
                          'single_m','fixture_sha256','mother_sha256','oracle_sha256','math_proof','simulator_config','fixture_source'):
                if arms['nd'][field]!=arms[candidate][field]:raise ValueError('unpaired '+field)
            candidate_a=payloads[candidate]['a'] if candidate=='b_nz' else fixture.unpack_nz(payloads[candidate]['a'],padded)
            if candidate_a!=payloads['nd']['a'] or fixture.unpack_nz(payloads[candidate]['b'])!=payloads['nd']['b']:
                raise ValueError('input raw-bit equality after NZ inverse failed')
            if payloads['nd']['c']!=payloads[candidate]['c']:raise ValueError('FP32 output bits differ')
            if arms['nd']['instructions']['sequence']!=arms[candidate]['instructions']['sequence']:
                raise ValueError('actual full MMAD operand/initialization/K128 sequence differs')
        for arm,r in arms.items():
            if r['org_m']!=(padded if arm=='ab_nz' else live):raise ValueError('ND production/NZ physical OrgM differs')
            if r['A_format']!=('NZ' if arm=='ab_nz' else 'ND') or r['B_format']!=('ND' if arm=='nd' else 'NZ') or r['prepack_in_timing'] is not False:
                raise ValueError('A/B layout/timing scope differs')
        ticks={arm:r['task_ticks'] for arm,r in arms.items()}
        cases.append({'case':case,'ticks':ticks,
                      'upper_bound_nd_over_b_nz':ticks['nd']/ticks['b_nz'],
                      'upper_bound_nd_over_ab_nz':ticks['nd']/ticks['ab_nz'],
                      'exact_integer_oracle':'passed','cross_format_output_bits':'identical',
                      'org_m':{'nd':live,'b_nz':live,'ab_nz':padded},'single_m':live,'physical_a_rows':padded,
                      'full_MMAD_K128_sequence':'identical'})
    return {'status':'complete_exact_ND_NZ_upper_bound','cases':cases,
            'scope':'same F32 Matmul, host prepack excluded; bounded publisher conversion is not measured',
            'all_cases_best_layout_at_least_1_5':all(max(row['upper_bound_nd_over_b_nz'],row['upper_bound_nd_over_ab_nz'])>=1.5 for row in cases),
            'production_qualification':'pending publisher/Rv/layout/lifetime/graph and full-operator gates',
            'NPU':'not_run','graph':'not_run','whole_service':'not_run'}



if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=compare(a.directory);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
