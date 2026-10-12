"""Archive #140/#148/#155/D.4: matched completed CAModel math-task comparison.

Model ticks are not NPU milliseconds or whole-service wall; never impute missing data.
"""
import argparse,hashlib,json,struct
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--kv',type=int,choices=(512,1024),required=True)
a=p.parse_args();root=Path('reports/wide_fp32_simd_camodel_20261006')
A=json.loads((root/f'b{a.kv}-kv256/report.json').read_text())
B=json.loads((root/f'b{a.kv}-wide/report.json').read_text())
for side,j in (('kv256',A),('wide',B)):
    assert j['status']=='passed' and not j['failures'],(side,j['status'],j['failures'])
    for name,m in j['outputs'].items():
        data=(root/f'b{a.kv}-{side}'/m['path']).read_bytes()
        assert len(data)==m['bytes'] and hashlib.sha256(data).hexdigest()==m['sha256']
    completion=json.loads((root/f'b{a.kv}-{side}/completion.json').read_text())
    assert completion['status']=='device_complete_outputs_saved' and not completion['guard_failures']
    assert completion['input_sha256']==j['input_sha256'] and completion['outputs']==j['outputs']
    host=json.loads((root/f'b{a.kv}-{side}-host.json').read_text())
    assert not host['timed_out'] and host['process_rc']==0 and host['fetch_rc']==0
for key in ('M','D','B','grid','actors','hf32','scope','input_sha256'):assert A[key]==B[key],key
pairs={}
for name in ('output','lse'):
    x=(root/f'b{a.kv}-kv256/{name}.bin').read_bytes();y=(root/f'b{a.kv}-wide/{name}.bin').read_bytes()
    x=struct.unpack('<'+'f'*(len(x)//4),x);y=struct.unpack('<'+'f'*(len(y)//4),y)
    pairs[name+'_max_abs']=max((abs(u-v) for u,v in zip(x,y) if u!=v),default=0.)
ratio=A['task_ticks']/B['task_ticks']
report={'status':'completed_matched_math_comparison','B':a.kv,'M':126,'D':256,'grid':1,
        'A_ticks':A['task_ticks'],'B_ticks':B['task_ticks'],'A_over_B':ratio,
        'B_duration_reduction_fraction':1-B['task_ticks']/A['task_ticks'],
        'gain_ge_1_5':ratio>=1.5,'decision':'B1024_eligible' if a.kv==512 and ratio>=1.5 else 'stop_no_shape_search',
        'A_oracle':A['oracle'],'B_oracle':B['oracle'],'A_B_comparison':pairs,
        'A_UB_bytes':A['ub_explicit_bytes'],'B_UB_bytes':B['ub_explicit_bytes'],
        'A_workspace_bytes':A['workspace_bytes'],'B_workspace_bytes':B['workspace_bytes'],
        'scope':A['scope'],'input_sha256':A['input_sha256'],
        'A_library_sha256':A['library_sha256'],'B_library_sha256':B['library_sha256'],
        'A_kernel_sha256':A['kernel_sha256'],'B_kernel_sha256':B['kernel_sha256'],
        'model_clock':'CAModel shared task ticks; no conversion to NPU ms',
        'NPU':'not_run','graph':'not_run','model_quality':'not_run','whole_service':'not_run','promotion_allowed':False}
(root/f'b{a.kv}-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
