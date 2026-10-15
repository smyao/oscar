"""Archive #94/#126/#129/#145/#154-157: inherit full C16 oracle/ownership gates.

Only candidate symbols, fixed workspace and package import binding change.
No fixture filtering, tolerance relaxation or previous timeout replacement.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'c16_member_pipeline'


def sha(data):return hashlib.sha256(data.encode() if isinstance(data,str) else data).hexdigest()


def artifacts():
    result={}
    for name in ('cv_probe.cpp','cv_probe_small.cpp','cv_replay.py','cv_replay_backend_split.py',
                 'cv_compare.py','backend_split.py','run.py','CMakeLists.txt'):
        source=(OLD/name).read_text();text=source
        text=text.replace('c16_member_pipeline','c16_l1_reuse_20261015')
        # Package name carries the date; exported C symbols and build targets do not.
        text=text.replace('oscar_c16_l1_reuse_20261015','oscar_c16_l1_reuse')
        text=text.replace('OSCAR_C16_PIPELINE_CPU_DEBUG','OSCAR_C16_L1_CPU_DEBUG')
        text=text.replace('5652480','7487488')
        if name=='CMakeLists.txt':
            text=text.replace('c16_baseline.cpp c16_candidate.cpp','c16_stable.cpp c16_baseline.cpp c16_candidate.cpp')
            anchor='  target_compile_options(oscar_c16_l1_reuse_small_cpu PRIVATE -O0 -g)'
            text=text.replace(anchor,anchor+'''
  add_executable(oscar_c16_l1_reuse_nonzero_cpu cv_probe_nonzero.cpp c16_stable.cpp c16_baseline.cpp c16_candidate.cpp ../../kernels/attention_tasks.cpp)
  target_link_libraries(oscar_c16_l1_reuse_nonzero_cpu PRIVATE tikicpulib::Ascend910B4)
  target_include_directories(oscar_c16_l1_reuse_nonzero_cpu SYSTEM AFTER PRIVATE "${ASCEND_HOME_PATH}/aarch64-linux/asc/include")
  target_compile_options(oscar_c16_l1_reuse_nonzero_cpu PRIVATE -O0 -g)
  add_executable(oscar_c16_l1_reuse_qr_cpu cv_probe_qr.cpp c16_stable.cpp c16_baseline.cpp c16_candidate.cpp ../../kernels/attention_tasks.cpp)
  target_link_libraries(oscar_c16_l1_reuse_qr_cpu PRIVATE tikicpulib::Ascend910B4)
  target_include_directories(oscar_c16_l1_reuse_qr_cpu SYSTEM AFTER PRIVATE "${ASCEND_HOME_PATH}/aarch64-linux/asc/include")
  target_compile_options(oscar_c16_l1_reuse_qr_cpu PRIVATE -O0 -g)
''')
        if name in ('cv_probe.cpp','cv_probe_small.cpp'):
            text=text.replace('extern "C" void oscar_c16_l1_reuse_baseline_kernel(CV_ARGS);',
                'extern "C" void oscar_c16_l1_reuse_stable_kernel(CV_ARGS);\nextern "C" void oscar_c16_l1_reuse_baseline_kernel(CV_ARGS);')
            text=text.replace('arm!="baseline"&&arm!="candidate"','arm!="stable"&&arm!="baseline"&&arm!="candidate"')
            text=text.replace('if(arm=="baseline"){RUN(oscar_c16_l1_reuse_baseline_kernel);}',
                'if(arm=="stable"){RUN(oscar_c16_l1_reuse_stable_kernel);}else if(arm=="baseline"){RUN(oscar_c16_l1_reuse_baseline_kernel);}')
        if name in ('cv_replay.py','cv_replay_backend_split.py'):
            text=text.replace("generate_c4.generate(a.variant=='candidate')","generate_c4.generate_arm(a.variant)")
            text=text.replace("choices=('baseline','candidate')","choices=('stable','baseline','candidate')")
        result[name]=(text,{'template':str((OLD/name).name),'template_sha256':sha(source),
            'generated_sha256':sha(text),'symbols_workspace_imports_only':True})
    text=result['cv_replay.py'][0].replace('672','336').replace('2016','1008')
    text=text.replace("'<4032i'","'<2016i'")
    text=text.replace('attrs[13:]!=[64,256,3,1]','attrs[13:]!=[64,1,3,1]')
    text=text.replace('(337,1341)','(65,337)')
    result['cv_replay_small.py']=(text,{'template':'generated cv_replay.py',
        'generated_sha256':sha(text),'scope':'Q336/recent1 C65/C337 complete, not Q672/recent256'})
    text=result['cv_probe_small.cpp'][0].replace('336','672').replace('1008','2016')
    text=text.replace('a[4]!=1','a[4]!=2')
    text=text.replace('a[5]*4)','2*a[5]*4)')
    text=text.replace('starts(dir/"starts.bin",8),lens(dir/"lens.bin",4)',
        'starts(dir/"starts.bin",12),lens(dir/"lens.bin",8)')
    text=text.replace('At<int32_t>(lens.p,0)-672','At<int32_t>(lens.p,1)-336')
    text=text.replace('(context!=65&&context!=337)','(context!=337||At<int32_t>(lens.p,0)!=336)')
    text=text.replace('int64_t{1},int64_t{672},int64_t{6}', 'int64_t{2},int64_t{672},int64_t{6}')
    text=text.replace('!=context+i)', '!=(i<336?i:337+i-336))')
    text=text.replace('if(clusters!=1||leaders!=16||skips!=16||',
        'if(At<int64_t>(stats.p,8)!=1||clusters!=1||leaders!=16||skips!=16||')
    anchor='for(int row=0;row<672*6;++row){double maximum='
    text=text.replace(anchor,'''for(int row=0;row<672*6;++row){
      if(row<336*6){for(int s=0;s<3;++s){
        const float l=At<float>(lse.p,row*3+s);
        if(!std::isinf(l)||l>=0)throw std::runtime_error("cold request CV LSE must be -inf");
        for(int d=0;d<256;++d)if(At<float>(partial.p,(row*3+s)*256+d)!=0.0F)
          throw std::runtime_error("cold request CV partial must be zero");
      }continue;}
      double maximum=''')
    text='// Legal two-request gate: cold CV empty, warm C16 owner1/qbegin336, all2016 tasks.\n'+text
    result['cv_probe_nonzero.cpp']=(text,{'template':'generated cv_probe_small.cpp',
        'generated_sha256':sha(text),'scope':'R2 contexts0/337 Q336 each, recent1, cluster owner1'})
    text=result['cv_probe_small.cpp'][0]
    anchor='if(At<int32_t>(status.p,id*2+lane)!=expected)'
    text=text.replace(anchor,'if(row[1]>0&&row[7]==0&&row[0]==315&&lane==0)expected=2;\n    '+anchor)
    anchor='for(int row=0;row<336*6;++row){double maximum='
    text=text.replace(anchor,'''for(int row=0;row<336*6;++row){
      if(row>=315*6&&row<315*6+64){
        if(!std::isnan(At<float>(lse.p,row*3)))throw std::runtime_error("QR fault history LSE must fail closed");
        for(int d=0;d<256;++d)if(!std::isnan(At<float>(partial.p,row*3*256+d)))
          throw std::runtime_error("QR fault history partial must fail closed");
        continue;
      }
      double maximum=''')
    text=text.replace('bad<0?"numeric":"invalid_metadata_status"',
        'bad<0?"numeric_except_expected_QR_fault_rows":"invalid_metadata_status"')
    text='// Last C16 member QR NaN: q315/head0, only history member15 AIV0 error2.\n'+text
    result['cv_probe_qr.cpp']=(text,{'template':'generated cv_probe_small.cpp',
        'generated_sha256':sha(text),'scope':'late member15 QR NaN; 64 history rows fail closed; remaining rows dense oracle'})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();manifest={}
    for name,(text,entry) in artifacts().items():
        if a.check:
            if (HERE/name).read_text()!=text:raise ValueError('harness drift '+name)
        else:(HERE/name).write_text(text)
        manifest[name]=entry
    text=json.dumps(manifest,indent=2)+'\n'
    if a.check:
        if (HERE/'harness.json').read_text()!=text:raise ValueError('manifest drift')
    else:(HERE/'harness.json').write_text(text)
    print(json.dumps({'status':'checked' if a.check else 'generated','execution':'not_run'}))
