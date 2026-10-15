"""Strict relocation of the complete qualified MDL CPU checker.

All actual prepare, full partial/LSE/status bytes, original merge, PR oracle,
input immutability and GM guards remain. Both q4 arms consume the same Rv NZ.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.decode_finite_20261015.generate import digest,sha
SOURCE=ROOT/'csrc/experiments/decode_mdl_load/probe.cpp'
PIN='90ced4a3090b774b361a43cd2db793a63085badb2af85b950afd0ccebbf71ce4'
def generate():
    original=SOURCE.read_text()
    if sha(original)!=PIN:raise ValueError('qualified checker source changed')
    text=original;edits=[]
    def edit(old,new):
        nonlocal text
        n=text.count(old)
        if not n:raise ValueError('checker anchor '+old)
        text=text.replace(old,new);edits.append((old,new,n))
    edit('oscar_mdl_baseline_q4_kernel','oscar_decode_finite_q4_baseline_kernel')
    edit('oscar_mdl_baseline_q1_kernel','oscar_decode_finite_q1_baseline_q1_kernel')
    edit('rv(dir+"/rv.bin",d*d*4)','rv(dir+(mode=="q4"?"/rv_nz.bin":"/rv.bin"),d*d*4)')
    edit('oscar_ascend::attention_decode_bundle_workspace_per_core(d)','size_t{1146880}')
    edit('current_production_decode_bundle','current_production_hybrid_q4_or_MDL_q1')
    edit('checkGuards();checkInputs();CheckStatuses(oldStatus,dir,fault!=0);',
         '''checkGuards();checkInputs();
      // Diagnostic receipt precedes expected-status assertion. Preserve the
      // original baseline, every status and the prepared task/input buffers.
      if(argc>=4)Write(std::string(argv[3])+"/baseline-status-diagnostic.bin",oldStatus.ptr,oldStatus.size);
      if(fault)for(size_t index=0;index<oldStatus.size/4;++index) {
        const int32_t value=At<int32_t>(oldStatus.ptr,index);
        if(value)std::cout<<"BASELINE_STATUS task="<<index/2<<" lane="<<index%2<<" code="<<value<<std::endl;
      }
      CheckStatuses(oldStatus,dir,fault!=0);''')
    restored=text
    for old,new,n in reversed(edits):
        if restored.count(new)!=n:raise ValueError('checker inverse count')
        restored=restored.replace(new,old)
    if restored!=original:raise ValueError('full checker inverse failed')
    return text,{'template':str(SOURCE.relative_to(ROOT)),'template_sha256':PIN,
       'generated_sha256':sha(text),'full_inverse':True,'math_oracle_prepare_error_guards_unchanged':True}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    text,proof=generate()
    for name,body in (('probe.cpp',text),('harness.json',json.dumps(proof,indent=2)+'\n')):
        path=HERE/name
        if a.check:
            if path.read_text()!=body:raise ValueError('checker drift '+name)
        else:path.write_text(body)
    print(json.dumps({'status':'checked' if a.check else 'generated'}))
