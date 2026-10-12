"""Reuse qualified full-C4 drivers with explicit bounded source edits.

No precision/oracle/guard/completion gate is removed. The source harness and
all edit identities are recorded, and no prior H32 evidence is mutated.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_member_pipeline.generate import HERE,sha
OLD=HERE.parent/'c4_unpack32'


def emit(name,changes):
    original=(OLD/name).read_text();text=original;edits=[]
    for old,new,count in changes:
        if text.count(old)!=count:raise ValueError((name,old,text.count(old),count))
        text=text.replace(old,new);edits.append((old,new,count))
    restored=text
    for old,new,count in reversed(edits):
        if restored.count(new)!=count:raise ValueError('entrypoint inverse drift '+name)
        restored=restored.replace(new,old)
    if restored!=original:raise ValueError('entrypoint source restore '+name)
    return text,{'qualified_template':str((OLD/name).relative_to(ROOT)),
        'template_sha256':sha(original),'generated_sha256':sha(text),'source_restore':True}


def artifacts():
    result={}
    common=[('oscar_c4_unpack32_','oscar_c4_member_pipeline_',4),('C4_UNPACK32_','C4_MEMBER_PIPELINE_',2)]
    result['cv_probe.cpp']=emit('cv_probe.cpp',common+[
        ('bad!=96','bad!=96&&bad!=336',1),
        ('workspace(20*1839104)','workspace(20*2494464)',1),
        ('if(At<int32_t>(lens.p,0)!=449)throw std::runtime_error("requires context113");',
         'const int context=At<int32_t>(lens.p,0)-336;\n  if((context!=113&&context!=337)||bad>=context)throw std::runtime_error("declared context/fault required");',1),
        ('!=113+i','!=context+i',1),
        ('\\"context\\":113','\\"context\\":"<<context<<"',2),
    ])
    result['cv_replay.py']=emit('cv_replay.py',[
        ('from csrc.experiments.c4_unpack32 import generate_c4','from csrc.experiments.c4_member_pipeline import generate as generate_c4',1),
        ("bad=meta['bad_position'];receipt=", "context=meta['context'];bad=meta['bad_position'];receipt=",1),
        ('bad not in (-1,80,96)','bad not in (-1,80,96,336) or bad>=context or context not in (113,337)',1),
        ("receipt['context']!=113", "receipt['context']!=context",1),
        ("'context':113", "'context':context",1),
        ("'workspace':20*1839104", "'workspace':20*2494464",1),
        ("'source_sha256':identity['source_sha256']", "'source_sha256':identity['source_sha256'],'H32_source_sha256':identity['H32_source_sha256']",1),
    ])
    result['cv_compare.py']=emit('cv_compare.py',[
        ("report['context']!=113", "report['context'] not in (113,337)",1),
        ("if old['bad_position']!=new['bad_position']", "if old['context']!=new['context'] or old['bad_position']!=new['bad_position']",1),
        ("'context':113", "'context':old['context']",1),
        ("'scope':'Q336/R1/C113/S1/20core source0+1, all1008tasks; not production/NPU/graph'", "'scope':f\"Q336/R1/C{old['context']}/S1/20core source0+1, all1008tasks; not production/NPU/graph\"",1),
        ("'source_sha256')", "'source_sha256','H32_source_sha256')",1),
    ])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();manifest={}
    for name,(text,identity) in artifacts().items():
        manifest[name]=identity;path=HERE/name
        if a.check:
            if path.read_text()!=text:raise ValueError('entrypoint changed '+name)
        else:path.write_text(text)
    text=json.dumps(manifest,indent=2)+'\n';path=HERE/'entrypoints.json'
    if a.check:
        if path.read_text()!=text:raise ValueError('entrypoint manifest changed')
    else:path.write_text(text)
    print(json.dumps({'status':'checked' if a.check else 'generated','templates_unchanged':True,'execution':'not_run'}))
