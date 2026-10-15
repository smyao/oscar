"""Archive #94/#126/#145/#148/#154, D.4: bounded C320 full-path harness.

Relocate the already-qualified Rv full checker, retaining all1008 tasks,
60 actors, exact outputs/status/counters, independent oracle and guard checks.
Only the fixed context/symbol/source identity changes. No device claims.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent

def sha(data): return hashlib.sha256(data.encode() if isinstance(data,str) else data).hexdigest()

def artifacts():
    result={}
    for name in ('cv_probe_small.cpp','cv_replay_small.py'):
        template=ROOT/'csrc/experiments/c16_rv_reuse_20261015'/name
        original=template.read_text();text=original;edits=[]
        def edit(old,new):
            nonlocal text
            count=text.count(old)
            if count<1:raise ValueError('missing fixed harness anchor '+old)
            text=text.replace(old,new);edits.append((old,new,count))
        edit('c16_rv_reuse','c16_b1_followup')
        if name.endswith('.cpp'):
            edit('(context!=65&&context!=337)','context!=320')
        else:
            edit('context not in (65,337)','context!=320')
            edit('real complete Q336/C113 CPU metadata/evidence required','real complete Q336/C320 CPU metadata/evidence required')
            edit("'H32_source_sha256':identity['H32_source_sha256'],",'')
        restored=text
        for old,new,count in reversed(edits):
            if new:
                if restored.count(new)!=count:raise ValueError('inverse count '+new)
                restored=restored.replace(new,old)
            else:
                # Sole removed obsolete provenance field; restore at its
                # original neighboring source identity field.
                anchor="'source_sha256':identity['source_sha256'],"
                if restored.count(anchor)!=1:raise ValueError('inverse provenance anchor')
                restored=restored.replace(anchor,anchor+old)
        if restored!=original:raise ValueError('full checker inverse mismatch')
        result[name]=(text,{'template':str(template.relative_to(ROOT)),
            'template_sha256':sha(original),'generated_sha256':sha(text),'full_inverse':True})
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    manifest={}
    for name,(text,identity) in artifacts().items():
        path=HERE/name
        if args.check:
            if path.read_text()!=text:raise ValueError('harness drift '+name)
        else:path.write_text(text)
        manifest[name]=identity
    path=HERE/'harness.json';body=json.dumps(manifest,indent=2)+'\n'
    if args.check:
        if path.read_text()!=body:raise ValueError('harness manifest drift')
    else:path.write_text(body)
    print(json.dumps({'status':'checked' if args.check else 'generated','device_execution':'not_run'}))
