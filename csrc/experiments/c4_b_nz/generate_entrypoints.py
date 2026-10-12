"""Archive #13-22/#108/#125/#126/#145/#148/#154, D.4: thin C4 B-NZ harness.

Preserve real prepare, all1008 tasks, independent oracle, guards, failure codes
and60 actor profiles. Only candidate Rv is a host bit permutation; K/V are not.
Every textual edit is reversible to the qualified pipeline harness.
"""
import argparse
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];MOTHER=HERE.parent/'c4_member_pipeline'
def sha(data):return hashlib.sha256(data.encode() if isinstance(data,str) else data).hexdigest()


def emit(name,edits):
    original=(MOTHER/name).read_text();text=original
    for old,new,count in edits:
        if text.count(old)!=count:raise ValueError((name,old[:90],text.count(old),count))
        text=text.replace(old,new)
    restored=text
    for old,new,count in reversed(edits):
        if restored.count(new)!=count:raise ValueError('inverse entrypoint ambiguity '+name)
        restored=restored.replace(new,old)
    if restored!=original:raise ValueError('entrypoint inverse differs '+name)
    return text,{'template':str((MOTHER/name).relative_to(ROOT)),'template_sha256':sha(original),
                 'generated_sha256':sha(text),'full_source_restore':True}


RV_PROOF='''  const auto rvNatural=Read(dir/"rv.bin",256*256*4),rvNz=Read(dir/"rv_nz.bin",256*256*4);
  for(size_t r=0;r<256;++r)for(size_t c=0;c<256;++c)
    if(At<uint32_t>(rvNatural.data(),r*256+c)!=At<uint32_t>(rvNz.data(),(c/8)*256*8+r*8+c%8))
      throw std::runtime_error("Rv full uint32 NZ inverse differs");
'''
RV_SAVE='''  rv.save(out/"rv_input.bin");
  {std::ofstream f(out/"rv_natural.bin",std::ios::binary),g(out/"rv_input.guarded.bin",std::ios::binary);
   f.write(reinterpret_cast<const char*>(rvNatural.data()),rvNatural.size());
   g.write(reinterpret_cast<const char*>(rv.base),rv.n+128);
   if(!f.flush()||!g.flush())throw std::runtime_error("Rv ownership evidence save");}
'''
RV_REPORT='''\\"rv_inverse\\":true,\\"rv_layout\\":\\""<<(arm=="candidate"?"NZ_C0_8":"ND")<<"\\",'''


def artifacts():
    probe=emit('cv_probe.cpp',[
        ('oscar_c4_member_pipeline_','oscar_c4_b_nz_',4),('C4_MEMBER_PIPELINE_','C4_B_NZ_',2),
        ('  Gm q(dir/"q.bin",',RV_PROOF+'  Gm q(dir/"q.bin",',1),
        ('rv(dir/"rv.bin",256*256*4)','rv(dir/(arm=="candidate"?"rv_nz.bin":"rv.bin"),256*256*4)',1),
        ('(context!=113&&context!=337)','context!=337',1),
        ('  for(size_t i=0;i<inputs.size();++i){inputs[i]->guard();if(std::memcmp(before[i].data(),inputs[i]->p,inputs[i]->n))throw std::runtime_error("input modified");}',
         RV_SAVE+'  for(size_t i=0;i<inputs.size();++i){inputs[i]->guard();if(std::memcmp(before[i].data(),inputs[i]->p,inputs[i]->n))throw std::runtime_error("input modified");}',1),
        ('\\"NPU\\":\\"not_run\\"}',RV_REPORT+'\\"NPU\\":\\"not_run\\"}',1),
    ])
    replay=emit('cv_replay.py',[
        ('from csrc.experiments.c4_member_pipeline import generate as generate_c4',
         'from csrc.experiments.c4_b_nz import generate as generate_c4\nfrom csrc.experiments.c4_b_nz.fixture import verify_rv',1),
        ("    out=a.output.resolve();", "    binding_path=a.library.parent.parent/'source_binding.json'\n    binding=json.loads(binding_path.read_text())\n    if binding!=generate_c4.build_binding():raise ValueError('configured build input binding differs')\n    if a.library.name!='liboscar_c4_b_nz.so':raise ValueError('explicit B-NZ research library required')\n    out=a.output.resolve();",1),
        ('context not in (113,337)','context!=337',1),
        ("receipt['context']!=context", "receipt.get('arm')!=a.variant or receipt.get('rv_inverse') is not True or receipt.get('rv_layout')!=('NZ_C0_8' if a.variant=='candidate' else 'ND') or receipt['context']!=context",1),
        ("    inputs['tasks']=(a.prepared/'tasks.bin').read_bytes();", "    natural_rv=inputs['rv'];packed_rv=(a.fixture/'rv_nz.bin').read_bytes()\n    if sha(packed_rv)!=meta['rv_nz_sha256'] or meta['host_prepacked_KV'] is not False:raise ValueError('Rv identity/KV publisher scope changed')\n    rv_proof=verify_rv(natural_rv,packed_rv)\n    if a.variant=='candidate':inputs['rv']=packed_rv\n    if (a.prepared/'rv_natural.bin').read_bytes()!=natural_rv or (a.prepared/'rv_input.bin').read_bytes()!=inputs['rv']:raise ValueError('CPU Rv evidence differs')\n    (out/'rv_natural.bin').write_bytes(natural_rv);(out/'rv_input.bin').write_bytes(inputs['rv'])\n    inputs['tasks']=(a.prepared/'tasks.bin').read_bytes();",1),
        ("'generated_sha256':sha(source.encode()),", "'generated_sha256':sha(source.encode()),'pipeline_source_sha256':identity['pipeline_source_sha256'],\n        'build_binding':binding,'build_binding_sha256':sha(binding_path.read_bytes()),\n        'rv_inverse':True,'rv_layout':'NZ_C0_8' if a.variant=='candidate' else 'ND','rv_proof':rv_proof,\n        'logical_input_sha256':{k:sha(natural_rv if k=='rv' else v) for k,v in inputs.items()},",1),
        ("        if host.raw!=guard+data+guard:failures.append(name+'_input_changed')", "        if name=='rv':(out/'rv_input.guarded.bin').write_bytes(host.raw)\n        if host.raw!=guard+data+guard:failures.append(name+'_input_changed')",1),
    ])
    compare=emit('cv_compare.py',[
        ('import struct\n','import struct\nimport sys\nROOT=Path(__file__).resolve().parents[3]\nif str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))\nfrom csrc.experiments.c4_b_nz.fixture import verify_rv\n',1),
        ("report['context'] not in (113,337)", "report['context']!=337",1),
        ("    hashes={}\n", "    expected_deadline=600 if 'outputs' in report else 120\n    if host.get('supervisor_returncode')!=0 or host['timeout_seconds']!=expected_deadline:raise ValueError('declared deadline/completion control differs')\n    natural=(path/'rv_natural.bin').read_bytes();physical=(path/'rv_input.bin').read_bytes()\n    if report.get('rv_inverse') is not True or report.get('rv_layout')!=('NZ_C0_8' if arm=='candidate' else 'ND'):raise ValueError('missing Rv domain proof')\n    if arm=='candidate':verify_rv(natural,physical)\n    elif physical!=natural:raise ValueError('baseline Rv is not natural')\n    guarded=(path/'rv_input.guarded.bin').read_bytes()\n    if guarded!=b'\\xa5'*64+physical+b'\\xa5'*64:raise ValueError('Rv input/guard bytes changed')\n    hashes={'logical_rv':sha(natural)}\n    if 'outputs' in report:\n        if report['input_sha256']['rv']!=sha(physical) or report['logical_input_sha256']['rv']!=sha(natural):raise ValueError('Rv saved input hash differs')\n        if report['logical_input_sha256']!={k:sha(natural) if k=='rv' else v for k,v in report['input_sha256'].items()}:raise ValueError('non-Rv input hash was excluded')\n",1),
        ("'input_sha256','oracle_sha256'", "'logical_input_sha256','oracle_sha256'",1),
        ("'source_sha256','H32_source_sha256')", "'source_sha256','H32_source_sha256','pipeline_source_sha256','build_binding','build_binding_sha256')",1),
        ("    return result\n", "    result['rv_layout']='baseline ND / candidate NZ_C0_8; logical raw bits identical'\n    result['publisher_cost']='real K/V NZ publisher included; Rv 256KiB constant prepack excluded'\n    return result\n",1),
    ])
    return {'cv_probe.cpp':probe,'cv_replay.py':replay,'cv_compare.py':compare}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    manifest={}
    for name,(text,identity) in artifacts().items():
        manifest[name]=identity;target=HERE/name
        if args.check:
            if target.read_text()!=text:raise ValueError('generated harness drift '+name)
        else:target.write_text(text)
    text=json.dumps(manifest,indent=2)+'\n';target=HERE/'entrypoints.json'
    if args.check:
        if target.read_text()!=text:raise ValueError('entrypoint manifest drift')
    else:target.write_text(text)
    print(json.dumps({'status':'checked' if args.check else 'generated','source_restore':True,'execution':'not_run'}))
