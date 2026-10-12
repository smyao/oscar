"""Archive #13-22/#126/#145/#151/#154-156, D.4: two independent writer goldens.

Reuse the existing PR byte/window/status oracle, explicitly select Hadamard.
No device run, and no inference of native service token budgets.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def export(root):
    from csrc.experiments.striped_store_simd_pack import generate
    from tools.experiments import active_identity_store_fixture as shared
    generate.check()
    binding=generate.source_binding()
    shared.torch.set_num_threads(min(4,shared.torch.get_num_threads()))
    original=inspect.getsource(shared.write_case)
    token='hadamard = mode == "decode_q4_hadamard"'
    if original.count(token)!=1:raise ValueError('fixture oracle source drift')
    # Private function globals, no edits or monkey-patches to the shared oracle.
    namespace=dict(shared.__dict__)
    exec(compile(original.replace(token,'hadamard = True'),__file__,'exec'),namespace)
    root.mkdir(parents=True,exist_ok=False)
    rows=[namespace['write_case'](root,n=n,heads=1,dim=d,dtype_name='bf16',mode=mode,block=128)
          for n,d,mode in ((128,256,'suffix_padding'),(16,64,'clip'))]
    for row in rows:
        row['writer_source_binding']=binding
        (root/row['name']/'fixture.json').write_text(json.dumps(row,indent=2)+'\n')
    report={'cases':rows,'scope':'two synthetic full-writer cases, actual binding grid, not a user dataset',
            'writer_source_binding':binding,
            'oracle_source_sha256':hashlib.sha256(original.encode()).hexdigest(),
            'only_oracle_selection_change':'explicit Hadamard for both fixed cases',
            'cann':'not_run','cpu_debug':'not_run','camodel':'not_run','npu':'not_run'}
    (root/'cases.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(json.dumps({'cases':[row['name'] for row in export(args.output)['cases']]}))
