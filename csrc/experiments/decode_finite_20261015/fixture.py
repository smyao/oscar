"""Archive #126/#140/#145/#148/#154-157: original independent PR fixtures.

Two production families, same full metadata protocol; q4 passes the existing
NZ rotation representation to BOTH arms. Oracle still uses natural Rv.
"""
from __future__ import annotations
import argparse,json,sys
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.window_batch_decode import fixture as shared
from tools.striped_fixture import raw_to_striped
from csrc.experiments.decode_finite_20261015.generate import digest

CASES=tuple(shared.Case(f'{f}_finite_{name}',f,contexts,padded=8,cores=1,fault=fault)
    for f in ('q4','q1') for name,contexts,fault in (
       ('cpu',(641,701),None),('query_metadata',(641,701),'query_validation_next_metadata'),
       ('long',(1344,1376),None)))

def export(path):
    import torch
    if path.exists():raise ValueError('fresh fixture root required')
    old_plan,old_case=shared.case_plan,shared.make_case
    def make(torch,case):
        result=old_case(torch,replace(case,fault=None));data=result['cpu']
        if case.fault:
            position=350
            physical=result['page_assignments'][0][position//shared.reuse.BLOCK_TOKENS]
            offset=shared.reuse.PREFIX+physical*result['stride']+position%shared.reuse.BLOCK_TOKENS*136
            data['raw'][offset+64:offset+66]=torch.tensor([0.],dtype=torch.float16).view(torch.uint8)
            data['qr'][0,0,0]=float('nan')
            # Same history task/lane, independently diagnosed original chain:
            # LoadQueries sum=NaN =>2, actual CPU QK row sums finite,
            # post-PV pending K metadata=>3; final score/normalization finite.
            # Do not describe this full fixture as a nonfinite-score witness.
            data['required_status_checks']=torch.tensor([(0,0,3)],dtype=torch.int64)
            data['raw_striped']=raw_to_striped(result)
        rv=data['rv'];data['rv_nz']=rv.reshape(256,32,8).permute(1,0,2).contiguous().view(256,256)
        inverse=data['rv_nz'].reshape(32,256,8).permute(1,0,2).contiguous().view(256,256)
        if not torch.equal(rv.view(torch.int32),inverse.view(torch.int32)):raise ValueError('NZ inverse bits')
        result['case']=case
        return result
    try:
        shared.case_plan=lambda *_:CASES;shared.make_case=make
        manifest=shared.export(path,cores=1)
    finally:shared.case_plan,shared.make_case=old_plan,old_case
    for case in manifest['cases']:
        if case['name'].endswith('_long') and case['source_work']['kv256_units']!={'history':10,'window':4,'current':2}:
            raise ValueError('complete multi-KV domain changed')
    manifest.update(generator_sha256=digest(__file__),baseline='current q4 hybrid/q1 MDL respectively',
        candidate='same family finite result batching only',q4_rotation='both arms same exact NZ permutation',
        minimum_speedup=1.0,performance_scope='complete R2 N8 C1344/1376 S1; not model or NPU')
    (path/'cases.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);export(p.parse_args().output)
