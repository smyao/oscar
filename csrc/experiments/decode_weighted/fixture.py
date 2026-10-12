"""Archive #129/#140/#145/#155: full task coverage at the planner's N128 shape.

Synthetic multi-request q4/q1 plus invalid-window domains, independent PR
oracle. These short CPU cases test ownership, not 20-30K performance.
"""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.window_batch_decode import fixture as shared

def cases(*_):
    return (
        shared.Case('q4_weighted_n128','q4',(1025,2049,829,4097),padded=128,splits=3),
        shared.Case('q1_weighted_n128','q1',(1025,2049,829,4097),padded=128,splits=3),
        shared.Case('q4_weighted_bad_tag_n128','q4',(511,),padded=128,fault='bad_tag'),
        shared.Case('q1_weighted_bad_page_n128','q1',(511,),padded=128,fault='bad_page'),
    )

def export(path):
    old_plan,old_make=shared.case_plan,shared.reuse.make_fixture
    def make(torch,shape,**kwargs):
        return old_make(torch,shape,**kwargs,
            rotation_mode='identity' if shape.slot_context else 'hadamard')
    try:
        shared.case_plan,shared.reuse.make_fixture=cases,make
        return shared.export(path,cores=2)
    finally:
        shared.case_plan,shared.reuse.make_fixture=old_plan,old_make

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    export(p.parse_args().output)
