"""q4 owner tasks with one/two/three live queries and an internal slot gap.

Frozen valid-row oracle remains the original dense causal result; only the
explicit inactive query rows are zero/-inf. Real task preparation remains in
the official CPU harness. This exercises capture/replay-style shape changes.
"""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from csrc.experiments.window_batch_decode import fixture as shared

MASKS={'q4_live1':(1,2,3),'q4_live2':(2,3),'q4_live3':(3,), 'q4_internal_gap':(1,)}
ORIGINAL_MAKE=shared.make_case


def cases(_large=False,_long_single=False):
    return tuple(shared.Case(name,'q4',(511,)) for name in MASKS)


def make(torch,case):
    fixture=ORIGINAL_MAKE(torch,case);cpu=fixture['cpu']
    for token in MASKS[case.name]:
        cpu['slots'][token]=-1
        cpu['expected_positions'][token]=-1
        cpu['expected_output'][token]=0
        cpu['expected_lse'][token]=-float('inf')
    return fixture


def export(path):
    old_plan,old_make=shared.case_plan,shared.make_case
    try:
        shared.case_plan,shared.make_case=cases,make
        return shared.export(path,cores=2)
    finally:
        shared.case_plan,shared.make_case=old_plan,old_make


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    export(parser.parse_args().output)
