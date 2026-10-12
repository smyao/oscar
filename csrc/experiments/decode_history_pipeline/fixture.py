"""Archive #126/#145/#154/#155: exercise three-tile reuse and future errors.

Reuses the current-SIMD independent fixtures. Diagnostic corruptions never
enter production or masquerade as finite-store numerical acceptance.
"""
from dataclasses import replace
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from csrc.experiments.window_batch_decode import fixture as shared
from tools.striped_fixture import raw_to_striped

ORIGINAL_MAKE = shared.make_case


def cases(_include_large=False,_include_long_single=False):
    return (
        shared.Case("q4_one_tile_511","q4",(511,)),
        shared.Case("q4_three_tiles_829","q4",(829,)),
        shared.Case("q1_three_tiles_832","q1",(832,)),
        shared.Case("q4_three_tiles_1025","q4",(1025,)),
        shared.Case("q1_seven_tiles_2049","q1",(2049,)),
        shared.Case("q4_multi_request_s3","q4",(2049,1025),padded=16,splits=3),
        shared.Case("q1_multi_request_s3","q1",(1025,2049,829),splits=3),
        shared.Case("q4_future_bad_scale","q4",(829,),fault="future_scale"),
        shared.Case("q4_score_before_future_scale","q4",(829,),fault="score_before_scale"),
        shared.Case("q4_dead_tail_scale","q4",(829,)),
    )


def _slot(fixture, position):
    physical=fixture["page_assignments"][0][position//shared.reuse.BLOCK_TOKENS]
    off=shared.reuse.PREFIX+physical*fixture["stride"]+(position%shared.reuse.BLOCK_TOKENS)*136
    return fixture["cpu"]["raw"][off:off+136]


def make_case(torch, case):
    fixture=ORIGINAL_MAKE(torch,replace(case,fault=None))
    data=fixture["cpu"]
    half=lambda values:torch.tensor(values,dtype=torch.float16).view(torch.uint8)
    if case.fault=="score_before_scale":
        # All raw values except tile0 are zero with valid positive scale.
        # Tile0's finite equal scores have an overflowing pre-mask sum, but
        # finite softmax statistics. Future tile1's bad scale must win last.
        # QR is deliberately diagnostic, not a model-generated tensor.
        for position in range(64,577):
            slot=_slot(fixture,position)
            slot.zero_();slot[64:68]=half((1.,0.));slot[132:136]=half((1.,0.))
            if position<320:
                slot[:64].fill_(0x22)  # levels 2,0,2,0, matching alternating QR
                slot[64:68]=half((32768.,-32768.))
        data["qr"][0,0,::2]=1.e31
        data["qr"][0,0,1::2]=-1.e31
    if case.fault in ("future_scale","score_before_scale"):
        _slot(fixture,350)[64:66]=half((0.,))
        data["required_status_checks"]=torch.tensor(((0,0,3),),dtype=torch.int64)
    elif case.name=="q4_dead_tail_scale":
        # History ends at577 exclusive; precise window is independent of
        # packed bytes at577. The existing independent oracle remains valid.
        _slot(fixture,577)[64:66]=half((float('nan'),))
    data["raw_striped"]=raw_to_striped(fixture)
    fixture["case"]=case
    return fixture


def export(path, cores=2):
    old_plan,old_make=shared.case_plan,shared.make_case
    try:
        shared.case_plan,shared.make_case=cases,make_case
        return shared.export(path,cores=cores)
    finally:
        shared.case_plan,shared.make_case=old_plan,old_make


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cores',type=int,default=2)
    args=parser.parse_args()
    export(args.output,args.cores)
