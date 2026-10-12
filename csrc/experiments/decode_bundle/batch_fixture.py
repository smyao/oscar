"""Synthetic concurrent 20/23/27/30K operator fixtures, not user data.

Keep actual production N128/S3/Hq6/Hkv1/D256 and 20 Cube cores, all tasks
and sources. Main q4 uses Hadamard; later MTP q1 uses identity. This prepares
the complete cohort, never replicates a single-query latency as batch time.
"""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from csrc.experiments.window_batch_decode import fixture as shared


def export(path,requests):
    if requests not in (4,32):raise ValueError('diagnostic cohort must be4 or32 requests')
    contexts=(20000,23000,27000,30000)*(requests//4)
    def cases(_large=False,_long_single=False):
        return tuple(shared.Case(f'{mode}_r{requests}_n128_20to30k_s3',mode,contexts,
                                 padded=128,splits=3,cores=20) for mode in ('q4','q1'))
    old_plan,old_make=shared.case_plan,shared.reuse.make_fixture
    def make(torch,shape,**kwargs):
        return old_make(torch,shape,**kwargs,
            rotation_mode='identity' if shape.slot_context else 'hadamard')
    try:
        shared.case_plan,shared.reuse.make_fixture=cases,make
        return shared.export(path,cores=20)
    finally:
        shared.case_plan,shared.reuse.make_fixture=old_plan,old_make


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--requests',type=int,choices=(4,32),required=True)
    args=parser.parse_args();export(args.output,args.requests)
