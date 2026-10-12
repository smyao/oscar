"""Complete long-history diagnostic shapes for current SIMD decode.

Archive #126/#129/#145/#148/#154/#155 and startup D.4: actual D256,
Hq6/Hkv1, N128, S3 and 20 Cube cores. One live request deliberately isolates
each long task, while all padding and all three sources still execute.
This is NOT a 32-request performance substitute. Main uses Hadamard and MTP
uses identity, matching RuntimeProvider. No user dataset is generated.
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from csrc.experiments.window_batch_decode import fixture as shared


def cases(_include_large=False,_include_long_single=False):
    return tuple(shared.Case(f"{mode}_r1_n128_c{context}_s3", mode,
                            (context,), padded=128, splits=3)
                 for context in (20000, 30000) for mode in ("q4", "q1"))


def export(path):
    old_plan, old_make = shared.case_plan, shared.reuse.make_fixture
    def make(torch, shape, **kwargs):
        # Query length is a fixed fixture label, never a runtime predicate.
        return old_make(torch, shape, **kwargs,
                        rotation_mode="identity" if shape.slot_context else "hadamard")
    try:
        shared.case_plan, shared.reuse.make_fixture = cases, make
        return shared.export(path, cores=20)
    finally:
        shared.case_plan, shared.reuse.make_fixture = old_plan, old_make


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.output)
