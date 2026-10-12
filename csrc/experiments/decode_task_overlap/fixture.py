"""Bounded cross-request overlap experiment, not a user dataset.

Archive #126/#140/#145/#148/#154-156 and startup D.4: retain the current
INT2 KV256 math, all original task owners/sources, and frozen PR oracle.
Two different requests exercise overlap without assuming their KV matches.
CPU/CAModel results do not certify target NPU/graph/model performance.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from csrc.experiments.window_batch_decode import fixture as shared
from tools.striped_fixture import raw_to_striped


def export(directory: Path, group: str):
    cases = {
        "smoke": tuple(
            shared.Case(f"{mode}_pair_c641_701", mode, (641, 701),
                        padded=8 if mode == "q4" else 4, cores=1)
            for mode in ("q4", "q1")
        ),
        "odd": tuple(
            shared.Case(f"{mode}_triple_c641_897_1153", mode, (641, 897, 1153),
                        padded=16 if mode == "q4" else 4, cores=1)
            for mode in ("q4", "q1")
        ),
        "long": tuple(
            shared.Case(f"{mode}_pair_c20000_20100", mode, (20000, 20100),
                        padded=8 if mode == "q4" else 4, cores=1)
            for mode in ("q4", "q1")
        ),
        "steady": tuple(
            shared.Case(f"{mode}_pair_c8193_8225", mode, (8193, 8225),
                        padded=8 if mode == "q4" else 4, cores=1)
            for mode in ("q4", "q1")
        ),
        "fault": tuple(
            shared.Case(f"{mode}_pair_distinct_errors", mode, (641, 701),
                        padded=8 if mode == "q4" else 4, cores=1,
                        fault="paired_future_scale_and_query_nan")
            for mode in ("q4", "q1")
        ),
    }[group]
    old_plan, old_make, old_case = shared.case_plan, shared.reuse.make_fixture, shared.make_case

    def make(torch, shape, **kwargs):
        return old_make(torch, shape, **kwargs,
                        rotation_mode="identity" if shape.slot_context else "hadamard")

    def make_case(torch, case):
        result = old_case(torch, replace(case, fault=None))
        if case.fault:
            # Request0's future K tile has invalid scale; request1 has a
            # nonfinite Q. These must not migrate between active contexts.
            pos = 350
            physical = result["page_assignments"][0][pos // shared.reuse.BLOCK_TOKENS]
            slot = (shared.reuse.PREFIX + physical * result["stride"] +
                    (pos % shared.reuse.BLOCK_TOKENS) * 136)
            data = result["cpu"]
            data["raw"][slot + 64:slot + 66] = torch.tensor([0.], dtype=torch.float16).view(torch.uint8)
            qlen = 4 if case.mode == "q4" else 1
            data["qr"][qlen, 0, 0] = float("nan")
            data["raw_striped"] = raw_to_striped(result)
            data["required_status_checks"] = torch.tensor(
                [(0, 0, 3), (qlen * 3, 0, 2)], dtype=torch.int64)
        result["case"] = case
        return result

    try:
        shared.case_plan = lambda *_args: cases
        shared.reuse.make_fixture = make
        shared.make_case = make_case
        result = shared.export(directory, cores=1)
    finally:
        shared.case_plan, shared.reuse.make_fixture = old_plan, old_make
        shared.make_case = old_case
    result["baseline"] = "current_production_attention_cv_decode_bundle.cpp"
    result["candidate"] = "independent_same_owner_cross_request_overlap"
    result["scope"] = "complete_one_core_mechanism_pair_not_K32_or_service"
    import json
    (directory / "cases.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", choices=("smoke", "odd", "long", "steady", "fault"), default="smoke")
    args = parser.parse_args()
    export(args.output, args.group)
