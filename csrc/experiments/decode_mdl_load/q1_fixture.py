"""Archive #126/#140/#145/#148/#154-157, D.4: complete q1 MDL goldens.

Offline independent PR oracle only. Reuse the existing fixture, quantizer,
physical-page permutation, exact windows and frozen tolerances. No serving
route or kernel changes; include all sources, padding, tails and error order.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from csrc.experiments.window_batch_decode import fixture as shared
from csrc.experiments.decode_mdl_load.run import Q1_CASE
from tools.striped_fixture import raw_to_striped

FAULT_PAIR = "q1_mdl_future_scale_query_nan"
CASES = (
    shared.Case(Q1_CASE, "q1", (1344, 1376), padded=8, cores=1),
    *(shared.Case("q1_mdl_" + fault, "q1", (511,), padded=8, cores=1, fault=fault)
      for fault in ("bad_page", "bad_tag", "vnan_then_tag", "tag_then_vnan")),
    shared.Case(FAULT_PAIR, "q1", (641, 701), padded=8, cores=1,
                fault="paired_future_scale_and_query_nan"),
)


def export(directory):
    directory = Path(directory)
    if directory.exists():
        raise ValueError("refuse to overwrite q1 fixture")
    old_plan, old_case = shared.case_plan, shared.make_case

    def make_case(torch, case):
        if case.name != FAULT_PAIR:
            return old_case(torch, case)
        result = old_case(torch, replace(case, fault=None))
        # Existing cross-request overlap fixture's error witness: request0's
        # future history tile has zero K scale; request1 has nonfinite Q.
        # Their errors must stay with their original request/owner.
        position = 350
        physical = result["page_assignments"][0][position // shared.reuse.BLOCK_TOKENS]
        slot = shared.reuse.PREFIX + physical * result["stride"] + position % shared.reuse.BLOCK_TOKENS * 136
        data = result["cpu"]
        data["raw"][slot + 64:slot + 66] = torch.tensor([0.], dtype=torch.float16).view(torch.uint8)
        data["qr"][1, 0, 0] = float("nan")
        data["raw_striped"] = raw_to_striped(result)
        data["required_status_checks"] = torch.tensor([(0, 0, 3), (3, 0, 2)], dtype=torch.int64)
        result["case"] = case
        return result

    try:
        shared.case_plan = lambda *_: CASES
        shared.make_case = make_case
        manifest = shared.export(directory, cores=1)
    finally:
        shared.case_plan, shared.make_case = old_plan, old_case
    work = manifest["cases"][0]["source_work"]
    if work["kv256_units"] != {"history": 10, "window": 4, "current": 2} or work["task_slots"] != 24:
        raise ValueError("complete q1 source/tail domain drift")
    manifest.update(baseline="current production q1 decode bundle, Normal B loader",
                    candidate="existing same-library q1 MDL symbol; unchanged kernel source",
                    measurement_scope="complete R2/q1/N8/C1344,1376/S1/1Cube+2AIV; not K32/NPU/service",
                    rotation="original nontrivial orthogonal fixture; no identity substitution",
                    minimum_local_speedup=1.15)
    (directory / "cases.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    export(parser.parse_args().output)
