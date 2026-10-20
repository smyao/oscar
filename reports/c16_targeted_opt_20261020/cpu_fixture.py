"""Archive #126/#145/#148/#164: input-driven, mode-0 C16 correctness fixtures.

Complete D.4 read: only offline tiny functional cases are constructed; no
full-history materialization enters the runtime and no speed claim is made.
The existing v3 reference receives actual native prepare tasks, never outputs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class Case:
    name: str
    query: int
    context: int
    padding: int = 1
    heads: int = 1
    splits: int = 1
    recent: int = 1
    bad_metadata: int = -1
    bad_qr: int = -1
    qr_value: str = "nan"
    dead_metadata: bool = False


# Cases are fixed before running the candidate. Mature count and boundary
# differences come from public task geometry, not any candidate output.
CASES = (
    Case("mature_h32_tail", 337, 96),
    Case("mature_multiple_kv_tiles", 337, 577),
    Case("independent_multiple_kv_tiles", 22, 577),
    Case("hkv2_s3_window_current", 5, 337, padding=3, heads=2, splits=3, recent=256),
    Case("cold_second_score_block", 6, 0, padding=2),
    Case("empty_history_window", 1, 64, recent=256),
    Case("metadata_lane1", 337, 96, bad_metadata=80),
    Case("qr_last_member_nan", 337, 96, bad_qr=315),
    Case("qr_last_member_inf", 337, 96, bad_qr=315, qr_value="inf"),
    Case("unread_dead_metadata", 22, 577, dead_metadata=True),
    # Exercise the newly selected full-KV256 finite path with real CV errors,
    # beyond the independent helper's row-wise nonfinite matrix.
    Case("full_tile_qr_nan", 22, 577, bad_qr=0),
    Case("full_tile_metadata_lane1", 22, 577, bad_metadata=80),
)


def dump(path, tensor):
    import ctypes
    contiguous = tensor.contiguous()
    raw = ctypes.string_at(contiguous.data_ptr(), contiguous.numel() * contiguous.element_size())
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def make(folder, selected):
    import torch
    from tools import probe_history_reuse as reuse
    from tools.aligned_fixture import make_fixture
    torch.set_num_threads(1)
    folder.mkdir(parents=True, exist_ok=False)
    cases = []
    for case in CASES:
        if selected and case.name not in selected:
            continue
        path = folder / case.name
        path.mkdir()
        spec = reuse.Shape(case.name, (case.query,), (case.context,), 256,
                           case.heads, case.splits, case.query >= 336,
                           speed_gate=False, recent_tokens=case.recent,
                           padded_tokens=case.query + case.padding,
                           corrupt_metadata=case.bad_metadata >= 0,
                           corrupt_position=max(0, case.bad_metadata),
                           corrupt_dead_tail=case.dead_metadata,
                           corrupt_qr=case.bad_qr >= 0,
                           corrupt_qr_token=max(0, case.bad_qr))
        fixture = make_fixture(torch, spec)
        data = fixture["cpu"]
        if case.bad_qr >= 0:
            data["qr"][case.bad_qr, 0, 0] = float(case.qr_value)
        n = data["q"].shape[0]
        # Padded queries and current inputs are poison; they must never be read.
        for key in ("q", "qr", "ck", "cv"):
            data[key][case.query:] = float("nan")
        positions = torch.full((n,), -1, dtype=torch.int64)
        positions[:case.query] = torch.arange(case.context, case.context + case.query)
        data["expected_positions"] = positions
        checks = []
        if case.bad_metadata >= 0:
            # Fixed cases use one head/split and history begins at sink=64.
            lane = ((case.bad_metadata - 64) % 32) // 16
            checks = [(qb * 3, lane, 3) for qb in range(0, case.query, 21)]
        if case.bad_qr >= 0:
            leader = case.bad_qr // 21 * 21
            lane = ((case.bad_qr % 21) * 6) // 64
            checks = [(leader * 3, lane, 2)]
        data["required_status_checks"] = torch.tensor(checks, dtype=torch.int64).reshape(-1, 3)
        torch.save(fixture, path / "fixture.pt")
        hashes = {name: dump(path / (name + ".bin"), value) for name, value in data.items()}
        dump(path / "raw_striped.bin", data["raw"])
        shape = (n, spec.heads, case.heads, 256, case.context, 64, case.recent, 3,
                 reuse.BLOCK_TOKENS, fixture["blocks"], reuse.PREFIX, fixture["stride"],
                 4, 1, case.splits, data["table"].shape[1])
        (path / "shape.txt").write_text(" ".join(map(str, shape)) + "\n")
        invalid = bool(checks)
        # Current long-prefill contributions belong to native FIA; this is the
        # unchanged C16 scope. Short cases cover all three sources directly.
        suppress = int(case.query >= 336)
        clusters = 1 if suppress else 0
        (path / "case_flags.txt").write_text(f"{int(invalid)}\n")
        (path / "cluster_flags.txt").write_text(
            f"{suppress} {clusters} {case.bad_metadata} {case.bad_qr}\n")
        (path / "expected_counts.txt").write_text(
            f"{((case.query + 20) // 21 - 16 * clusters) * case.heads * case.splits}\n")
        item = {**case.__dict__, "dequant_mode": 0, "input_sha256": hashes,
                "fault": invalid, "source2_suppressed": bool(suppress),
                "oracle": "pending_actual_prepare_tasks", "NPU": "not_run",
                "performance": "not_run"}
        (path / "case.json").write_text(json.dumps(item, indent=2) + "\n")
        cases.append(item)
    if not cases:
        raise ValueError("no declared case selected")
    (folder / "cases.json").write_text(json.dumps({"cases": cases}, indent=2) + "\n")


def golden(folder, output):
    import torch
    from tools.aligned_reference import fixture_tasks_reference
    torch.set_num_threads(1)
    fixture = torch.load(folder / "fixture.pt", weights_only=False)
    info = json.loads((folder / "case.json").read_text())
    tasks = torch.frombuffer(bytearray((output / "tasks.bin").read_bytes()),
                             dtype=torch.int64).reshape(-1, 16).clone()
    dump(output / "rv_bf16.bin", fixture["cpu"]["rv"])
    if info["fault"]:
        # Invalid inputs have exact explicit status gates and complete
        # unchanged-kernel bitwise checks, never a numerical oracle bypass.
        return
    results = fixture_tasks_reference(fixture, tasks, 0)
    for key in ("partial", "lse", "output", "output_lse"):
        dump(output / ("bf16_expected_" + key + ".bin"), results[key])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("make", "golden"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", action="append", choices=[case.name for case in CASES])
    args = parser.parse_args()
    if args.action == "make":
        make(args.folder, args.case)
    else:
        if args.output is None:
            parser.error("--output is required for golden")
        golden(args.folder, args.output)


if __name__ == "__main__":
    main()
