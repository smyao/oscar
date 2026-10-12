"""Archive #132-135/#145: actual CAModel formats, ambiguity/truncation rejection.

These tests reuse saved complete instruction evidence; they do not run a
simulator or claim the old matrix micro is a decode performance measurement.
"""
from copy import deepcopy
import gzip
from pathlib import Path

import pytest

from tools.experiments import analyze_decode_completion as audit

ROOT = Path(__file__).resolve().parents[1]
ACTOR = "core0.cubecore0"
NZ = ROOT / "reports/fp32_nz_ab_matmul_20261010/reports/nz-ab-camodel/pv_m126/nz/log_ca"
DECODE = ROOT / "reports/request_window_decode_completion_20261009/q4_r32_n128_c1025_s3-candidate/instruction_samples"


def real_pair():
    return [audit.read_trace(NZ / f"{ACTOR}.{suffix}.gz", ACTOR)
            for suffix in ("instr_popped_log.dump", "instr_log.dump")]


def test_real_popped_completion_ids_keep_observed_latency_not_busy():
    paired = audit.pair_instructions(*real_pair())
    first_mmad = next(row for row in paired if row.op == "MMAD")
    assert (first_mmad.ident, first_mmad.cycle, first_mmad.complete) == (1242, 6705, 7238)
    first_fix = next(row for row in paired if row.op == "FIX_L0C_TO_DST")
    assert (first_fix.ident, first_fix.cycle, first_fix.complete) == (13, 7012, 9985)
    assert sum(row.op == "MMAD" for row in paired) == 16
    assert audit.instruction_summary([first_fix], 8000)["duration_semantics"] == (
        "queue_plus_dependencies_plus_service_not_busy")


def test_real_decode_markers_close_but_do_not_mean_math_completed():
    rows = audit.read_trace(DECODE / f"{ACTOR}.instr_log.dump.gz", ACTOR)
    scopes = audit.scalar_scopes(rows)
    qk = [scope for scope in scopes if scope["source"] == 0 and scope["field"] == 14]
    assert len(qk) == 5
    assert (qk[1]["start"], qk[1]["end"]) == (156308, 163347)
    late = next(row for row in rows if row.op == "MMAD" and row.cycle == 170668)
    assert late.cycle > qk[1]["end"]
    ready = next(row for row in rows if row.op == "SET_CROSS_CORE" and row.cycle == 171149)
    assert (ready.flag, ready.mode) == (9, 2)


@pytest.mark.parametrize("damage", ["duplicate_id", "missing_completion", "wrong_pc", "flag_count"])
def test_actual_pair_rejects_ambiguity_and_missing_witness(damage):
    popped, completed = real_pair()
    i = next(i for i, row in enumerate(completed) if row.op == "MMAD")
    if damage == "duplicate_id":
        popped.append(deepcopy(next(row for row in popped if row.op == "MMAD")))
        completed.append(deepcopy(completed[i]))
    elif damage == "missing_completion":
        completed.pop(i)
    elif damage == "wrong_pc":
        completed[i].pc += 4
    else:
        completed.pop(next(i for i, row in enumerate(completed) if row.op == "WAIT_FLAG"))
    with pytest.raises(audit.TraceError):
        audit.pair_instructions(popped, completed)


def test_truncated_real_line_and_unclosed_real_markers_fail(tmp_path):
    path = DECODE / f"{ACTOR}.instr_log.dump.gz"
    with gzip.open(path, "rt") as stream:
        first = next(stream)
    truncated = tmp_path / "trace.txt"
    truncated.write_text(first[:-1])
    with pytest.raises(audit.TraceError, match="trailing partial"):
        audit.read_trace(truncated, ACTOR)
    rows = audit.read_trace(path, ACTOR)
    end = next(i for i, row in enumerate(rows) if row.marker == (3, 23, True))
    with pytest.raises(audit.TraceError, match="truncated"):
        audit.scalar_scopes(rows[:end])
