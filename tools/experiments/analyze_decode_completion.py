"""Archive #129/#132-135/#145/#148/#154-156; startup D.4: decode attribution.

Read complete paired CAModel popped/completion traces for the pinned d817
bundle. Popped-to-completion includes queueing and dependencies, never pure
busy time. Scalar markers describe method submission, not asynchronous finish.
Cube results are bounded by the original in-order FIXP cross-core ready flags.
No kernel changes, added fences, NPU claim, or cross-actor duration summation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_SHA = "d8171f0403cf176087e1c76c64e454e6fb1e74fd79252aab3b59e848cca19401"
LINE = re.compile(r"^\[info\]\s+\[(\d+)\]\s+\(PC: (0x[0-9a-fA-F]+)\)\s+"
                  r"(\w+)\s+:\s+\(Binary: (0x[0-9a-fA-F]+)\)\s+(\w+)\s*(.*)$")
ID = re.compile(r"\bid:\s*(\d+)\b")
XD = re.compile(r"\bXD:X\d+=(0x[0-9a-fA-F]+|\d+)\b")
XT = re.compile(r"\bXT:X\d+=(0x[0-9a-fA-F]+|\d+)\b")
UIMM = re.compile(r"\bUIMM:(0x[0-9a-fA-F]+|\d+)\b")
FIELDS = {3: "task", 9: "aiv_qk_wait", 11: "aiv_pv_wait", 13: "aic_kv_wait",
          14: "qk", 15: "aic_p_wait", 16: "pv", 18: "rotation", 23: "compute"}


class TraceError(ValueError):
    pass


@dataclass
class Instruction:
    actor: str
    cycle: int
    pc: int
    pipe: str
    op: str
    operands: str
    sequence: int
    ident: int | None = None
    marker: tuple[int, int, bool] | None = None
    flag: int | None = None
    mode: int | None = None
    complete: int | None = None
    context: dict = field(default_factory=dict)


def parse_line(line: str, actor: str, sequence: int) -> Instruction:
    match = LINE.fullmatch(line.rstrip("\n"))
    if not match:
        raise TraceError(f"{actor}:{sequence}: unsupported/truncated instruction line")
    cycle, pc, pipe, _binary, op, operands = match.groups()
    found = ID.search(operands)
    row = Instruction(actor, int(cycle), int(pc, 16), pipe, op, operands,
                      sequence, int(found[1]) if found else None)
    if op == "MOV_XD_XN" and (value := XD.search(operands)):
        tag = int(value[1], 0)
        if tag >> 32 == 0x5A77C01D:
            if tag & 0xFFFFFC00 or not 3 <= ((tag & 255) >> 1) <= 23:
                raise TraceError("unknown marker tag bits")
            row.marker = ((tag >> 8) & 3, (tag & 255) >> 1, bool(tag & 1))
    if op == "WAIT_FLAG_DEVI":
        value = UIMM.search(operands)
        if not value:
            raise TraceError("cross-core wait has no witnessed UIMM")
        row.flag = int(value[1], 0)
    if op == "SET_CROSS_CORE":
        value = XT.search(operands)
        if not value:
            raise TraceError("cross-core set has no witnessed XT register value")
        encoding = int(value[1], 0)
        # Official A2 GetffstMsg: 1 + ((mode & 3) << 4) + ((flag & 15) << 8).
        if encoding & ~0xF31 or not encoding & 1:
            raise TraceError("unrecognized A2 cross-core encoding")
        row.flag, row.mode = (encoding >> 8) & 15, (encoding >> 4) & 3
    return row


def read_trace(path: Path, actor: str) -> list[Instruction]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt") as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                raise TraceError(f"{actor}: trailing partial line")
            row = parse_line(line, actor, number)
            if relevant(row):
                rows.append(row)
    if not rows:
        raise TraceError(f"{actor}: empty trace")
    return rows


def relevant(row: Instruction) -> bool:
    return row.ident is not None or row.marker is not None or row.op in {
        "SET_FLAG", "WAIT_FLAG", "WAIT_FLAG_DEVI", "SET_CROSS_CORE"}


def matching_key(row: Instruction):
    if row.ident is not None:
        return row.actor, row.pipe, row.op, "id", row.ident
    # No global instruction ID exists for these events. Keep PC, pipe,
    # marker/flag and the full witnessed operands, then use dynamic ordinal.
    return row.actor, row.pipe, row.op, "ordinal", row.pc, row.marker, row.flag, row.operands


def pair_instructions(popped: list[Instruction], completed: list[Instruction]):
    groups = []
    for rows in (popped, completed):
        table = defaultdict(list)
        for row in rows:
            if relevant(row):
                table[matching_key(row)].append(row)
        groups.append(table)
    left, right = groups
    if set(left) != set(right):
        raise TraceError("popped/completion keys differ: trace truncated or ambiguous")
    for key, starts in left.items():
        ends = right[key]
        if len(starts) != len(ends) or (key[3] == "id" and len(starts) != 1):
            raise TraceError(f"ambiguous ID/ordinal count for {key}")
        for start, end in zip(starts, ends):
            if start.pc != end.pc or end.cycle < start.cycle:
                raise TraceError(f"PC or completion order mismatch for {key}")
            start.complete = end.cycle
    return [row for row in popped if relevant(row)]


def scalar_scopes(rows: list[Instruction]):
    """Only scalar marker order establishes task/source and method order."""
    stack, scopes = [], []
    task_ordinal = -1
    for row in rows:
        if row.marker is not None:
            source, index, end = row.marker
            if not end:
                if index == 3:
                    task_ordinal += 1
                if not stack and (source, index) != (3, 23):
                    raise TraceError("trace does not begin with compute marker")
                scope = {"source": source, "field": index, "task_ordinal": task_ordinal,
                         "start": row.cycle, "start_sequence": row.sequence}
                stack.append(scope)
            else:
                if not stack or (stack[-1]["source"], stack[-1]["field"]) != (source, index):
                    raise TraceError("unclosed/ambiguous scalar marker nesting")
                scope = stack.pop()
                scope.update(end=row.cycle, end_sequence=row.sequence)
                scopes.append(scope)
        row.context = {"source": stack[-1]["source"], "task_ordinal": task_ordinal,
                       "fields": [item["field"] for item in stack]} if stack else {}
    if stack or sum(item["field"] == 23 for item in scopes) != 1:
        raise TraceError("truncated or multiple compute spans")
    return sorted(scopes, key=lambda item: item["start_sequence"])


def instruction_summary(rows: list[Instruction], wrapper_end: int):
    return {"count": len(rows),
            "first_popped": min((row.cycle for row in rows), default=None),
            "last_complete": max((row.complete for row in rows), default=None),
            "complete_after_wrapper_count": sum(row.complete > wrapper_end for row in rows),
            "pop_to_complete_sum": sum(row.complete - row.cycle for row in rows),
            "duration_semantics": "queue_plus_dependencies_plus_service_not_busy"}


def cube_methods(rows: list[Instruction], scopes):
    methods = [scope for scope in scopes if scope["field"] in (14, 16, 18)]
    ready = [row for row in rows if row.op == "SET_CROSS_CORE" and row.pipe == "FIXP"]
    if len(methods) != len(ready):
        raise TraceError("Matmul method/ready count mismatch; cannot assign asynchronous work")
    all_math = [row for row in rows if row.op in ("MMAD", "FIX_L0C_TO_DST")]
    result, previous_complete, assigned = [], -1, set()
    for method, flag in zip(methods, ready):
        expected = {14: 9, 16: 11, 18: 13}[method["field"]] if method["source"] == 0 else 9
        if flag.flag != expected or flag.mode != 2 or flag.complete < method["start"]:
            raise TraceError("Matmul/ready order differs from pinned d817 protocol")
        # This source ends each Matmul with FIXP->CUBE ownership and emits one
        # FIXP ready. Both pipes are ordered. This is a completion delimiter,
        # NOT attribution by the wrapper's scalar end time.
        owned = [row for row in all_math if previous_complete < row.complete <= flag.complete]
        if not owned or any(row.cycle < method["start"] for row in owned):
            raise TraceError("ambiguous Matmul instruction boundary")
        for row in owned:
            assigned.add(row.sequence)
        result.append({**method, "phase": FIELDS[method["field"]],
                       "submission_span": method["end"] - method["start"],
                       "ready_complete": flag.complete,
                       "delivery_span": flag.complete - method["start"],
                       "tail_after_wrapper": max(0, flag.complete - method["end"]),
                       "mmad": instruction_summary([r for r in owned if r.op == "MMAD"], method["end"]),
                       "fix": instruction_summary([r for r in owned if r.op == "FIX_L0C_TO_DST"], method["end"])})
        previous_complete = flag.complete
    if assigned != {row.sequence for row in all_math}:
        raise TraceError("unassigned MMAD/FIX instructions")
    return result


def cross_dependencies(actors: dict[str, list[Instruction]]):
    result = []
    for actor, rows in actors.items():
        core, engine = actor.split(".", 1)
        for flag in ((8, 10) if engine == "cubecore0" else (9, 11)):
            waits = [row for row in rows if row.op == "WAIT_FLAG_DEVI" and row.flag == flag]
            producers = [f"{core}.veccore0", f"{core}.veccore1"] if engine == "cubecore0" else [f"{core}.cubecore0"]
            sets = [[row for row in actors[name] if row.op == "SET_CROSS_CORE" and row.flag == flag]
                    for name in producers]
            if any(len(items) != len(waits) for items in sets):
                raise TraceError(f"{actor} flag{flag} set/wait count mismatch")
            if any(row.mode != 2 for items in sets for row in items):
                raise TraceError("cross-core producer is not pinned mode2 protocol")
            for ordinal, wait in enumerate(waits):
                producer_complete = max(items[ordinal].complete for items in sets)
                if producer_complete > wait.complete:
                    raise TraceError("wait completes before required producers")
                missing = max(0, producer_complete - wait.cycle)
                result.append({"actor": actor, "flag": flag, "ordinal": ordinal,
                               **wait.context, "wait_popped": wait.cycle, "wait_complete": wait.complete,
                               "producer_ready_complete": producer_complete,
                               "observed_producer_not_ready_ticks": missing,
                               "remaining_wait_queue_or_protocol_unknown_ticks": wait.complete - wait.cycle - missing,
                               "not_inference": "not removable wait or hardware busy time"})
    return result


def analyze_files(files: dict[str, tuple[Path, Path]]):
    actors, scope_map = {}, {}
    for actor, (popped_path, completed_path) in files.items():
        actors[actor] = pair_instructions(read_trace(popped_path, actor), read_trace(completed_path, actor))
        scope_map[actor] = scalar_scopes(actors[actor])
    expected = {f"core0.{kind}" for kind in ("cubecore0", "veccore0", "veccore1")}
    if set(actors) != expected:
        raise TraceError("this bounded diagnostic requires exactly one Cube/two AIV actors")
    methods = cube_methods(actors["core0.cubecore0"], scope_map["core0.cubecore0"])
    rounds, current = [], None
    for method in methods:
        if method["source"] != 0 or method["phase"] == "rotation":
            continue
        if method["phase"] == "qk":
            if current is not None:
                raise TraceError("QK without preceding PV completion")
            current = {"task_ordinal": method["task_ordinal"], "source": "history", "qk": method}
        else:
            if current is None or current["task_ordinal"] != method["task_ordinal"]:
                raise TraceError("PV without same-task QK")
            current["pv"] = method
            current["kv_round"] = sum(r["task_ordinal"] == current["task_ordinal"] for r in rounds)
            rounds.append(current)
            current = None
    if current is not None or not rounds:
        raise TraceError("missing history round completion")
    dependencies = cross_dependencies(actors)
    by_task_flag = defaultdict(list)
    for dependency in dependencies:
        if dependency.get("source") == 0:
            by_task_flag[dependency["actor"], dependency["task_ordinal"], dependency["flag"]].append(dependency)
    for row in rounds:
        row["dependencies"] = []
        for actor, flags in (("core0.cubecore0", (8, 10)),
                             ("core0.veccore0", (9, 11)), ("core0.veccore1", (9, 11))):
            for flag in flags:
                values = by_task_flag[actor, row["task_ordinal"], flag]
                count = sum(r["task_ordinal"] == row["task_ordinal"] for r in rounds)
                if len(values) != count:
                    raise TraceError("history round/flag count differs across actors")
                row["dependencies"].append(values[row["kv_round"]])
    return {"status": "complete_bounded_instruction_attribution", "source_sha256": SOURCE_SHA,
            "history_rounds": rounds, "cross_dependencies": dependencies,
            "other_pipes": {name: dict(Counter(row.pipe for row in rows if row.ident is not None))
                            for name, rows in actors.items()},
            "MTE_Vector_task_ownership": "unknown_no_cross_pipe_global_issue_sequence; IDs matched only",
            "scope": "pinned_source_single_core_synthetic_operator_not_64K_K32_or_NPU",
            "not_inferences": ["pop_to_complete_is_not_busy", "no_cross_actor_duration_sum",
                               "ready_dependency_wait_is_not_automatically_removable"]}


def main():
    from tools.experiments import decode_completion_probe as gate

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("normal", "profile", "normal-control", "profile-control"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Reuse the collector's exact schema, numerical completion, immutable run
    # plan, six full-file hashes and same-library perturbation gate. Presence
    # of a pair of gzip files alone is never a complete-run witness.
    comparison_path = args.output.with_suffix(".pair.json")
    comparison_args = argparse.Namespace(**{**vars(args), "output": comparison_path})
    if gate.compare(comparison_args) != 0:
        raise TraceError("normal/profile perturbation gate failed")
    directory = args.profile / "full_instructions"
    files = {f"core0.{kind}": tuple(directory / f"core0.{kind}.{suffix}.gz" for suffix in
             ("instr_popped_log.dump", "instr_log.dump")) for kind in ("cubecore0", "veccore0", "veccore1")}
    result = analyze_files(files)
    result["complete_pair_evidence"] = str(comparison_path)
    result["input_gzip_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for pair in files.values() for path in pair}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "history_rounds": len(result["history_rounds"]),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
