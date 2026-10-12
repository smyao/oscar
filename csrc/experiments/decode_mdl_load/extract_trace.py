"""Archive #129/#132-135/#145/#148/#154-156, D.4: unmarked complete MDL traces.

Use actual prepared task bytes, pinned Process order and GM workspace objects
to bind B loads; never assign queued work merely by a wrapper timestamp.
ND2NZ XM/XT are retained verbatim, not decoded by guessed ISA bit positions.
Logical rectangles follow the pinned SDK call geometry plus observed source
origins. No popped-to-completion sum is called busy time or HBM traffic.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from csrc.experiments.decode_mdl_load.run import checked
from tools.experiments.analyze_decode_completion import (TraceError, cross_dependencies,
                                                       pair_instructions, read_trace)

REG = {key: re.compile(rf"\b{key}:X\d+=(0x[0-9a-fA-F]+|\d+)\b") for key in ("XD", "XN", "XM", "XT")}
COPY = "MOV_OUT_TO_L1_MULTI_ND2NZ"


def registers(row):
    values = {}
    for key, pattern in REG.items():
        match = pattern.search(row.operands)
        if not match:
            raise TraceError("ND2NZ instruction missing actual register value: " + key)
        values[key] = int(match[1], 0)
    if "Dtype:B32S" not in row.operands:
        raise TraceError("not the pinned FP32 ND2NZ opcode")
    return values


def task_plan(report, tasks):
    if hashlib.sha256(tasks).hexdigest() != report["input_sha256"]["tasks"]:
        raise TraceError("actual prepared task identity differs")
    n, hq, hk, dim, requests = report["attrs"][:5]
    splits = report["attrs"][-1]
    if (n, hq, hk, dim, requests, splits, report["cores"]) != (8, 6, 1, 256, 2, 1, 1):
        raise TraceError("only the declared C1341 R2 q4 single-core experiment is supported")
    if len(tasks) != report["attrs"][6] * 128:
        raise TraceError("task byte extent mismatch")
    rows = list(struct.iter_unpack("<16q", tasks))
    result = []
    query_tiles, per_token = (n + 3)//4, hk*3*splits
    # Exact d817 CvTaskSchedule workId/TokenBegin/TaskId ordering.
    for work in range(query_tiles * per_token):
        begin, segment = (work % query_tiles)*4, work//query_tiles
        for token in range(begin, min(begin+4, n)):
            task_id = token*per_token + segment
            row = rows[task_id]
            if row[1] == 0:
                continue
            qbegin, qcount, _head, kvbegin, kvend, request, split, kind, context, _request_begin, error = row[:11]
            if qcount != 4 or error or kind not in (0,1,2) or kvend < kvbegin or context != 1341:
                raise TraceError("unexpected live task; no inferred replacement")
            base = {"task_id": task_id, "request": request, "source": kind,
                    "qbegin": qbegin, "M": qcount*(hq//hk), "split": split}
            for index, start in enumerate(range(kvbegin, kvend, 256)):
                columns = 256 if kind == 0 else ((min(256, kvend-start)+15)//16)*16
                buffer = index % 2 if kind == 0 else 0
                shared = {**base, "kv_round": index, "start": start,
                          "live_KV": min(256,kvend-start), "physical_B_rows": columns, "physical_B_columns": dim}
                result.append({**shared, "phase": "qk", "B_object": f"K{buffer}", "ready_flag": 9})
                result.append({**shared, "phase": "pv", "B_object": f"V{buffer}", "ready_flag": 11 if kind == 0 else 9})
            if kind == 0 and kvbegin < kvend:
                result.append({**base, "phase": "rotation", "B_object": "Rv", "ready_flag": 13,
                               "physical_B_rows": dim, "physical_B_columns": dim})
    if Counter((r["source"],r["phase"]) for r in result) != Counter({(0,"qk"):10,(0,"pv"):10,
            (0,"rotation"):2,(1,"qk"):4,(1,"pv"):4,(2,"qk"):2,(2,"pv"):2}):
        raise TraceError("actual task plan is not the declared full-source work")
    return result


def classify_gm(pointer, base):
    # Exact D256 d817 bounded work layout; source values are not shared
    # across requests. Labels name reused physical buffers only.
    regions = (("Q",0,32768),("K0",32768,294912),("V0",294912,557056),
               ("K1",557056,819200),("V1",819200,1081344),
               ("P",1081344,1114112),("PV",1114112,1146880))
    for name, low, high in regions:
        if base+low <= pointer < base+high:
            return name, pointer-base-low
    return "Rv", None


def logical_rectangles(group, method):
    height, width = method["physical_B_rows"], method["physical_B_columns"]
    if method["B_object"] == "Rv":
        origin = min(row["source_GM"] for row in group)
        offsets = [row["source_GM"]-origin for row in group]
    else:
        offsets = [row["offset_in_object"] for row in group]
    starts = [divmod(offset//4,256) for offset in offsets]
    if any(offset % 4 for offset in offsets) or len(set(starts)) != len(starts):
        return {"status": "unknown_nonunique_or_unaligned_origins", "starts": starts}
    axis0, axis1 = sorted({s[0] for s in starts}), sorted({s[1] for s in starts})
    if (not axis0 or axis0[0] != 0 or axis1[0] != 0 or max(axis0)>=height or max(axis1)>=width or
            set(starts) != set(itertools.product(axis0,axis1))):
        return {"status": "unknown_nonrectangular_origin_grid", "starts": starts}
    ends0, ends1 = axis0[1:]+[height], axis1[1:]+[width]
    sizes0, sizes1 = dict(zip(axis0,[b-a for a,b in zip(axis0,ends0)])), dict(zip(axis1,[b-a for a,b in zip(axis1,ends1)]))
    rectangles = [{"row": row, "column": col, "rows": sizes0[row], "columns": sizes1[col],
                   "logical_bytes": sizes0[row]*sizes1[col]*4} for row,col in starts]
    return {"status": "SDK_call_geometry_and_observed_origin_grid", "rectangles": rectangles,
            "total_logical_B_bytes": sum(r["logical_bytes"] for r in rectangles),
            "not_inference": "not independent ISA bitfield decoding or measured HBM bytes"}


def analyze_arm(path, report, tasks):
    plan = task_plan(report,tasks)
    actors = {}
    for kind in ("cubecore0","veccore0","veccore1"):
        actor = "core0."+kind
        directory = path/"full_instructions"
        actors[actor] = pair_instructions(read_trace(directory/(actor+".instr_popped_log.dump.gz"),actor),
                                         read_trace(directory/(actor+".instr_log.dump.gz"),actor))
    cube = actors["core0.cubecore0"]
    copies = [r for r in cube if r.op == COPY and r.pipe == "MTE2"]
    if not copies:
        raise TraceError("no FP32 ND2NZ copies in this arm")
    # First Matmul is the first source0 QK: A=qOffset0; baseline/MDL both
    # load A before B in the witnessed SDK. Enforce a first K0 base copy too.
    workspace_base = registers(copies[0])["XN"]
    collected = []
    for row in copies:
        regs = registers(row)
        obj, offset = classify_gm(regs["XN"], workspace_base)
        collected.append({"row": row, "object": obj, "offset_in_object": offset,
            "source_GM": regs["XN"], "destination_L1": regs["XD"],
            "raw_XM": hex(regs["XM"]), "raw_XT": hex(regs["XT"]),
            "popped": row.cycle, "complete": row.complete, "instruction_id": row.ident})
    if collected[0]["object"] != "Q" or not any(r["object"]=="K0" and r["offset_in_object"]==0 for r in collected):
        raise TraceError("GM workspace base lacks source-bound first Q/K0 witness")
    bcopy = [r for r in collected if r["object"] in ("K0","K1","V0","V1","Rv")]
    groups = [list(values) for _,values in itertools.groupby(bcopy,key=lambda row:row["object"])]
    ready = [r for r in cube if r.op=="SET_CROSS_CORE" and r.pipe=="FIXP"]
    if len(groups)!=len(plan) or len(ready)!=len(plan):
        raise TraceError("B object group/ready/actual task Matmul count mismatch")
    methods=[]
    previous_ready=-1
    all_math=[r for r in cube if r.op in ("MMAD","FIX_L0C_TO_DST")]
    used=set()
    for method,group,signal in zip(plan,groups,ready):
        if group[0]["object"]!=method["B_object"] or signal.flag!=method["ready_flag"] or signal.mode!=2:
            raise TraceError("source/task B order or ready protocol differs")
        if max(r["complete"] for r in group)>signal.complete:
            raise TraceError("B DMA completes after its result ready")
        math=[r for r in all_math if previous_ready<r.complete<=signal.complete]
        if not math:
            raise TraceError("Matmul ready without completed math")
        used.update(r.sequence for r in math)
        methods.append({**method,"B_copies":len(group),"B_first_popped":min(r["popped"] for r in group),
            "B_last_complete":max(r["complete"] for r in group),"ready_complete":signal.complete,
            "B_logical_partition":logical_rectangles(group,method),
            "B_instructions":[{k:v for k,v in item.items() if k!="row"} for item in group],
            "MMAD_count":sum(r.op=="MMAD" for r in math),
            "MMAD_raw_operands":[r.operands for r in math if r.op=="MMAD"],
            "last_MMAD_complete":max(r.complete for r in math if r.op=="MMAD"),
            "FIX_count":sum(r.op=="FIX_L0C_TO_DST" for r in math),
            "last_FIX_complete":max(r.complete for r in math if r.op=="FIX_L0C_TO_DST")})
        previous_ready=signal.complete
    if used!={r.sequence for r in all_math}:
        raise TraceError("unassigned math beyond ready boundaries")
    dependencies=cross_dependencies(actors)
    by_flag=defaultdict(list)
    for row in dependencies:
        by_flag[row["actor"],row["flag"]].append(row)
    rounds=[];history_index=0;previous_pv={}
    for index,method in enumerate(methods):
        if method["source"]!=0 or method["phase"]!="qk":continue
        pv=methods[index+1]
        if pv["task_id"]!=method["task_id"] or pv["phase"]!="pv":raise TraceError("history QK/PV order")
        kv=by_flag["core0.cubecore0",8][history_index]
        p=by_flag["core0.cubecore0",10][history_index]
        if not method["ready_complete"]<=p["producer_ready_complete"]<=pv["ready_complete"]:
            raise TraceError("history ready chain does not close")
        row={"task_id":method["task_id"],"kv_round":method["kv_round"],
             "KV_producer_not_ready":kv["observed_producer_not_ready_ticks"],
             "ScoreReady_to_PReady":p["producer_ready_complete"]-method["ready_complete"],
             "PReady_to_PvReady":pv["ready_complete"]-p["producer_ready_complete"],
             "qk":method,"pv":pv}
        if method["task_id"] in previous_pv:
            last=previous_pv[method["task_id"]]
            row.update(previous_PvReady_to_ScoreReady=method["ready_complete"]-last,
                       PvReady_period=pv["ready_complete"]-last)
        previous_pv[method["task_id"]]=pv["ready_complete"]
        rounds.append(row);history_index+=1
    if history_index!=10:raise TraceError("missing history rounds")
    return {"task_ticks":report["task_ticks"],"workspace_base_virtual_address":hex(workspace_base),
        "B_copy_count":len(bcopy),"all_ND2NZ_count":len(copies),"methods":methods,"history_rounds":rounds,
        "ownership": "pinned actual task order + ordered B GM objects + original ready flags",
        "MTE_Vector_global_ownership": "not_generalized", "unmatched":[]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("baseline","candidate","baseline-control","candidate-control","tasks","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args()
    results={};reports={}
    for arm in ("baseline","candidate"):
        path=getattr(args,arm)
        report=checked(path,getattr(args,arm+"_control"),arm)
        reports[arm]=report
        results[arm]=analyze_arm(path,report,args.tasks.read_bytes())
    for key in ("case","mode","cores","attrs","scale","input_sha256","oracle_sha256",
                "cpu_baseline_sha256","library_sha256","simulator_configuration","cpu_reference",
                "cpu_reference_policy","workspace_bytes_per_core","trace_policy"):
        if reports["baseline"][key]!=reports["candidate"][key]:
            raise TraceError("unpaired complete traces: "+key)
    for key in ("partial","lse","status"):
        if reports["baseline"]["outputs"][key]["sha256"]!=reports["candidate"]["outputs"][key]["sha256"]:
            raise TraceError("complete trace pair changed output bits: "+key)
    for old,new in zip(results["baseline"]["methods"],results["candidate"]["methods"]):
        def raw_xt(method):
            values=[]
            for row in method["MMAD_raw_operands"]:
                match=re.search(r"\bXT:(0x[0-9a-fA-F]+)\b",row)
                if not match or "dtype:F32F32" not in row:
                    raise TraceError("MMAD dtype/parameter witness missing")
                values.append(match[1])
            return values
        if raw_xt(old)!=raw_xt(new):
            raise TraceError("per-method MMAD parameter/order changed; source claim needs re-audit")
    result={"status":"complete_numeric_gated_B_load_comparison","arms":results,
        "same_library_inputs_and_output_bits":True,"per_method_FP32_MMAD_XT_sequence_identical":True,
        "task_ratio":results["baseline"]["task_ticks"]/results["candidate"]["task_ticks"],
        "not_inferences":["no_pop_to_complete_busy_sum","no_HBM_bytes_from_GM_addresses",
            "no_64K_or_NPU_speed_claim","logical_rectangles_not_guessed_ISA_bitfields"]}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"task_ratio":result["task_ratio"],
        "B_copies":{name:r["B_copy_count"] for name,r in results.items()},"output":str(args.output)}))


if __name__=="__main__":main()
