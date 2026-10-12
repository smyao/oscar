"""Archive #94/#95/#125/#126/#145/#151/#154/#155, D.4: locked CPU evidence.

Run only explicit reader cases;120s per case, real stdout/rc, bounded SDK trace
summaries. Math and checks remain enabled. CPU elapsed is never device speed.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time


def signature(directory):
    digest=hashlib.sha256()
    for p in sorted(directory.iterdir()):
        if p.is_file():digest.update(p.name.encode()+b"\0"+p.read_bytes())
    return digest.hexdigest()


def file_sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        while chunk:=stream.read(1024*1024):digest.update(chunk)
    return digest.hexdigest()


def baseline_sizes(directory):
    shape=list(map(int,(directory/"shape.txt").read_text().split()))
    if len(shape)!=19 or min(shape[:4])<=0 or not 1<=shape[17]<=32 or not 1<=shape[18]<=32:
        raise ValueError("invalid fixture shape for completed baseline validation")
    n,h,k,d=shape[:4];splits,cores=shape[17:19];tasks=n*k*3*splits
    return {"old_partial.bin":n*h*3*splits*d*4,"old_lse.bin":n*h*3*splits*4,
            "old_status.bin":tasks*2*4,"old_tasks.bin":tasks*16*8,
            "old_cluster_stats.bin":cores*8*8}


def verify_baseline(directory,baseline_root,*,mode):
    """A completion checkpoint is not an original oracle/binary attestation.

    The C++ process must still compare real freshly prepared tasks, validate
    all loaded status/counters and independently run the frozen oracle BEFORE
    executing the candidate. Original failed reports/logs are never rewritten.
    """
    if mode!="c16":raise ValueError("baseline reuse is only qualified for C16")
    name=directory.name;report_path=baseline_root/"cpu_cases_report.json"
    report=json.loads(report_path.read_text())
    if report.get("backend")!="official_ascendc_cpu_debug":
        raise ValueError("baseline report is not official CPU-debug evidence")
    matches=[row for row in report.get("cases",[]) if row.get("name")==name]
    if len(matches)!=1:raise ValueError("baseline report requires one matching case")
    entry=matches[0]
    command=entry.get("command")
    if not isinstance(command,list) or len(command)<4 or command[1]!="c16":
        raise ValueError("baseline command does not witness the C16 harness")
    if entry.get("fixture_sha256")!=signature(directory):
        raise ValueError("baseline fixture hash differs")
    log_path=baseline_root/("cpu-"+name+".log")
    lines=log_path.read_text().splitlines()
    marker="REQUEST_WINDOW_CPU_ARM_COMPLETE old"
    if marker not in lines:raise ValueError("baseline lacks the completed old-arm marker")
    if any("[ERROR]" in line for line in lines[:lines.index(marker)+1]):
        raise ValueError("baseline checker reported an error before old completion")
    outputs=baseline_root/("cpu-"+name+"-outputs")
    files={}
    for filename,size in baseline_sizes(directory).items():
        path=outputs/filename
        if not path.is_file() or path.stat().st_size!=size:
            raise ValueError("baseline output missing or incomplete: "+filename)
        files[filename]={"bytes":size,"sha256":file_sha256(path)}
    original_sha=entry.get("executable_sha256")
    original_oracle=entry.get("evidence",{}).get("baseline_oracle_checked_this_run") is True
    return {"output_directory":str(outputs.resolve()),"report":str(report_path.resolve()),
        "report_sha256":file_sha256(report_path),"log":str(log_path.resolve()),"log_sha256":file_sha256(log_path),
        "original_case_status":entry.get("status"),"original_returncode":entry.get("returncode"),"original_command":command,
        "fixture_sha256":entry["fixture_sha256"],"files":files,
        "original_executable_sha256":original_sha,
        "original_binary_hash_recorded":isinstance(original_sha,str) and len(original_sha)==64,
        "original_baseline_oracle":"recorded_passed" if original_oracle else "not_recorded",
        "provenance_limit":"Original source staging/build context is caller-supplied; the original binary cannot be authenticated when its hash was not recorded.",
        "required_current_checks":["real_old_new_prepare","saved_task_byte_parity","all_baseline_status",
                                   "baseline_cluster_counters","independent_source0_source1_oracle","gm_guards"]}


def baseline_unchanged(proof):
    return (file_sha256(Path(proof["report"]))==proof["report_sha256"] and
            file_sha256(Path(proof["log"]))==proof["log_sha256"] and
            all(file_sha256(Path(proof["output_directory"])/name)==info["sha256"]
                for name,info in proof["files"].items()))


def trace_summary(work,destination):
    destination.mkdir(parents=True,exist_ok=True)
    total=0;records=[];count=0
    for path in sorted(work.rglob("*")):
        if not path.is_file():continue
        size=path.stat().st_size;total+=size;count+=1
        with path.open("rb") as stream:
            head=stream.read(4096);stream.seek(max(len(head),size-4096));tail=stream.read(4096)
        digest=hashlib.sha256()
        with path.open("rb") as stream:
            while chunk:=stream.read(1024*1024):digest.update(chunk)
        name=f"{len(records):03d}.preview.txt" if len(records)<64 else None
        if name:(destination/name).write_bytes(head+b"\n[bounded head/tail]\n"+tail)
        records.append({"path":str(path.relative_to(work)),"bytes":size,"preview":name,"sha256":digest.hexdigest()})
    summary={"total_bytes":total,"file_count":count,"files":records,"temporary_case_tree_removed":True}
    (destination/"manifest.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def run_case(executable,directory,logdir,mode,*,baseline_root=None,baseline_only=False):
    if baseline_root is not None and baseline_only:raise ValueError("select only one baseline mode")
    if (baseline_root is not None or baseline_only) and mode!="c16":raise ValueError("split-arm mode requires C16")
    if baseline_root is not None and logdir.resolve()==baseline_root.resolve():
        raise ValueError("reuse requires a new output directory; preserve the original report")
    proof=verify_baseline(directory,baseline_root,mode=mode) if baseline_root is not None else None
    name=directory.name;logdir.mkdir(parents=True,exist_ok=True)
    exports=logdir/("cpu-"+name+"-outputs");exports.mkdir(parents=True,exist_ok=True)
    command=[str(executable.resolve()),mode,str(directory.resolve()),str(exports.resolve())]
    if proof:command += ["candidate-from",proof["output_directory"]]
    elif baseline_only:command += ["baseline-only"]
    (logdir/("cpu-"+name+".command.json")).write_text(json.dumps(command)+"\n")
    executable_sha=file_sha256(executable)
    if proof:(logdir/("cpu-"+name+".baseline-reuse.json")).write_text(json.dumps(proof,indent=2)+"\n")
    before=signature(directory);start=time.monotonic();timed_out=threading.Event();force=None
    temp_parent="/dev/shm" if Path("/dev/shm").is_dir() else None
    with tempfile.TemporaryDirectory(prefix="oscar-request-reader-",dir=temp_parent) as work:
        process=subprocess.Popen(command,cwd=work,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
        def terminate():
            nonlocal force
            if process.poll() is not None:return
            timed_out.set()
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:return
            def kill():
                if process.poll() is None:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
            force=threading.Timer(5,kill);force.daemon=True;force.start()
        deadline=threading.Timer(120,terminate);deadline.daemon=True;deadline.start()
        lines=[]
        try:
            with (logdir/("cpu-"+name+".log")).open("w") as dest:
                for line in process.stdout:
                    print(line,end="",flush=True);dest.write(line);dest.flush();lines.append(line)
            rc=process.wait()
        finally:
            deadline.cancel()
            if process.poll() is None:
                terminate();process.wait(timeout=6)
            if force is not None:force.cancel()
            traces=trace_summary(Path(work),logdir/("cpu-"+name+"-trace"))
    stdout="".join(lines)
    markers=[line for line in stdout.splitlines() if line.startswith('{"backend":"official_cpu_debug"')]
    evidence=json.loads(markers[0]) if len(markers)==1 else {}
    expected_status="baseline_checkpoint_passed" if baseline_only else "passed"
    passed=(rc==0 and not timed_out.is_set() and evidence.get("status")==expected_status and
            evidence.get("all_metadata_equal") is True and evidence.get("input_immutable") is True and
            evidence.get("gm_guards") is True and "[ERROR]" not in stdout and before==signature(directory) and
            executable_sha==file_sha256(executable))
    if proof:
        passed=passed and baseline_unchanged(proof) and evidence.get("baseline_reused") is True and evidence.get("baseline_oracle_checked_this_run") is True and evidence.get("candidate_executed") is True
    if baseline_only:
        passed=passed and evidence.get("baseline_oracle_checked_this_run") is True and evidence.get("baseline_status_checked") is True and evidence.get("baseline_cluster_checked") is True and evidence.get("candidate_executed") is False
    return {"name":name,"status":expected_status if passed else "failed","command":command,"returncode":rc,
            "timed_out":timed_out.is_set(),"cpu_wall_seconds_not_device_timing":time.monotonic()-start,
            "fixture_sha256":before,"executable_sha256":executable_sha,"baseline_reuse":proof,
            "scope":"baseline_checkpoint_only" if baseline_only else "candidate_vs_completed_baseline" if proof else "paired_cpu_numerics",
            "evidence":evidence,"sdk_trace":traces}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable",type=Path,required=True)
    parser.add_argument("--fixtures",type=Path,required=True)
    parser.add_argument("--log-dir",type=Path,required=True)
    parser.add_argument("--cases",required=True,help="Comma separated explicit case names, or ALL for the fixed exported small-case manifest")
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--baseline-root",type=Path,help="Prior report/log/output directory with witnessed complete old arms")
    group.add_argument("--baseline-only",action="store_true",help="C16 old arm plus full oracle/status checkpoint; no candidate execution")
    args=parser.parse_args();selected=args.cases.split(",")
    manifest=json.loads((args.fixtures/"cases.json").read_text())
    known={c["name"]:c for c in manifest["cases"]}
    if selected==["ALL"]:selected=[c["name"] for c in manifest["cases"]]
    if any(name not in known for name in selected):raise ValueError("Unknown requested case")
    if (args.log_dir/"cpu_cases_report.json").exists():raise ValueError("Use a new log directory; never overwrite prior evidence")
    result={"backend":"official_ascendc_cpu_debug","status":"running","cases":[],"NPU":"not_run","graph":"not_run","device_performance":"not_run"}
    args.log_dir.mkdir(parents=True,exist_ok=True)
    with Path("/tmp/oscar-cann-validation.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in selected:
            entry=run_case(args.executable,args.fixtures/name,args.log_dir,known[name]["mode"],baseline_root=args.baseline_root,baseline_only=args.baseline_only);result["cases"].append(entry)
            print("REQUEST_READER_CASE",json.dumps({k:v for k,v in entry.items() if k not in ("sdk_trace","command")}),flush=True)
            if entry["status"]!=("baseline_checkpoint_passed" if args.baseline_only else "passed"):result["status"]="failed"
            (args.log_dir/"cpu_cases_report.json").write_text(json.dumps(result,indent=2)+"\n")
            if result["status"]=="failed":return 1
        result["status"]="baseline_checkpoint_passed" if args.baseline_only else "passed"
        (args.log_dir/"cpu_cases_report.json").write_text(json.dumps(result,indent=2)+"\n")
    return 0


if __name__=="__main__":raise SystemExit(main())
