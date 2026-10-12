"""Archive #94/#95/#125/#126/#145/#151/#154/#155, D.4: locked CPU evidence.

Run only explicit small cases;120s per case, real stdout/rc, bounded SDK trace
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


def trace_summary(work,destination):
    destination.mkdir(parents=True,exist_ok=True)
    total=0;records=[];count=0
    for path in sorted(work.rglob("*")):
        if not path.is_file():continue
        size=path.stat().st_size;total+=size;count+=1
        if len(records)>=64:continue
        with path.open("rb") as stream:
            head=stream.read(4096);stream.seek(max(len(head),size-4096));tail=stream.read(4096)
        name=f"{len(records):03d}.preview.txt"
        (destination/name).write_bytes(head+b"\n[bounded head/tail]\n"+tail)
        records.append({"path":str(path.relative_to(work)),"bytes":size,"preview":name})
    summary={"total_bytes":total,"file_count":count,"files":records,"temporary_case_tree_removed":True}
    (destination/"manifest.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def run_case(executable,directory,logdir):
    name=directory.name;logdir.mkdir(parents=True,exist_ok=True)
    command=[str(executable.resolve()),str(directory.resolve())]
    (logdir/("cpu-"+name+".command.json")).write_text(json.dumps(command)+"\n")
    before=signature(directory);start=time.monotonic();timed_out=threading.Event();force=None
    temp_parent="/dev/shm" if Path("/dev/shm").is_dir() else None
    with tempfile.TemporaryDirectory(prefix="oscar-request-store-",dir=temp_parent) as work:
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
    markers=[line.split(" ",1)[1] for line in stdout.splitlines() if line.startswith("REQUEST_STORE_CPU_RESULT ")]
    evidence=json.loads(markers[0]) if len(markers)==1 else {}
    passed=rc==0 and not timed_out.is_set() and evidence.get("status")=="passed" and evidence.get("all_bytes")=="identical" and "[ERROR]" not in stdout and before==signature(directory)
    return {"name":name,"status":"passed" if passed else "failed","command":command,"returncode":rc,
            "timed_out":timed_out.is_set(),"cpu_wall_seconds_not_device_timing":time.monotonic()-start,
            "fixture_sha256":before,"evidence":evidence,"sdk_trace":traces}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable",type=Path,required=True)
    parser.add_argument("--fixtures",type=Path,required=True)
    parser.add_argument("--log-dir",type=Path,required=True)
    parser.add_argument("--cases",required=True,help="Comma separated explicit case names, or ALL for the fixed exported small-case manifest")
    args=parser.parse_args();selected=args.cases.split(",")
    manifest=json.loads((args.fixtures/"cases.json").read_text())
    known={c["name"] for c in manifest["cases"]}
    if selected==["ALL"]:selected=[c["name"] for c in manifest["cases"]]
    if any(name not in known for name in selected):raise ValueError("Unknown requested case")
    result={"backend":"official_ascendc_cpu_debug","status":"running","cases":[],"NPU":"not_run","graph":"not_run","device_performance":"not_run"}
    args.log_dir.mkdir(parents=True,exist_ok=True)
    with Path("/tmp/oscar-cann-validation.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in selected:
            entry=run_case(args.executable,args.fixtures/name,args.log_dir);result["cases"].append(entry)
            print("REQUEST_STORE_CASE",json.dumps({k:v for k,v in entry.items() if k not in ("sdk_trace","command")}),flush=True)
            if entry["status"]!="passed":result["status"]="failed"
            (args.log_dir/"cpu_cases_report.json").write_text(json.dumps(result,indent=2)+"\n")
            if result["status"]=="failed":return 1
        result["status"]="passed"
        (args.log_dir/"cpu_cases_report.json").write_text(json.dumps(result,indent=2)+"\n")
    return 0


if __name__=="__main__":raise SystemExit(main())
