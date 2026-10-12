"""Archive #94/#95/#125/#126/#140/#145/#148/#155: serial bounded CPU A/B.

D.4: current production SIMD is the denominator; these official CPU results
and optional CAModel exports are distinct from NPU/model performance.
Large32-request cases remain export-only unless explicitly selected here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.phase import atomic_json,run_phase


@contextmanager
def validation_lock():
    """CPU-debug workers and CAModel must not compete for VM memory.

    2026-10-07 local OOM killed an unrelated in-progress CAModel while the
    N128 CPU checker spawned several2GB workers. A nonblocking shared lease
    makes this a visible scheduling error, not an apparent kernel failure.
    """
    with Path('/tmp/oscar-cann-validation.lock').open('a+') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another owned CANN CPU/CAModel validation is running; run serially') from error
        try:
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)


def _signature(directory):
    digest=hashlib.sha256()
    for path in sorted(directory.glob("*")):
        if path.is_file():digest.update(path.name.encode()+b"\0"+path.read_bytes())
    return digest.hexdigest()


def _save_trace_summary(workdir,destination):
    """Bounded SDK evidence; TemporaryDirectory removes only this case tree."""
    destination.mkdir(parents=True,exist_ok=True)
    records=[];total=0;count=0
    for path in sorted(workdir.rglob("*")):
        if not path.is_file():continue
        size=path.stat().st_size;count+=1;total+=size
        if len(records)>=64:continue
        with path.open("rb") as stream:
            head=stream.read(4096)
            stream.seek(max(len(head),size-4096));tail=stream.read(4096)
        preview=destination/f"{len(records):03d}.preview.txt"
        preview.write_text(head.decode("utf-8",errors="replace")+"\n[bounded head/tail]\n"+
            tail.decode("utf-8",errors="replace"))
        records.append({"path":str(path.relative_to(workdir)),"bytes":size,"preview":str(preview)})
    summary={"files":records,"file_count":count,"total_bytes":total,"entries_truncated":count>len(records),
        "complete_stdout":"separate run_phase log","sdk_workdir_cleaned_after_summary":True}
    atomic_json(destination/"manifest.json",summary)
    return summary


def run(executable,cases_path,output,log_dir,*,selected=None,export_dir=None):
    manifest=json.loads(cases_path.read_text())
    cases=[case for case in manifest["cases"] if case["name"] in selected] if selected else [
        case for case in manifest["cases"] if case["run_default"]]
    if not cases or (selected and {case["name"] for case in cases}!=set(selected)):
        raise ValueError("unknown or empty selected fixture set")
    report={"status":"running","backend":"official_ascendc_cpu_debug","cases":[],
        "baseline":"current_production_simd","npu":"not_run","device_speed":"not_established"}
    atomic_json(output,report)
    for index,case in enumerate(cases):
        # Resolve beside the manifest so a shared/moved fixture directory does
        # not depend on a stale absolute macOS path recorded at export time.
        directory=cases_path.parent/case["name"]
        before=_signature(directory)
        command=[str(executable.resolve()),case["mode"],str(directory.resolve())]
        if export_dir is not None:
            export=export_dir/case["name"];export.mkdir(parents=True,exist_ok=True)
            command.append(str(export.resolve()))
        temporary_parent=Path("/dev/shm") if Path("/dev/shm").is_dir() else Path(tempfile.gettempdir())
        with tempfile.TemporaryDirectory(prefix="oscar-window-case-",dir=temporary_parent) as work:
            workdir=Path(work)
            try:
                result=run_phase(f"{index:02d}-{case['name']}",command,cwd=workdir,log_dir=log_dir,
                    timeout=120,grace=5,heartbeat=30,
                    env={**os.environ,"OSCAR_TERMINAL_LOG_MODE":"compact"})
            finally:
                traces=_save_trace_summary(workdir,log_dir/f"{index:02d}-{case['name']}-trace")
        log=Path(result.log).read_text()
        markers=[line for line in log.splitlines() if line.startswith('{"backend":"ascendc_cpu_debug"')]
        evidence=json.loads(markers[0]) if len(markers)==1 else {}
        passed=(result.returncode==0 and not result.timed_out and result.cleanup_complete and
            evidence.get("status")=="passed" and evidence.get("baseline")=="current_production_simd"
            and evidence.get("bitwise") is True and evidence.get("cores")==case["cores"]
            and evidence.get("input_immutable") is True and evidence.get("gm_guards") is True
            and evidence.get("fault_order") is case["expected_error"]
            and evidence.get("oracle") is (not case["expected_error"])
            and "[ERROR]" not in log and "error happened!" not in log and before==_signature(directory))
        report["cases"].append({"name":case["name"],"status":"passed" if passed else "failed",
            "returncode":result.returncode,"timed_out":result.timed_out,"log":result.log,
            "golden_sha256":before,"evidence":evidence,"sdk_trace_summary":traces})
        if not passed:
            report["status"]="failed";atomic_json(output,report);return 1
        atomic_json(output,report)
    report["status"]="passed";atomic_json(output,report);return 0


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable",type=Path,required=True)
    parser.add_argument("--cases",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--log-dir",type=Path,required=True)
    parser.add_argument("--case",action="append")
    parser.add_argument("--export-dir",type=Path)
    args=parser.parse_args()
    with validation_lock():
        raise SystemExit(run(args.executable,args.cases,args.output,args.log_dir,
            selected=args.case,export_dir=args.export_dir))
