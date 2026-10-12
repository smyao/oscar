"""Archive #126/#145/#151/#154/#155: host checkpoint admission, not CPU/NPU proof.

Small synthetic files test the controller. A standalone host C++ executable
tests the actual production task-byte admission helper without SDK/kernel code.
"""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from csrc.experiments.request_windows_attention import run_cpu

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def checkpoint(tmp_path):
    fixture=tmp_path/"fixture"/"c16_history_window_segment0";fixture.mkdir(parents=True)
    # Tiny byte-accounting fixture only; it is never submitted to a C16 kernel.
    (fixture/"shape.txt").write_text("2 2 1 64 1 1 512 1 1024 1 64 69632 4 19 64 1 3 1 1\n")
    (fixture/"input.bin").write_bytes(b"independent-fixture")
    baseline=tmp_path/"original-failed-pair";baseline.mkdir()
    name=fixture.name;outputs=baseline/("cpu-"+name+"-outputs");outputs.mkdir()
    for filename,size in run_cpu.baseline_sizes(fixture).items():(outputs/filename).write_bytes(bytes(size))
    entry={"name":name,"status":"failed","returncode":-15,"timed_out":True,
           "fixture_sha256":run_cpu.signature(fixture),"evidence":{},
           "command":["/original/c16-probe","c16",str(fixture),str(outputs)]}
    (baseline/"cpu_cases_report.json").write_text(json.dumps({"backend":"official_ascendc_cpu_debug","status":"failed","cases":[entry]}))
    (baseline/("cpu-"+name+".log")).write_text("SDK old finished\nREQUEST_WINDOW_CPU_ARM_COMPLETE old\nCandidate interrupted at unchanged120s limit\n")
    return fixture,baseline,outputs


def test_complete_old_arm_can_be_revalidated_without_claiming_original_binary_or_oracle_pass(checkpoint):
    fixture,baseline,outputs=checkpoint
    original=(baseline/"cpu_cases_report.json").read_bytes()
    proof=run_cpu.verify_baseline(fixture,baseline,mode="c16")
    assert proof["output_directory"]==str(outputs.resolve())
    assert proof["original_case_status"]=="failed" and proof["original_returncode"]==-15
    assert proof["original_binary_hash_recorded"] is False
    assert proof["original_baseline_oracle"]=="not_recorded"
    assert "cannot be authenticated" in proof["provenance_limit"]
    assert "independent_source0_source1_oracle" in proof["required_current_checks"]
    assert run_cpu.baseline_unchanged(proof)
    assert (baseline/"cpu_cases_report.json").read_bytes()==original


@pytest.mark.parametrize("damage",["no_marker","different_fixture","missing_output","truncated_output","old_checker_error"])
def test_incomplete_or_mismatched_old_arm_is_rejected(checkpoint,damage):
    fixture,baseline,outputs=checkpoint
    log=baseline/("cpu-"+fixture.name+".log")
    if damage=="no_marker":log.write_text("old started but never completed\n")
    elif damage=="different_fixture":(fixture/"input.bin").write_bytes(b"changed")
    elif damage=="missing_output":(outputs/"old_cluster_stats.bin").unlink()
    elif damage=="truncated_output":(outputs/"old_tasks.bin").write_bytes(b"incomplete")
    else:log.write_text("[ERROR] old invalid DMA\nREQUEST_WINDOW_CPU_ARM_COMPLETE old\n")
    with pytest.raises(ValueError):run_cpu.verify_baseline(fixture,baseline,mode="c16")


def test_source_files_must_remain_unchanged_after_validation(checkpoint):
    fixture,baseline,outputs=checkpoint
    proof=run_cpu.verify_baseline(fixture,baseline,mode="c16")
    path=outputs/"old_status.bin";content=bytearray(path.read_bytes());content[0]=1;path.write_bytes(content)
    assert not run_cpu.baseline_unchanged(proof)


def test_resume_cannot_overwrite_the_original_report_or_reuse_for_another_mode(checkpoint,tmp_path):
    fixture,baseline,_=checkpoint
    with pytest.raises(ValueError,match="new output directory"):
        run_cpu.run_case(tmp_path/"never-launched",fixture,baseline,"c16",baseline_root=baseline)
    with pytest.raises(ValueError,match="only qualified for C16"):
        run_cpu.verify_baseline(fixture,baseline,mode="q4")


def test_actual_cpp_task_snapshot_guard_rejects_different_or_short_saved_tasks(tmp_path):
    compiler=shutil.which("clang++")
    if compiler is None:pytest.skip("host compiler missing; no SDK gate was attempted")
    source=(ROOT/"csrc/experiments/request_windows_attention/cpu_probe.cpp").read_text()
    helper=source[source.index("std::vector<uint8_t> Read("):source.index("void Write(")]
    assert "RequireMatchingTasks" in helper
    cpp=tmp_path/"task_guard.cpp"
    cpp.write_text("#include <vector>\n#include <cstdint>\n#include <cstring>\n#include <fstream>\n#include <stdexcept>\n#include <string>\n#include <iostream>\n"+helper+'''
int main(int argc,char** argv) {
  const uint8_t actual[4]={1,2,3,4};
  try {RequireMatchingTasks(argv[1],actual,4);return 0;}
  catch(const std::exception& e){std::cerr<<e.what();return 1;}
}
''')
    executable=tmp_path/"task_guard"
    subprocess.run([compiler,"-std=c++17",str(cpp),"-o",str(executable)],check=True,capture_output=True,timeout=30)
    for content,expected in ((bytes((1,2,3,4)),0),(bytes((1,2,0,4)),1),(bytes((1,2,3)),1)):
        path=tmp_path/"tasks.bin";path.write_bytes(content)
        result=subprocess.run([str(executable),str(path)],capture_output=True,text=True,timeout=5)
        assert result.returncode==expected
        if len(content)==4 and expected:assert "task domain differs" in result.stderr
    assert source.index('RequireMatchingTasks(reference+"/old_tasks.bin"') < source.index("// The candidate may not mutate")
