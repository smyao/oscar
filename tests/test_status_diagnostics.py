"""Archive #157: CPU packet/contracts/host formatter; never NPU execution evidence."""
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

from oscar_ascend.status_diagnostics import enabled, make_packet, status_guard

ROOT = Path(__file__).resolve().parents[1]


def test_empty_and_zero_statuses_are_not_errors():
    groups = [torch.zeros(n, dtype=torch.int32) for n in (0, 7, 2049, 0)]
    before = [v.clone() for v in groups]
    report = make_packet(groups)
    assert report.shape == (4, 32) and report.dtype == torch.int64 and report.is_contiguous()
    assert report[:, 0].tolist() == [-1] * 4
    assert report[:, 1:3].tolist() == [[0, 0]] * 4
    assert report[:, 3].tolist() == [0, 7, 2049, 0]
    assert all(torch.equal(a, b) for a, b in zip(groups, before))


def test_packet_traces_without_host_scalar_reads_and_reads_new_values():
    compiled = torch.compile(make_packet, backend="eager", fullgraph=True)
    groups = [torch.zeros(2049, dtype=torch.int32) for _ in range(4)]
    assert compiled(groups)[:, 2].tolist() == [0, 0, 0, 0]
    groups[2][1500] = 4
    assert compiled(groups)[2, :4].tolist() == [1500, 4, 1, 2049]


@pytest.mark.parametrize("group", range(4))
@pytest.mark.parametrize("index", (0, 1023, 1024, 2048))
def test_first_error_covers_every_group_and_segment(group, index):
    groups = [torch.zeros(2050, dtype=torch.int32) for _ in range(4)]
    groups[group][index] = -7
    groups[group][2049] = 4
    report = make_packet(groups)
    assert report[group, :4].tolist() == [index, -7, 2, 2050]
    assert report[:, 2].sum() == 2


def test_live_packet_maps_attention_rotate_merge_store_and_invalid_position():
    # Same storage is reused, just as in graph replay. Task 2 is source2/token0.
    tasks = torch.zeros((6, 16), dtype=torch.int64)
    tasks[:, 0] = torch.tensor([0, 0, 0, 1, 1, 1])
    tasks[:, 5] = 0
    tasks[:, 7] = torch.tensor([0, 1, 2, 0, 1, 2])
    tasks[:, 8] = 1290
    tasks[2, 10] = 3
    groups = [torch.zeros(n, dtype=torch.int32) for n in (12, 12, 12, 2)]
    groups[0][5], groups[1][10], groups[2][11], groups[3][1] = 7, 2, 3, 4
    slots = torch.tensor([3338, 3339]); positions = torch.tensor([-1, 1291])
    starts = torch.tensor([0, 2]); lengths = torch.tensor([1292])
    table = torch.full((1, 16), 26, dtype=torch.int32)
    def packet():
        return make_packet(groups, tasks=tasks, slots=slots, positions=positions,
            starts=starts, seq_lens=lengths, block_tables=table,
            heads=6, kv_heads=1, block_size=2304, attention_layout=True)
    result = packet()
    assert result[0, 4:10].tolist() == [0, 0, 1, 3338, -1, 2]
    assert result[0, 10:16].tolist() == [0, 2, 1292, 26, 3338, 1]
    assert result[0, 26] == 3
    assert result[1, 4:6].tolist() == [1, 4]
    assert result[2, 4:6].tolist() == [1, 5]
    assert result[3, 4:6].tolist() == [1, 0]
    lengths[0] += 128; table[0, 11] = 41; slots[0] = 5258; tasks[:, 8] += 128
    updated = packet()
    assert updated[0, 7] == 5258
    assert updated[0, 12:15].tolist() == [1420, 41, 5258]


def test_later_mtp_uses_repaired_position_not_native_rejected_length():
    a = torch.tensor([5, 0], dtype=torch.int32); empty = a[:0]
    tasks = torch.zeros((1, 16), dtype=torch.int64); tasks[0, 8] = 1280
    slots = torch.tensor([128]); positions = torch.tensor([1280])
    table = torch.arange(16, dtype=torch.int32).reshape(1, -1); table[0, 10] = 1
    kwargs = dict(attention_layout=True, tasks=tasks, slots=slots, positions=positions,
                  starts=torch.tensor([0, 1]), seq_lens=torch.tensor([1400]), block_tables=table)
    result = make_packet((a, empty, empty, empty), **kwargs)
    assert result[0, 13:15].tolist() == [1, 128]
    tasks[0, 10] = 5; positions.fill_(-1)
    unknown = make_packet((a, empty, empty, empty), **kwargs)
    assert unknown[0, 13:15].tolist() == [-1, -1]


def test_flat_guards_and_out_of_bounds_metadata_do_not_oob_gather():
    group = torch.tensor([0, 0, 1, 1], dtype=torch.int32); empty = group[:0]
    report = make_packet((group, empty, empty, empty), slots=torch.tensor([10, -1]), flat_heads=2)
    assert report[0, 4:8].tolist() == [1, 0, -1, -1]
    tasks = torch.full((1, 16), 10000, dtype=torch.int64)
    result = make_packet((group, empty, empty, empty), attention_layout=True,
                        tasks=tasks, slots=torch.tensor([7]), seq_lens=torch.tensor([1]),
                        starts=torch.tensor([0, 1]), block_tables=torch.tensor([[0]]))
    assert result[0, 7:9].tolist() == [-1, -1]
    assert result[0, 10:15].tolist() == [-1] * 5


def test_disabled_diagnostic_is_original_guard_only(monkeypatch):
    monkeypatch.delenv("OSCAR_STATUS_DIAGNOSTICS", raising=False)
    seen = []
    tensors = tuple(torch.tensor([0], dtype=torch.int32) for _ in range(4))
    ops = SimpleNamespace(status_guard=lambda *args: seen.append(args))
    monkeypatch.setattr("oscar_ascend.status_diagnostics.make_packet",
                        lambda *args, **kwargs: pytest.fail("disabled diagnostic built a packet"))
    status_guard(ops, *tensors)
    assert len(seen) == 1 and all(a is b for a, b in zip(seen[0], tensors))


def test_enabled_calls_combined_op_and_requires_rebuild(monkeypatch):
    monkeypatch.setenv("OSCAR_STATUS_DIAGNOSTICS", "1")
    tensors = tuple(torch.tensor([0], dtype=torch.int32) for _ in range(4))
    seen = []
    ops = SimpleNamespace(status_guard_diagnostic=lambda *args: seen.append(args))
    status_guard(ops, *tensors, context={"layer": "layers.3.attn", "phase": "final"})
    assert len(seen) == 1 and seen[0][4].shape == (4, 32)
    assert json.loads(seen[0][5])["layer"] == "layers.3.attn"
    with pytest.raises(RuntimeError, match="rebuilding"):
        status_guard(SimpleNamespace(), *tensors)
    monkeypatch.setenv("OSCAR_STATUS_DIAGNOSTICS", "yes")
    with pytest.raises(ValueError, match="0 or 1"):
        enabled()


def test_real_host_formatter_and_json_lines(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable; formatter execution NOT verified")
    if subprocess.run([compiler, "--version"], capture_output=True).returncode != 0:
        pytest.skip("host C++ toolchain unavailable; formatter execution NOT verified")
    # Compile the exact production formatter, not a Python equivalent.
    source = r'''
    #include "status_diagnostic_packet.h"
    #include <iostream>
    int main() {
        int64_t packet[128] = {};
        if (!oscar_diagnostic::ErrorLines(packet, "{}", 123, 0, 1).empty()) return 1;
        for (int group = 0; group < 4; ++group) {
            packet[group*32] = 1300; packet[group*32+1] = -7;
            packet[group*32+2] = 1; packet[group*32+3] = 2049;
        }
        std::cout << oscar_diagnostic::ErrorLines(packet, "{\"layer\":\"layer.3\"}", 123, 0, 8);
    }
    '''
    binary = tmp_path / "formatter"
    compiled = subprocess.run([compiler, "-std=c++17", "-x", "c++", "-", "-I", str(ROOT / "csrc"),
                               "-o", str(binary)], input=source, text=True, capture_output=True)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(binary)], text=True, capture_output=True, check=True)
    lines = [json.loads(line.removeprefix("OSCAR_FIRST_ERROR ")) for line in result.stdout.splitlines()]
    assert [line["group"] for line in lines] == ["attention", "rotate", "merge", "store"]
    assert all(line["index"] == 1300 and line["code"] == -7 and len(line["task"]) == 16 for line in lines)
    assert lines[0]["callsite_invocation"] == 8


def test_copy_callback_guard_order_and_bounded_replay_lifetime():
    text = (ROOT / "csrc/status_diagnostic_bindings.cpp").read_text()
    start = text.index("const auto copied =")
    text = text[start:]
    assert text.index("aclrtMemcpyAsync") < text.index("aclrtLaunchHostFunc") < text.index("oscar_status_guard_launch")
    full = (ROOT / "csrc/status_diagnostic_bindings.cpp").read_text()
    assert "registry.entries.size() < 65536" in full and "registry.entries.push_back" in full
    assert "delete " not in full and "aclrtSynchronize" not in full


def test_target_env_propagates_diagnostic_without_changing_candidate():
    from tools.target_cli import target_env, serve_argv
    from tools.serving_variants import variant_config
    config = variant_config(json.loads((ROOT / "configs/target.json").read_text()), "candidate")
    assert target_env(config, {"OSCAR_STATUS_DIAGNOSTICS": "1"})["OSCAR_STATUS_DIAGNOSTICS"] == "1"
    assert "--enable-prefix-caching" in serve_argv(config)
    assert config["experimental_decode_bundle"] and config["experimental_mixed_cv"]
    with pytest.raises(ValueError, match="0 or 1"):
        target_env(config, {"OSCAR_STATUS_DIAGNOSTICS": "true"})


def test_failure_packet_is_not_hidden_in_compact_terminal():
    from tools.phase import _compact_should_mirror
    assert _compact_should_mirror('(Worker_TP3 pid=123) OSCAR_FIRST_ERROR {"code": 7}', {})
    assert _compact_should_mirror('OSCAR_STATUS_DIAGNOSTICS_ACTIVE pid=123', {})


def test_negative_probe_requires_real_replay_and_fault_evidence():
    from tools.probe_status_diagnostics import evidence
    record = dict(group="attention", index=1300, code=7, nonzero_count=1,
                  size=2049, callsite_invocation=3)
    text = ('OSCAR_STATUS_DIAGNOSTICS_ACTIVE pid=123\nOSCAR_DIAGNOSTIC_HEALTHY_REPLAYS=2\n'
            'OSCAR_FIRST_ERROR ' + json.dumps(record))
    result = SimpleNamespace(returncode=1, timed_out=False, interrupted=False, cleanup_complete=True)
    assert evidence(text, result, "attention")[0]
    assert not evidence(text, result, "store")[0]
    assert not evidence("no torch_npu", result, "attention")[0]
    assert not evidence("OSCAR_FIRST_ERROR {truncated", result, "attention")[0]
    assert not evidence("OSCAR_FIRST_ERROR []", result, "attention")[0]
    assert not evidence(text + "\noriginal status_guard failed to Trap", result, "attention")[0]
    result.returncode = 0
    assert not evidence(text, result, "attention")[0]
    result.returncode = 1; result.timed_out = True
    assert not evidence(text, result, "attention")[0]
