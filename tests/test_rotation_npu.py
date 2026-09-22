# Archive G25-G34/#13-22/#37-49/#69/#98/#111: independent CPU-prepared
# goldens -> real AscendC NPU dispatch -> explicit completion -> exact cache,
# raw-window/tag/guard/status comparisons. CPU pytest skips are NOT acceptance.
"""Opt-in real-device rotation gate; a requested but unavailable NPU is a failure.

Run with OSCAR_RUN_NPU_TESTS=1, OSCAR_TARGET_CONFIG=/absolute/target.json and
ASCEND_RT_VISIBLE_DEVICES exactly matching that config's selected four cards.
All 26 golden cases run on each of the four selected logical devices. The
CPU oracle only prepares external expected data; operators under test always
execute through torch.ops.oscar_ascend_ops on NPU, never the oracle.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    os.environ.get("OSCAR_RUN_NPU_TESTS") != "1",
    reason="not_run: real NPU rotation gate requires OSCAR_RUN_NPU_TESTS=1",
)
CASE_NAMES = tuple(
    name
    for dim in (64, 128, 256)
    for hadamard in (0, 1)
    for name in (
        f"rotate_{dim}_{hadamard}",
        f"rotate_masked_{dim}_{hadamard}",
        f"rotate_store_{dim}_{hadamard}_0.00",
        f"rotate_store_{dim}_{hadamard}_0.95",
    )
) + ("rotate_store_errors", "rotate_store_sink_only")


@pytest.fixture(scope="module")
def rotation_npu_context(tmp_path_factory):
    # Validate the current task's ownership before loading a backend or making
    # a context. An explicitly requested gate must fail, never importorskip.
    configured = os.environ.get("OSCAR_TARGET_CONFIG")
    if not configured:
        pytest.fail("OSCAR_TARGET_CONFIG is mandatory for the requested NPU rotation gate")
    path = Path(configured).expanduser().resolve()
    config = json.loads(path.read_text())
    from tools.target_cli import target_env
    selected = target_env(config, base={})["ASCEND_RT_VISIBLE_DEVICES"]
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != selected:
        pytest.fail("ASCEND_RT_VISIBLE_DEVICES must exactly match OSCAR_TARGET_CONFIG.devices; no inherited card selection")
    import torch
    import torch_npu  # noqa: F401 -- a missing driver/backend must fail this opt-in gate.
    if not torch.npu.is_available() or torch.npu.device_count() != 4:
        pytest.fail("all four explicitly selected NPU devices must be available")
    from oscar_ascend.ops.loader import require_capabilities
    require_capabilities({"rotate_out", "rotate_clip_store_out"})
    from tools.generate_rotation_cpu_cases import generate
    cases = generate(tmp_path_factory.mktemp("rotation-npu-goldens"))
    indexed = {Path(case["path"]).name: case for case in cases}
    assert set(indexed) == set(CASE_NAMES), "golden inventory changed; keep every case in the NPU gate"
    acceptance = json.loads((ROOT / "configs/acceptance.json").read_text())
    assert acceptance["frozen_before_measurement"] is True
    assert acceptance["pack_unpack"]["exact"] is True
    return torch, config, indexed, acceptance["store_dequant"]


def _cpu_tensor(torch, path, dtype, shape):
    # Use owned bytes: frombuffer must not alias a read-only bytes object.
    content = bytearray(Path(path).read_bytes())
    assert content, f"empty golden/input file: {path}"
    result = torch.frombuffer(content, dtype=torch.uint8).clone().view(dtype)
    assert result.numel() == math.prod(shape), f"golden/input byte count mismatch: {path}"
    return result.reshape(shape)


def _exact(torch, actual, directory, name):
    expected = _cpu_tensor(torch, directory / f"{name}.bin", actual.dtype, tuple(actual.shape))
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


def _rotate_case(torch, directory, device, tolerance):
    fields = (directory / "shape.txt").read_text().split()
    assert len(fields) == 6
    rows, dim, dtype_code, hadamard, heads, masked = map(int, fields)
    dtype = {0: torch.float32, 1: torch.float16, 2: torch.bfloat16}[dtype_code]
    assert heads > 0 and rows % heads == 0
    shape = (rows // heads, heads, dim)
    inputs = _cpu_tensor(torch, directory / "input.bin", dtype, shape).to(device)
    rotation = _cpu_tensor(torch, directory / "rotation.bin", torch.float32, (dim, dim)).to(device)
    output = torch.full(shape, torch.nan, dtype=torch.float32, device=device)
    status = torch.full(shape[:2], -123, dtype=torch.int32, device=device)
    slots = _cpu_tensor(torch, directory / "slots.bin", torch.int64, (shape[0],)).to(device) if masked else None
    torch.ops.oscar_ascend_ops.rotate_out(inputs, rotation, output, status, bool(hadamard), slots)
    torch.npu.synchronize()
    _exact(torch, status, directory, "expected_status")
    expected = _cpu_tensor(torch, directory / "expected_output.bin", torch.float32, tuple(output.shape))
    # Expected NaNs occur only in the deliberately active error row, whose
    # exact status 2 was checked above; padding must match finite zeros.
    torch.testing.assert_close(output.cpu(), expected, atol=tolerance["atol"], rtol=tolerance["rtol"], equal_nan=True)
    if slots is not None:
        inactive = _cpu_tensor(torch, directory / "slots.bin", torch.int64, (shape[0],)) < 0
        torch.testing.assert_close(output.cpu()[inactive], torch.zeros_like(expected[inactive]), atol=0, rtol=0)


def _rotate_store_case(torch, directory, device):
    fields = (directory / "shape.txt").read_text().split()
    assert len(fields) == 16
    n, heads, dim, dtype_code, block, blocks, offset, stride, ks, vs, ts, sink, recent = map(int, fields[:13])
    k_clip, v_clip = map(float, fields[13:15])
    hadamard = bool(int(fields[15]))
    dtype = {0: torch.float32, 1: torch.float16, 2: torch.bfloat16}[dtype_code]
    key = _cpu_tensor(torch, directory / "key.bin", dtype, (n, heads, dim)).to(device)
    value = _cpu_tensor(torch, directory / "value.bin", dtype, (n, heads, dim)).to(device)
    rk = _cpu_tensor(torch, directory / "rk.bin", torch.float32, (dim, dim)).to(device)
    rv = _cpu_tensor(torch, directory / "rv.bin", torch.float32, (dim, dim)).to(device)
    slots = _cpu_tensor(torch, directory / "slots.bin", torch.int64, (n,)).to(device)
    positions = _cpu_tensor(torch, directory / "positions.bin", torch.int64, (n,)).to(device)
    packed = torch.full((offset + blocks * stride,), 173, dtype=torch.uint8, device=device)
    raw_key_storage = torch.full((blocks * ks * 2,), 173, dtype=torch.uint8, device=device)
    raw_value_storage = torch.full((blocks * vs * 2,), 173, dtype=torch.uint8, device=device)
    tag_storage = torch.full((blocks * ts * 8,), 173, dtype=torch.uint8, device=device)
    window = sink + recent
    raw_key = raw_key_storage.view(torch.bfloat16).as_strided(
        (blocks, window, heads, dim), (ks, heads * dim, dim, 1))
    raw_value = raw_value_storage.view(torch.bfloat16).as_strided(
        (blocks, window, heads, dim), (vs, heads * dim, dim, 1))
    tags = tag_storage.view(torch.int64).as_strided((blocks, window), (ts, 1))
    status = torch.full((n, heads), -123, dtype=torch.int32, device=device)
    torch.ops.oscar_ascend_ops.rotate_clip_store_out(
        key, value, rk, rv, slots, positions, packed, raw_key, raw_value, tags,
        status, block, blocks, offset, stride, sink, recent, k_clip, v_clip, hadamard)
    torch.npu.synchronize()
    # Compare complete backing allocations, including every guard gap and all
    # untouched rows; comparing compacted views alone would miss stray DMA.
    for actual, name in (
        (status, "expected_status"), (packed, "expected_packed"),
        (raw_key_storage, "expected_raw_key"), (raw_value_storage, "expected_raw_value"),
        (tag_storage, "expected_tags"),
    ):
        _exact(torch, actual, directory, name)


@pytest.mark.parametrize("logical_device", range(4), ids=lambda index: f"logical-npu-{index}")
@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_rotation_ascendc_real_npu(rotation_npu_context, logical_device, case_name, record_property):
    torch, config, cases, tolerance = rotation_npu_context
    torch.npu.set_device(logical_device)
    device = torch.device(f"npu:{logical_device}")
    case = cases[case_name]
    directory = Path(case["path"])
    if case["op"] == "rotate":
        _rotate_case(torch, directory, device, tolerance)
    elif case["op"] == "rotate_store":
        _rotate_store_case(torch, directory, device)
    else:
        pytest.fail(f"unrecognized real-NPU rotation golden case: {case}")
    record_property("execution_backend", "npu_ascendc")
    record_property("physical_device", config["devices"][logical_device])
    record_property("device_completion", "passed")
    record_property("graph_capture", "not_run")
    record_property("graph_replay", "not_run")
    record_property("performance", "not_run")
