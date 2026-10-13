# 档案 #142/#157/#158：默认路径逐字节等价于融合启动；诊断按族/设备谓词归因。
import pytest
import torch

from oscar_ascend.integration import guard_diag
from oscar_ascend.integration.guard_diag import guard_diag_enabled, run_status_guard
from oscar_ascend.integration.runtime_api import OscarReadinessError


class Ops:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def status_guard(self, a, b, c, d):
        self.calls.append((a, b, c, d))
        if self.fail_on is not None and a.data_ptr() == self.fail_on.data_ptr():
            raise RuntimeError("507035 vector core exception")


class _Npu:
    def __init__(self):
        self.syncs = 0

    def synchronize(self):
        self.syncs += 1


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(guard_diag, "_ENABLED", None)
    monkeypatch.delenv("OSCAR_GUARD_HOST_SYNC", raising=False)
    yield


def _segments():
    return [("attention", torch.zeros(6, 2, dtype=torch.int32)),
            ("rotate", torch.zeros(3, 2, dtype=torch.int32)),
            ("merge", torch.zeros(3, 2, dtype=torch.int32)),
            ("store", torch.zeros(3, 1, dtype=torch.int32))]


def test_default_path_is_one_fused_launch_without_sync(monkeypatch):
    npu = _Npu()
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    ops = Ops()
    segments = _segments()
    run_status_guard(ops, "forward:layer", segments)
    assert len(ops.calls) == 1
    assert [tuple(t.shape) for t in ops.calls[0]] == [tuple(s.shape) for _, s in segments]
    assert npu.syncs == 0
    assert guard_diag_enabled() is False


def test_capture_path_stays_fused_even_when_enabled(monkeypatch):
    monkeypatch.setenv("OSCAR_GUARD_HOST_SYNC", "1")
    npu = _Npu()
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    ops = Ops()
    run_status_guard(ops, "forward:layer", _segments(), capture=True)
    assert len(ops.calls) == 1 and npu.syncs == 0


def test_diag_path_attributes_the_frozen_trap_without_readback(monkeypatch):
    monkeypatch.setenv("OSCAR_GUARD_HOST_SYNC", "1")
    npu = _Npu()
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    segments = _segments()
    segments[2][1][1, 0] = 4
    before = [tensor.clone() for _, tensor in segments]
    class PredicateOps(Ops):
        def status_guard(self, a, b, c, d):
            super().status_guard(a, b, c, d)
            # This is a CPU implementation of the frozen guard's test only,
            # never production code or an NPU readback fallback.
            if bool(torch.any(a != 0)):
                raise RuntimeError("507035 vector core exception")
    ops = PredicateOps()
    def forbidden(*args, **kwargs):
        raise AssertionError("production guard must not read device contents on host")
    for name in ("item", "cpu", "tolist", "numpy"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    with pytest.raises(OscarReadinessError, match=r"site=forward:layer family=merge code=4"):
        run_status_guard(ops, "forward:layer", segments)
    assert len(ops.calls) == 16 and npu.syncs == 16
    assert all(torch.equal(tensor, saved) for (_, tensor), saved in zip(segments, before))


def test_diag_path_skips_empty_families(monkeypatch):
    monkeypatch.setenv("OSCAR_GUARD_HOST_SYNC", "1")
    npu = _Npu()
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    ops = Ops()
    run_status_guard(ops, "current_only.prepare",
                     [("metadata", torch.zeros(4, dtype=torch.int32)),
                      ("rotate", torch.zeros(0, dtype=torch.int32)),
                      ("merge", torch.zeros(0, dtype=torch.int32)),
                      ("store", torch.zeros(0, dtype=torch.int32))])
    assert len(ops.calls) == 6 and npu.syncs == 7
    a, b, c, d = ops.calls[0]
    assert a.numel() == 4 and b.numel() == c.numel() == d.numel() == 0


@pytest.mark.parametrize("code", [1, 2, 3, 4, 5, -1, 6, 2147483647, -2147483648])
def test_device_predicates_cover_all_status_domains_without_losing_errors(monkeypatch, code):
    monkeypatch.setenv("OSCAR_GUARD_HOST_SYNC", "1")
    monkeypatch.setattr(torch, "npu", _Npu(), raising=False)
    seen = []
    class PredicateOps:
        def status_guard(self, a, b, c, d):
            seen.append(a.clone())
            if bool(torch.any(a != 0)):
                raise RuntimeError("frozen guard trap")
    status = torch.tensor([0, 0, code, 0], dtype=torch.int32)
    empty = status[:0]
    expected = str(code) if 1 <= code <= 5 else "unknown"
    with pytest.raises(OscarReadinessError, match="family=attention code=" + expected):
        run_status_guard(PredicateOps(), "forward:mtp", [("attention", status),
            ("rotate", empty), ("merge", empty), ("store", empty)])
    assert torch.equal(status, torch.tensor([0, 0, code, 0], dtype=torch.int32))
    assert torch.equal(seen[-1], torch.tensor([0, 0, 1, 0], dtype=torch.int32))


def test_prior_async_failure_is_not_misattributed_to_current_family_or_code(monkeypatch):
    monkeypatch.setenv("OSCAR_GUARD_HOST_SYNC", "1")
    class PriorFailure:
        def synchronize(self):
            raise RuntimeError("previous target graph guard Trap")
    monkeypatch.setattr(torch, "npu", PriorFailure(), raising=False)
    ops = Ops()
    with pytest.raises(OscarReadinessError, match="family=prior_work code=unavailable"):
        run_status_guard(ops, "forward:mtp", _segments())
    assert ops.calls == []
