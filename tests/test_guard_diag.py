# 档案 #142/#157：归因开关的默认路径必须逐字节等价于融合启动；诊断路径按族归因。
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


def test_diag_path_launches_per_family_and_names_the_trapped_family(monkeypatch):
    monkeypatch.setenv("OSCAR_GUARD_HOST_SYNC", "1")
    npu = _Npu()
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    segments = _segments()
    segments[2] = ("merge", torch.ones(3, 2, dtype=torch.int32))
    ops = Ops(fail_on=segments[2][1])
    with pytest.raises(OscarReadinessError, match="site=forward:layer family=merge"):
        run_status_guard(ops, "forward:layer", segments)
    # The trapped family's own synchronize never runs; the raise is the sync point.
    assert len(ops.calls) == 3 and npu.syncs == 3  # producer drain + two healthy guards


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
    assert len(ops.calls) == 1 and npu.syncs == 2
    a, b, c, d = ops.calls[0]
    assert a.numel() == 4 and b.numel() == c.numel() == d.numel() == 0


def test_upstream_async_failure_is_not_misattributed_to_first_guard_family(monkeypatch):
    monkeypatch.setenv("OSCAR_GUARD_HOST_SYNC", "1")
    class FailedNpu:
        def synchronize(self):
            raise RuntimeError("earlier asynchronous kernel failed")
    monkeypatch.setattr(torch, "npu", FailedNpu(), raising=False)
    ops = Ops()
    with pytest.raises(OscarReadinessError, match="site=forward:layer family=upstream") as caught:
        run_status_guard(ops, "forward:layer", _segments())
    assert not ops.calls
    assert "earlier asynchronous" in str(caught.value.__cause__)
