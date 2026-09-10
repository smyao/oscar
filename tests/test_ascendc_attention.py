import pytest

from oscar_ascend.kernels import ascendc_attention


def test_ascendc_is_opt_in(monkeypatch):
    monkeypatch.delenv("OSCAR_ASCEND_USE_ASCENDC", raising=False)
    monkeypatch.setattr(ascendc_attention, "_resolve_op", lambda: object())
    assert ascendc_attention.ascendc_mode() == "0"
    assert not ascendc_attention.ascendc_enabled()


def test_required_ascendc_fails_closed(monkeypatch):
    monkeypatch.setenv("OSCAR_ASCEND_USE_ASCENDC", "required")
    monkeypatch.setattr(ascendc_attention, "_resolve_op", lambda: None)
    with pytest.raises(RuntimeError, match="required"):
        ascendc_attention.ascendc_enabled()


def test_invalid_ascendc_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("OSCAR_ASCEND_USE_ASCENDC", "maybe")
    with pytest.raises(ValueError, match="OSCAR_ASCEND_USE_ASCENDC"):
        ascendc_attention.ascendc_mode()
