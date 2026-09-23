# Archive #55–69/#60–62/#126: a native FIA microprobe must reject placeholder
# LSE and invalid geometry before any device result can be reported as valid.
"""Pure contract checks for the isolated target-NPU FIA experiment."""

import json

import pytest

from tools.probe_native_current_fia import (
    NativeCurrentFIAProbeError,
    _check_shapes,
    _cumulative_lengths,
    _target_geometry,
)


def test_tnd_lengths_are_cumulative_and_nonempty():
    assert _cumulative_lengths((17, 33, 1)) == [17, 50, 51]
    for bad in ((), (0,), (-1,), (True,)):
        with pytest.raises(ValueError):
            _cumulative_lengths(bad)


def test_fia_requires_full_output_and_lse():
    _check_shapes((50, 6, 256), (50, 6, 1), 50, 6, 256)
    for output, lse in (((50, 6, 256), (0,)), ((50, 6, 256), (1,)),
                        ((50, 6, 256), (50, 6)), ((50, 256), (50, 6, 1))):
        with pytest.raises(NativeCurrentFIAProbeError):
            _check_shapes(output, lse, 50, 6, 256)


def test_target_head_geometry_must_match_tp(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    config_path = model / "config.json"
    config_path.write_text(json.dumps({"text_config": {"num_attention_heads": 24,
                                                        "num_key_value_heads": 4,
                                                        "head_dim": 256}}))
    target = {"model": str(model), "tensor_parallel_size": 4}
    assert _target_geometry(target) == (6, 1, 256)
    config_path.write_text(json.dumps({"text_config": {"num_attention_heads": 0,
                                                        "num_key_value_heads": 4,
                                                        "hidden_size": 6144}}))
    with pytest.raises(NativeCurrentFIAProbeError):
        _target_geometry(target)
