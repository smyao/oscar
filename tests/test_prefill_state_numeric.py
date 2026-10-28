"""Archive #163/#174: actual recurrent equations expose finite stale state.

This independent native-reference witness protects both the incoming state and
the outgoing checkpoint. It is not a model-quality or device-performance test.
"""
import importlib.util
from pathlib import Path

import pytest


def _repro():
    path = Path(__file__).resolve().parents[1] / "reports/gdn_prefill_state_20261028/numeric_repro.py"
    spec = importlib.util.spec_from_file_location("gdn_prefill_numeric_repro", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reproduce


@pytest.mark.parametrize("stale", [-42.0, -1.0, 0.0, 1.0, 42.0])
def test_native_fold_reads_finite_stale_state_and_selector_only_misses_commit(stale):
    result = _repro()(stale)
    assert result["folded_accepted"] == [4]
    assert result["gdn_old_output"] != result["gdn_correct_output"]
    assert result["gdn_correct_output"] == [0.05517578125, 0.07177734375, 0.080078125, 0.08447265625]
    assert result["gdn_correct_final_col0"] == .953125
    assert result["gdn_selector_only_col0"] == .625
    assert result["next_gdn_correct"] == .08642578125
    assert result["next_gdn_selector_only"] == .07177734375
    assert result["next_conv_correct"] == 14.0
    assert result["next_conv_selector_only"] == 6.5
