"""Archive #145/#164: current candidate uses tuned C16; other routes unchanged."""
import pytest


@pytest.mark.parametrize('enabled,name', [
    (True, 'attention_cv_aligned_cluster16_out'),
    (False, 'attention_cv_aligned_cluster16_control_out'),
])
def test_actual_forward_selects_tuned_or_preserved_control(enabled, name):
    from test_runtime import test_forward_dispatches_native_draft_strides_and_int32_slots_in_order
    test_forward_dispatches_native_draft_strides_and_int32_slots_in_order(
        8192, 1, 1, 8192, 16, False, True, True, True,
        _member_pipeline=True, _c16_l1=True, _bf16_expected=name,
        _aligned=True, _c16_tuned=enabled)


def test_targeted_flag_cannot_change_legacy_precision_or_storage():
    from tools.serving_variants import variant_config
    with pytest.raises(ValueError, match='aligned BF16'):
        variant_config({'experimental_c16_tuned': True}, None)
