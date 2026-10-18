"""Archive #34/#36/#126/#148/#153: stable launch, qualified internal routes.

These are host ABI/allocation checks; CPU tensor operations are not NPU math.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from oscar_ascend.ops.cv_dispatch import select_cv_op, CLUSTER_CV_OPS
from oscar_ascend.ops.hybrid_layout import pack_rotation_nz
from oscar_ascend.runtime import AscendRuntimeProvider, WorkspaceGeometry
from oscar_ascend.integration.runtime_api import OscarReadinessError
from tools.serving_variants import variant_config
from tools import observe_serve

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('tokens,max_q,later,expected', [
    (128, 4, False, 'attention_cv_bf16_decode_out'),
    (128, 1, True, 'attention_cv_bf16_decode_q1_out'),
    (2048, 2048, False, 'attention_cv_bf16_cluster4_out'),
    (16384, 16384, False, 'attention_cv_bf16_cluster16_out'),
    (128, 5, False, 'attention_cv_bf16_cluster4_out'),
    (16384, 5, False, 'attention_cv_bf16_cluster4_out'),
])
def test_shared_candidate_selects_complete_reader(tokens, max_q, later, expected):
    from oscar_ascend.ops.contracts import SOURCE_CAPABILITIES
    config = variant_config(json.loads((ROOT/'configs/target.json').read_text()), 'candidate')
    flags = {name: config.get('experimental_'+name, False) for name in (
        'fast_unpack', 'mixed_cv', 'striped_cache', 'decode_bundle', 'mdl_decode',
        'c4_member_pipeline', 'c16_member_pipeline', 'c16_l1_reuse', 'hybrid_l1_decode',
        'bf16_attention')}
    assert select_cv_op(16, 6, 1, tokens, max_q, q1_draft=later, **flags) == expected
    assert expected in SOURCE_CAPABILITIES
    assert (expected in CLUSTER_CV_OPS) == (max_q >= 5)


def test_c16_capacity_is_fixed_and_counted_before_kv_budget():
    base = WorkspaceGeometry(16384, 6, 1, 256, cube_cores=20, history_cluster_size=16)
    candidate = replace(base, c16_member_pipeline=True)
    assert candidate.cv_bytes == 20 * 5652480
    assert candidate.total_bytes - base.total_bytes == 20 * 655360
    assert replace(candidate, tokens=32768).cv_bytes == candidate.cv_bytes
    b1 = replace(base, c16_l1_reuse=True)
    assert b1.cv_bytes == 20 * 7487488
    assert b1.total_bytes - base.total_bytes == 20 * 2490368
    assert replace(b1, c16_member_pipeline=True).cv_bytes == b1.cv_bytes


def test_c16_l1_reader_reaches_real_forward_with_natural_rv_and_stats():
    from test_runtime import test_forward_dispatches_native_draft_strides_and_int32_slots_in_order
    test_forward_dispatches_native_draft_strides_and_int32_slots_in_order(
        8192, 20, 1, 8192, 16, False, True, True, True, _c16_l1=True)


def test_rotation_nz_is_bitwise_address_permutation_including_nonfinite():
    words = torch.arange(65536, dtype=torch.int32).reshape(256, 256)
    words[0, :4] = torch.tensor([0, -2147483648, 2139095040, 2143289345], dtype=torch.int32)
    natural = words.view(torch.float32)
    packed = pack_rotation_nz(natural).view(torch.int32).flatten()
    row = torch.arange(256)[:, None]
    col = torch.arange(256)[None, :]
    offsets = (col // 8) * 2048 + row * 8 + col % 8
    assert torch.equal(packed[offsets], words)
    assert packed.data_ptr() != words.data_ptr()


@pytest.mark.parametrize('flag', ['experimental_hybrid_l1_decode', 'experimental_c16_member_pipeline',
                                 'experimental_c16_l1_reuse', 'experimental_prepare_reuse',
                                 'experimental_bf16_attention'])
def test_new_flags_reject_non_boolean_values(flag):
    with pytest.raises(ValueError, match='explicit boolean'):
        variant_config({flag: 1}, 'candidate')
    with pytest.raises(OscarReadinessError, match='explicit boolean'):
        AscendRuntimeProvider({flag: 1})


def test_frozen_fe0_kernels_and_acceptance_are_unchanged():
    for name, expected in observe_serve.FE0_KERNEL_SHA256.items():
        assert hashlib.sha256((ROOT/'csrc/kernels'/name).read_bytes()).hexdigest() == expected
    # Stable's launch fixture is independently captured from the user's base.
    frozen = json.loads((ROOT/'reports/stable_32k_64k_20261015/frozen_launch.json').read_text())
    from test_stable_launch_frozen import authorized_model_arguments
    from tools.target_cli import serve_argv
    from tools.serving_variants import variant_config
    config=variant_config(json.loads((ROOT/'configs/target.json').read_text()),'candidate')
    assert serve_argv(config)==authorized_model_arguments(frozen['cases']['direct_prefix_false']['argv'])
    assert hashlib.sha256((ROOT/'configs/acceptance.json').read_bytes()).hexdigest() == frozen['acceptance_sha256']
