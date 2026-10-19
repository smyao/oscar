"""Archive #34/#36/#148/#157/#164-165: APC must test the serving dtype.

CPU allocations and call witnesses only; no NPU execution is simulated.
The independent expected byte count is the public D256 BF16 MDL ABI.
"""
import json

import pytest
import torch

from tools import probe_prefix_cache as probe
from tools.serving_variants import variant_config


def config():
    source = json.loads((probe.ROOT / "configs/target.json").read_text())
    return variant_config({**source, "enable_prefix_caching": True}, "candidate")


def test_apc_reader_matches_current_production_precision():
    assert probe.selected_reader(config()) == "attention_bf16_history_mdl_out"


def test_apc_actual_allocator_reserves_bf16_mdl_arena():
    state = probe.allocate(torch, torch.device("cpu"), 2)
    assert state["workspace"].dtype == torch.uint8
    assert state["workspace"].numel() == 2 * 655_360
    assert state["qr"].dtype == torch.bfloat16 and state["qr"].shape[-1] == 272
    # All public output/state precision stays FP32; operands are separate.
    for name in ("partial", "lse", "output", "output_lse"):
        assert state[name].dtype == torch.float32


@pytest.mark.parametrize("flag", ["experimental_bf16_history",
    "experimental_mdl_decode", "experimental_c4_member_pipeline"])
def test_apc_rejects_an_incomplete_or_retired_production_bundle(flag):
    with pytest.raises(ValueError):
        probe.selected_reader({**config(), flag: False})


def test_apc_source_binding_covers_actual_reader_helpers_and_policy():
    bound = probe.source_binding()
    assert {"tools/probe_prefix_cache.py", "tools/serving_variants.py",
        "oscar_ascend/bf16_history_policy.py", "csrc/bf16_history_bindings.cpp",
        "csrc/include/oscar_bf16_history_launch.h",
        "csrc/kernels/attention_bf16_history_mdl.cpp",
        "csrc/kernels/attention_bf16_history_operands.h",
        "csrc/kernels/attention_striped_unpack_bf16.h"} <= bound.keys()
