"""Archive #127/#128/#156: executable cache geometry, independent of vLLM imports."""
from types import SimpleNamespace

import pytest

from oscar_ascend.integration.prefix_geometry import resolve_prefix_geometry, validate_prefix_geometry
from oscar_ascend.integration.runtime_api import OscarReadinessError


def config(**changes):
    return SimpleNamespace(**(dict(enable_prefix_caching=True, block_size=768,
        mamba_block_size=768, mamba_cache_mode="align", hash_block_size=None) | changes))


def test_align_and_uncached_hash_units_are_not_interchangeable():
    on = resolve_prefix_geometry(config(), 2304)
    assert (on["scheduler_lcm_block_size"], on["hash_block_size"]) == (2304, 768)
    off = resolve_prefix_geometry(config(enable_prefix_caching=False, mamba_cache_mode="none",
                                        mamba_block_size=262144), 2816)
    assert (off["scheduler_lcm_block_size"], off["hash_block_size"]) == (2883584, 2883584)
    assert on["hash_block_size_requested"] is None


@pytest.mark.parametrize("override", [128, 256, 768])
def test_native_smaller_hash_override_is_legal(override):
    assert resolve_prefix_geometry(config(hash_block_size=override), 2304)["hash_block_size"] == override


@pytest.mark.parametrize("changes", [
    {"mamba_cache_mode": "none"}, {"mamba_block_size": 262144},
    {"enable_prefix_caching": "true"}, {"hash_block_size": 512},
    {"hash_block_size": 0}, {"hash_block_size": True},
])
def test_rejects_inconsistent_config(changes):
    with pytest.raises(OscarReadinessError):
        resolve_prefix_geometry(config(**changes), 2304)


@pytest.mark.parametrize("changes", [
    {"physical_full_block_size": 2816, "scheduler_lcm_block_size": 2883584},
    {"scheduler_lcm_block_size": 768}, {"hash_block_size": None},
    {"physical_full_block_size": 2305},
])
def test_rejects_contradictory_runtime_evidence(changes):
    layout = resolve_prefix_geometry(config(), 2304) | changes
    with pytest.raises(OscarReadinessError):
        validate_prefix_geometry(layout)
