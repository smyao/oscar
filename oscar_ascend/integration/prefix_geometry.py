"""Archive #127/#128/#156: physical, scheduler and hash units are distinct.

Pure configuration validation, shared with service evidence checks. Resolution
matches vLLM v0.23.0 resolve_kv_cache_block_sizes for this local hybrid target.
"""
from math import gcd, lcm

from .runtime_api import OscarReadinessError


def validate_prefix_geometry(layout):
    """Reject contradictory effective geometry; returns no inferred evidence."""
    enabled = layout.get("prefix_caching")
    mode = layout.get("mamba_mode")
    if type(enabled) is not bool or mode not in {"none", "align", "all"}:
        raise OscarReadinessError("cache layout requires explicit prefix/mamba mode")
    names = ("native_full_block_size", "mamba_block_size", "physical_full_block_size",
             "scheduler_lcm_block_size", "hash_block_size")
    if any(type(layout.get(k)) is not int or layout[k] <= 0 for k in names):
        raise OscarReadinessError("effective block/hash sizes must be positive integers")
    native, mamba, physical, scheduler, hashed = (layout[k] for k in names)
    if physical % 128 or scheduler != lcm(physical, mamba):
        raise OscarReadinessError("physical virtual128/scheduler LCM geometry disagrees")
    if enabled:
        if mode != "align" or mamba != native or physical % mamba:
            raise OscarReadinessError("prefix caching requires aligned native Mamba/FULL geometry")
        # A smaller explicit native hash override is valid: H divides groups,
        # not the reverse. There is no requirement H % Bm == 0.
        if physical % hashed or mamba % hashed:
            raise OscarReadinessError("hash block size must divide every native group")
    elif hashed != scheduler:
        raise OscarReadinessError("local uncached hybrid hashing must use scheduler LCM")


def resolve_prefix_geometry(cache, physical):
    enabled = getattr(cache, "enable_prefix_caching", None)
    mamba = getattr(cache, "mamba_block_size", None)
    native = getattr(cache, "block_size", None)
    mode = getattr(cache, "mamba_cache_mode", None)
    if any(type(v) is not int or v <= 0 for v in (physical, mamba, native)):
        raise OscarReadinessError("native cache configuration has invalid block sizes")
    scheduler = lcm(physical, mamba)
    requested_hash = getattr(cache, "hash_block_size", None)
    resolved_hash = (gcd(physical, mamba) if requested_hash is None else requested_hash) if enabled else scheduler
    layout = dict(prefix_caching=enabled, mamba_mode=mode, native_full_block_size=native,
                  mamba_block_size=mamba, physical_full_block_size=physical,
                  scheduler_lcm_block_size=scheduler, hash_block_size=resolved_hash,
                  hash_block_size_requested=requested_hash)
    validate_prefix_geometry(layout)
    return layout
