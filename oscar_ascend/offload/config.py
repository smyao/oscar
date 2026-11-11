"""Offload launch contract; no native config or numerical policy mutation.

Archive #34/#36/#127/#128: physical groups and native none/align stay intact.
Archive #77/#78: usable in the launcher without importing device libraries.
Archive #94/#95/#125: invalid transport configuration fails visibly.
"""
from __future__ import annotations

CONNECTOR = "OscarAscendStoreConnector"
MODULE = "oscar_ascend.offload.connector"
ALIGNMENT = 2 * 1024 * 1024


def offload_options(config: dict) -> dict | None:
    value = config.get("kv_offload")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("kv_offload must be an object")
    allowed = {"enabled", "backend", "load_async", "max_batch_objects", "timeout_seconds"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown kv_offload options: {sorted(unknown)}")
    if type(value.get("enabled", False)) is not bool:
        raise ValueError("kv_offload.enabled must be a boolean")
    load_async = value.get("load_async", True)
    if type(load_async) is not bool:
        raise ValueError("kv_offload.load_async must be a boolean")
    if not value.get("enabled", False):
        return None
    if value.get("backend", "memcache") != "memcache":
        raise ValueError("OSCAR offload requires the AscendStore memcache backend")
    batch = value.get("max_batch_objects", 256)
    timeout = value.get("timeout_seconds", 60)
    if type(batch) is not int or batch < 1 or batch > 4096:
        raise ValueError("kv_offload.max_batch_objects must be in [1, 4096]")
    if type(timeout) not in (int, float) or not 0 < timeout <= 600:
        raise ValueError("kv_offload.timeout_seconds must be in (0, 600]")
    if config.get("experimental_request_windows", False):
        raise ValueError("offload requires the production physical-page windows")
    return {"enabled": True, "backend": "memcache", "use_layerwise": False,
            "load_async": load_async, "max_batch_objects": batch, "timeout_seconds": timeout}


def transfer_config(config: dict) -> dict | None:
    options = offload_options(config)
    if options is None:
        return None
    return {"kv_connector": CONNECTOR, "kv_connector_module_path": MODULE,
            "kv_role": "kv_both", "kv_load_failure_policy": "fail",
            "kv_connector_extra_config": options}


def validate_transfer_config(transfer) -> None:
    """Only this byte-complete adapter can enter the hybrid raw allocation."""
    if (getattr(transfer, "kv_connector", None) != CONNECTOR
            or getattr(transfer, "kv_connector_module_path", None) != MODULE
            or getattr(transfer, "kv_role", None) != "kv_both"):
        raise ValueError("OSCAR requires its external AscendStore byte-layout connector")
    extra = getattr(transfer, "kv_connector_extra_config", {})
    if not isinstance(extra, dict) or extra.get("enabled") is not True:
        raise ValueError("OSCAR offload requires an explicit enabled connector contract")
    if extra.get("backend") != "memcache" or extra.get("use_layerwise", False):
        raise ValueError("OSCAR offload requires non-layerwise memcache transfers")
    checked = offload_options({"kv_offload": {k: v for k, v in extra.items() if k != "use_layerwise"}})
    if checked != extra:
        raise ValueError("OSCAR offload connector options differ from the canonical launch contract")
    if getattr(transfer, "kv_load_failure_policy", "fail") != "fail":
        raise ValueError("OSCAR offload load failures must fail the request")


def allocate_aligned_bytes(torch, size: int, device):
    """Same 2 MiB ownership/alignment as native AscendStore allocations."""
    if type(size) is not int or size <= 0:
        raise ValueError("KV allocation size must be positive")
    owner = torch.zeros(size + ALIGNMENT, dtype=torch.int8, device=device)
    offset = (-owner.data_ptr()) % ALIGNMENT
    # The view retains its storage owner throughout graph capture and replay.
    return owner[offset:offset + size]
