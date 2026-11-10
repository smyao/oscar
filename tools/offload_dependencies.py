"""Archive #74/#75/#94/#175/#176: preserve explicitly installed SDK builds.

This selects installation arguments only. Runtime imports still surface missing
SDKs through their existing errors; no SDK discovery or replacement is implied.
"""
from __future__ import annotations

from oscar_ascend.offload.config import offload_options


def offload_install_dependencies(config: dict) -> list[str]:
    source = config.get("offload_sdk_source", "pip")
    if source not in ("pip", "installed"):
        raise ValueError("offload_sdk_source must be 'pip' or 'installed'")
    if offload_options(config) is None:
        return []
    if source == "installed":
        return ["pyzmq"]
    return ["memfabric-hybrid", "memcache-hybrid", "pyzmq"]
