"""Archive #27/#29/#74: honor SDKs explicitly sourced by the user launcher.

Only dependency installation changes; no SDK discovery, configuration writes,
service startup, or changes to the inference path occur here.
"""
from __future__ import annotations

import os

from oscar_ascend.offload.config import offload_options


def offload_install_dependencies(config: dict, environ=None) -> list[str]:
    if offload_options(config) is None:
        return []
    environ = os.environ if environ is None else environ
    source = environ.get("OSCAR_OFFLOAD_SDK_SOURCE", "pip")
    if source == "installed":
        return ["pyzmq"]
    if source == "pip":
        return ["memfabric-hybrid", "memcache-hybrid", "pyzmq"]
    raise ValueError("OSCAR_OFFLOAD_SDK_SOURCE must be 'installed' or 'pip'")
