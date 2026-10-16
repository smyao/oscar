"""Archive #148/#153: reject an unqualified, unimplemented numerical route.

The BF16 history experiment has not passed the frozen CPU numerical gate.
Keeping its config key must never be mistaken for installing a production op.
This module is deliberately independent of torch, device loading and probes.
"""

BF16_HISTORY_FLAG = "experimental_bf16_history"


def validate_bf16_history(config, *, variant=None) -> bool:
    """Return the disabled effective value or reject an unsupported request.

    Explicit baseline/native selection clears a valid boolean request, just as
    the existing experiment presets do. Type validation precedes that clearing.
    Missing/False leaves all established serving configuration untouched.
    """
    enabled = config.get(BF16_HISTORY_FLAG, False)
    if type(enabled) is not bool:
        raise ValueError(f"{BF16_HISTORY_FLAG} must be an explicit boolean")
    if variant in {"baseline", "native"}:
        return False
    if enabled:
        raise ValueError(
            f"{BF16_HISTORY_FLAG} is unavailable: the BF16 history candidate "
            "has not passed the frozen CPU numerical gate and has no production "
            "implementation; keep this flag false")
    return False
