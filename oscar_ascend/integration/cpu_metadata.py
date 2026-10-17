"""Archive #34/#36/#69/#140/#142/#157-165: checked native CPU mirrors.

Only native integer CPU metadata may enter Python shape/partition planning.
Keep the single tensor-to-list conversion here, after an explicit device,
dtype, rank and capacity check; cached plans do not waive these checks.
This never copies a device tensor to the host or reads historical KV data.
"""

import torch


def validate_cpu_int_mirror(starts, count, *, label):
    """Validate without reading values, including when a caller has a cache."""
    if (type(count) is not int or count < 0 or not isinstance(starts, torch.Tensor)
            or starts.device.type != "cpu" or starts.dtype not in (torch.int32, torch.int64)
            or starts.ndim != 1 or starts.numel() < count):
        raise ValueError(f"{label} is missing or invalid")
    return starts


def cpu_int_mirror_values(starts, count, *, label):
    """The only online tensor-to-list conversion, for a validated CPU view."""
    starts = validate_cpu_int_mirror(starts, count, label=label)
    return tuple(starts[:count].tolist())
