"""Archive #97/#165: pinned official calibration input, with no startup download.

The packaged gzip preserves the exact public simple-evals GPQA CSV bytes.
Hash validation happens before the native model is loaded. No TLS policy or
calibration prompt selection is changed to work around a network failure.
"""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
import zlib

GPQA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
GPQA_SHA256 = "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305"
BUNDLED_GPQA = Path(__file__).with_name("data") / "gpqa_diamond.csv.gz"


def calibration_prompt_source(config, output_dir) -> Path:
    """Resolve explicit input or materialize the verified bundled CSV per capture.

    A fresh capture owns this directory. The global HF cache is deliberately
    not consulted: an old/truncated download cannot silently change calibration.
    """
    configured = config.get("rotation_calibration_prompts")
    if configured:
        source = Path(configured).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"calibration prompts file not found: {source}")
        return source
    try:
        data = gzip.decompress(BUNDLED_GPQA.read_bytes())
    except (OSError, EOFError, zlib.error) as error:
        raise ValueError(f"bundled GPQA data is missing or corrupt: {BUNDLED_GPQA}") from error
    actual = hashlib.sha256(data).hexdigest()
    if actual != GPQA_SHA256:
        raise ValueError(f"bundled GPQA SHA256 mismatch: expected {GPQA_SHA256}, got {actual}")
    source = Path(output_dir).resolve() / "gpqa_diamond.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    # No consumer runs until the write returns. Exclusive creation also prevents
    # a second capture from overwriting this capture's input provenance.
    with source.open("xb") as handle:
        handle.write(data)
    return source
