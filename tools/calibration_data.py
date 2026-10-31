"""Archive #97/#165/#173/#175/#176: pinned source and disjoint OptR records.

The packaged gzip preserves the exact public simple-evals GPQA CSV bytes.
Hash validation happens before the native model is loaded. No TLS policy or
calibration prompt selection is changed to work around a network failure.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import zlib

GPQA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
GPQA_SHA256 = "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305"
BUNDLED_GPQA = Path(__file__).with_name("data") / "gpqa_diamond.csv.gz"
CALIBRATION_PARTITION = "canonical_json_sha256_sorted_round_robin_train_validation_test_v1"
CALIBRATION_SPLITS = ("train", "validation", "test")


def partition_calibration_messages(messages):
    """Share the existing OptR record split with its train-only initialization.

    Records are canonical UTF-8 JSON (sorted keys, no whitespace, unescaped
    Unicode), deduplicated by SHA256, sorted by digest and assigned round-robin
    to train/validation/test. Never partition already packed token prefixes:
    those would leak source records between initialization and heldout tests.
    """
    unique = {}
    for message in messages:
        encoded = json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        identity = hashlib.sha256(encoded).hexdigest()
        unique.setdefault(identity, message)
    if len(unique) < len(CALIBRATION_SPLITS):
        raise ValueError("OptR needs at least three distinct complete source records for disjoint splits")
    partitions = {split: [] for split in CALIBRATION_SPLITS}
    for index, item in enumerate(sorted(unique.items())):
        partitions[CALIBRATION_SPLITS[index % len(CALIBRATION_SPLITS)]].append(item)
    return partitions


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
