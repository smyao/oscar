"""Archive #148/#155: recover one complete step from pasted target fragments.

Keeps historical device-event costs separate from current source work counts.
Nested model envelopes are never added to their attention children. A single
old sample is a calibration prior, not an observed whole-run time fraction.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re


def complete_samples(path: Path):
    text = path.read_text()
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{\s*"accepted_scope"\s*:', text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue  # Only incomplete fragment objects are excluded.
        if isinstance(value, dict) and isinstance(value.get("scopes"), list):
            yield value


def ledger(path: Path, step_id="12263-1"):
    matches = [row for row in complete_samples(path) if row.get("step_id") == step_id]
    if len(matches) != 1:
        raise ValueError(f"expected one complete {step_id} sample, got {len(matches)}")
    sample = matches[0]
    groups = defaultdict(lambda: {"count": 0, "device_ms": 0.0, "tokens": []})
    envelopes = {}
    unclassified = []
    for scope in sample["scopes"]:
        duration = scope["duration_ms"]
        if not isinstance(duration, (float, int)) or not math.isfinite(duration) or duration < 0:
            raise ValueError("invalid device duration")
        if "phase" not in scope:
            unclassified.append(scope)
            continue
        phase = scope["phase"]
        if phase in ("target_forward", "draft_proposal"):
            if phase in envelopes:
                raise ValueError("ambiguous repeated model envelope")
            envelopes[phase] = duration
            continue
        fields = scope.get("fields", {})
        if "is_draft" not in fields:
            continue
        role = f"draft{fields['draft_index']}" if fields["is_draft"] else "main"
        group = groups[f"{role}/{phase}"]
        group["count"] += 1
        group["device_ms"] += duration
        group["tokens"].append(fields.get("tokens"))
    for role, count in (("main", 16), ("draft0", 1), ("draft1", 1), ("draft2", 1)):
        for phase in ("rotate", "phase1_stores"):
            if groups[f"{role}/{phase}"]["count"] != count:
                raise ValueError(f"incomplete {role}/{phase} evidence")
    later_store = sum(groups[f"draft{i}/phase1_stores"]["device_ms"] for i in (1, 2))
    return {
        "source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "step_id": step_id, "rank": sample["rank"], "scope": "one complete historical step JSON; only classified scopes attributed",
        "unclassified_scopes_excluded_from_attribution": unclassified,
        "step_device_ms": sample["step_device_ms"],
        "scheduled_prompt_tokens": sample["scheduled_prompt_tokens"],
        "model_envelopes_do_not_add_to_children_ms": envelopes,
        "leaf_phase_groups": dict(groups),
        "confirmed_source_mechanisms": {
            "padded_store": "Current striped RotateStore transforms/quantizes rows before checking negative slots",
            "dense_identity": "Runtime assigns MTP an identity matrix with hadamard=False; RotationEngine uses D dense dot products per row",
        },
        "historical_later_padding_store_ms": later_store,
        "budget_exclusions": [
            "Compact MTP and active-slot store overlap on the same padded-store work; subtract it only once",
            "First current-only already removes first-MTP Q rotation; identity-store cannot claim it again",
            "Identity-store still performs quantization, packing and exact window writes",
            "No whole-run fraction or current-device milliseconds inferred from this old single step",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = ledger(args.source)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"PREFILL_LEDGER step={result['step_id']} later_store_ms={result['historical_later_padding_store_ms']:.3f}")


if __name__ == "__main__":
    main()
