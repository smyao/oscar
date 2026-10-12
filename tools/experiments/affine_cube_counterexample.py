"""Reachable INT2 counterexample to affine FP16 Cube reassociation.

Archive #13-22/#126/#148/#154 and startup D.4: this is a CPU mathematical
falsification, never a production route or an NPU performance claim. The
independent PR-style quantizer produces every scale/zero/code used below.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from oscar_ascend.ops.reference import dequantize_int2, quantize_int2, unpack_int2
from tools.experiments.affine_cube_reference import split_half_matmul


def run() -> dict:
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    dim = 256
    generator = torch.Generator().manual_seed(1753)
    query = torch.randn((1, dim), generator=generator)
    query[0, :2] = 0.0
    codes = torch.ones((2, dim))
    codes[:, 0], codes[:, 1] = 0.0, 3.0
    key_scale = torch.tensor([[16.0], [1.0]])
    key_zero = -key_scale
    value_scale = torch.tensor([[2048.0], [2048.0]])
    value_zero = torch.tensor([[0.0], [-4096.0]])
    keys = codes * key_scale + key_zero
    values = codes * value_scale + value_zero
    encoded_k, encoded_v = quantize_int2(keys), quantize_int2(values)
    for encoded, original, scales, zeros in (
        (encoded_k, keys, key_scale, key_zero),
        (encoded_v, values, value_scale, value_zero),
    ):
        assert torch.equal(unpack_int2(encoded.packed, dim).float(), codes)
        assert torch.equal(encoded.scale.float(), scales)
        assert torch.equal(encoded.zero.float(), zeros)
        assert torch.equal(dequantize_int2(encoded), original)
    scale = dim ** -0.5
    reference_logits = (query @ keys.T) * scale
    reference = torch.softmax(reference_logits, dim=-1) @ values
    frozen = {"atol": 0.005, "rtol": 0.005}
    rows = []
    for components in (2, 3, 4):
        candidate_logits = (
            split_half_matmul(query, codes.T, components) * key_scale.T
            + query.sum(-1, keepdim=True) * key_zero.T
        ) * scale
        probability = torch.softmax(candidate_logits, dim=-1)
        candidate = (
            split_half_matmul(probability * value_scale.T, codes, components)
            + probability @ value_zero
        )
        difference = (candidate - reference).abs()
        allowed = frozen["atol"] + frozen["rtol"] * reference.abs()
        rows.append({
            "components": components,
            "logits": candidate_logits.flatten().tolist(),
            "max_abs": float(difference.max()),
            "failed_elements": int((difference > allowed).sum()),
        })
    return {
        "status": "affine_reassociation_rejected",
        "scope": "CPU mathematical counterexample; no production or NPU performance claim",
        "dimension": dim,
        "seed": 1753,
        "query_norm": float(query.norm()),
        "key_norms": keys.norm(dim=1).tolist(),
        "key_scale": encoded_k.scale.flatten().tolist(),
        "key_zero": encoded_k.zero.flatten().tolist(),
        "value_scale": encoded_v.scale.flatten().tolist(),
        "value_zero": encoded_v.zero.flatten().tolist(),
        "reference_logits": reference_logits.flatten().tolist(),
        "reference_value_at_dimension_2": float(reference[0, 2]),
        "frozen_tolerance": frozen,
        "results": rows,
        "all_components_violate_frozen_gate": all(row["failed_elements"] > 0 for row in rows),
    }


def main() -> None:
    report = run()
    if not report["all_components_violate_frozen_gate"]:
        raise AssertionError("the designed affine counterexample no longer violates the frozen gate")
    path = Path("reports/affine_cube_counterexample_20260930.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("AFFINE_CUBE_COUNTEREXAMPLE " + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
