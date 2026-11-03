"""Archive #165/#176: restored proposal tensors must match imported metrics."""
import json

import pytest

from oscar_ascend import optr_checkpoint, optr_trust_region
from tools import prepare_optr_recipe3, prepare_optr_rotations
from test_optr_recipe3_pipeline import NAME, case, forbidden
from test_optr_trust_region_pipeline import (
    _forbid_new_fit_or_capture, _make_rejected_source, _prepare,
)


def test_republished_partial_proposals_cannot_borrow_other_matrices_saved_metrics(case, monkeypatch):
    config, output, _, _ = _make_rejected_source(case, monkeypatch)
    _forbid_new_fit_or_capture(monkeypatch)
    assert _prepare(config, output)["accepted_layers"] == [NAME]
    shared = next((output.parent / "fit-checkpoints").glob("*/layer-3-shared.json"))
    manifest = json.loads(shared.read_text())
    saved = optr_checkpoint.FitCheckpoints._load_from(shared, "layer-3-shared", manifest["identity"])
    seed = case["baseline"]["layers"][NAME]
    source_policy = saved["result"]["decision"]["policy"]
    assert len(saved["result"]["candidates"]) == 1
    # Replace both primary fields consistently by another valid orthogonal
    # pair. Keep the provenance/policy and rewrite the real atomic hash chain:
    # this is a semantic source mismatch, not corrupt bytes or bad matrices.
    for key in ("Rk", "Rv"):
        saved["result"][key] = seed[key].clone()
        saved["result"]["candidates"][0][key] = seed[key].clone()
    prepare_optr_recipe3._validate_result(saved["result"], seed, source_policy)
    saved.pop("recipe3_repair_lock")
    cache = optr_checkpoint.FitCheckpoints(shared.parent.parent, manifest["identity"])
    cache.save("layer-3-shared", saved)
    shared.with_name("layer-3.json").unlink()
    output.unlink()
    (output.parent / ".optr.complete.json").unlink()
    monkeypatch.setattr(optr_trust_region, "select_repair", forbidden)
    monkeypatch.setattr(prepare_optr_rotations, "_metrics", forbidden)
    with pytest.raises(ValueError, match="authenticated v10 source metrics"):
        _prepare(config, output)
    assert not output.exists()
