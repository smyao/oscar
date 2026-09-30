"""Archive #148-151: the user's fixed serving command selects the probed bundle."""
import json

import pytest

from tools import install_serve, observe_serve
from tools.serving_variants import variant_config, variant_features


@pytest.mark.parametrize("variant,enabled", [("candidate", True), ("baseline", False)])
def test_install_and_probe_plans_select_the_same_bundle(variant, enabled, capsys):
    assert install_serve.main(["--variant", variant, "--plan"]) == 0
    install = json.loads(capsys.readouterr().out)
    assert observe_serve.main(["--variant", variant, "--plan"]) == 0
    probe = json.loads(capsys.readouterr().out)
    assert install["optimizations"] == probe["optimizations"] == {
        "history_cluster4": enabled, "later_mtp_q1": enabled,
        "fast_unpack": enabled, "mixed_cv": enabled,
        "striped_cache": enabled}
    assert install["probes"] == "none"


def test_default_retains_config_and_candidate_does_not_mutate_input():
    original = {"devices": [0, 1, 2, 3], "port": 8989}
    assert variant_config(original, None) == original
    selected = variant_config(original, "candidate")
    assert all(variant_features(selected).values())
    assert original == {"devices": [0, 1, 2, 3], "port": 8989}
    assert not any(variant_features(variant_config(selected, "native")).values())
    with pytest.raises(ValueError, match="explicit boolean"):
        variant_config({"experimental_fast_unpack": "true"}, "candidate")
    with pytest.raises(ValueError, match="requires"):
        variant_config({"experimental_fast_unpack": True}, None)
    with pytest.raises(ValueError, match="explicit boolean"):
        variant_config({"experimental_mixed_cv": "true"}, "candidate")
    with pytest.raises(ValueError, match="requires"):
        variant_config({"experimental_mixed_cv": True,
                        "experimental_history_reuse": True}, None)
    with pytest.raises(ValueError, match="explicit boolean"):
        variant_config({"experimental_striped_cache": "true"}, "candidate")
    with pytest.raises(ValueError, match="striped cache requires"):
        variant_config({"experimental_striped_cache": True,
                        "experimental_fast_unpack": True,
                        "experimental_history_reuse": True}, None)
