"""Archive #148-151/#150-P0: fixed serving command selects the probed bundle."""
import json

import pytest

from tools import install_serve, observe_serve
from tools.serving_variants import variant_config, variant_features


def test_install_and_probe_plans_select_the_same_unified_path(capsys):
    assert install_serve.main(["--variant", "candidate", "--plan"]) == 0
    install = json.loads(capsys.readouterr().out)
    assert observe_serve.main(["--variant", "candidate", "--plan"]) == 0
    probe = json.loads(capsys.readouterr().out)
    assert install["optimizations"] == probe["optimizations"] == {
        "unified_int2_cv": True}
    assert install["probes"] == "none"


def test_default_retains_config_and_candidate_does_not_mutate_input():
    original = {"devices": [0, 1, 2, 3], "port": 8989}
    assert variant_config(original, None) == original
    selected = variant_config(original, "candidate")
    assert all(variant_features(selected).values())
    assert original == {"devices": [0, 1, 2, 3], "port": 8989}
    with pytest.raises(ValueError, match="unknown serving variant"):
        variant_config(selected, "baseline")
    with pytest.raises(ValueError, match="removed production switches"):
        variant_config({"experimental_fast_unpack": "true"}, "candidate")
    with pytest.raises(ValueError, match="removed production switches"):
        variant_config({"experimental_fast_unpack": True}, None)
    with pytest.raises(ValueError, match="removed production switches"):
        variant_config({"experimental_weighted_q4": True}, None)
    with pytest.raises(ValueError, match="removed production switches"):
        variant_config({"experimental_weighted_q4_split2": True}, None)
