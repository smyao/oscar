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
        "striped_cache": enabled, "current_only": enabled,
        "first_mtp_current_fia": enabled, "compact_later_mtp": enabled,
        "mixed_decode_split": enabled, "decode_bundle": enabled, "whole_prefill": enabled}
    assert install["probes"] == "none"


def test_default_retains_config_and_candidate_does_not_mutate_input():
    original = {"devices": [4, 5, 6, 7], "port": 7878,"max_num_batched_tokens":16384}
    assert variant_config(original, None) == original
    selected = variant_config(original, "candidate")
    features = variant_features(selected)
    assert all(features.values())
    assert original == {"devices": [4, 5, 6, 7], "port": 7878,"max_num_batched_tokens":16384}
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


@pytest.mark.parametrize("flag", ["experimental_current_only", "experimental_first_mtp_current_fia",
                                  "experimental_compact_later_mtp", "experimental_mixed_decode_split",
                                  "experimental_decode_bundle"])
def test_candidate_features_cannot_leak_to_native(flag):
    original = {flag: True,"max_num_batched_tokens":16384}
    if flag == "experimental_mixed_decode_split":
        original["experimental_decode_bundle"] = True
        original["experimental_first_mtp_current_fia"] = True
    assert variant_config(original, "candidate")[flag] is True
    for variant in ("baseline", "native"):
        assert variant_config(original, variant)[flag] is False
    assert original[flag] is True
    for variant in (None, "candidate", "baseline", "native"):
        with pytest.raises(ValueError, match="explicit boolean"):
            variant_config({flag: "true"}, variant)


def test_both_one_key_entries_enable_the_complete_candidate_without_manual_flags(tmp_path,capsys):
    from tools.serving_variants import WHOLE_PREFILL_SCHEDULER
    target=json.loads((install_serve.ROOT/'configs/target.json').read_text())
    path=tmp_path/'candidate.json';path.write_text(json.dumps(target))
    assert install_serve.main(['--config',str(path),'--variant','candidate','--plan'])==0
    direct=json.loads(capsys.readouterr().out)
    assert observe_serve.main(['--config',str(path),'--variant','candidate','--plan','--probe-only'])==0
    observed=json.loads(capsys.readouterr().out)
    assert direct['runtime_selection']==observed['runtime_selection']
    selected=direct['runtime_selection']
    assert all(selected['optimizations'].values())
    assert selected['max_num_batched_tokens']==32768
    assert selected['scheduler_cls']==WHOLE_PREFILL_SCHEDULER
    assert selected['devices']==[4,5,6,7] and selected['port']==7878
    assert direct['probes']=='none'
    assert {'decode_bundle_gate','current_only_gate','first_mtp_current_gate','mixed_decode_split_gate'}<=set(observed['phases'])
    assert observed['inference_requests_generated']==0
