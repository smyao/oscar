"""Archive #148/#170/#173: frame regressions cannot hide in request averages.

Synthetic certificate metrics test acceptance/reporting only; actual replay's
counterexample is in test_optr_traces. No NPU or model-quality claims follow.
"""
import copy

import pytest

from oscar_ascend.optr_artifacts import (
    experimental_certificate, independent_test_summary, nonworse,
    validate_refinement_metadata,
)


def metric(errors=(4., 1.), *, split="validation"):
    frames = [
        {"frame_id": 0, "context": 0, "current_end": 4, "query_positions": [0, 3]},
        {"frame_id": 1, "context": 4, "current_end": 5, "query_positions": [4]},
    ]
    for frame, error in zip(frames, errors):
        frame.update(squared_error=error, teacher_energy=4., normalized_error=error / 4.)
        fraction = 1 / len(frame["query_positions"])
        frame["terminal_query"] = {"position": frame["query_positions"][-1],
            "squared_error": error * fraction, "teacher_energy": 4. * fraction,
            "normalized_error": error / 4.}
    total = sum(errors)
    row = {"request_id": split, "sample_sha256": ("a" if split == "validation" else "b") * 64,
           "squared_error": total, "teacher_energy": 8., "normalized_error": total / 8.,
           "frames": frames}
    return {key: row[key] for key in ("squared_error", "teacher_energy", "normalized_error")} | {"requests": [row]}


def test_lower_request_error_cannot_hide_worse_decode_frame():
    before, after = metric(), metric((1., 2.))
    assert after["squared_error"] < before["squared_error"]
    assert not nonworse(before, after)
    assert nonworse(before, metric((2., .5)), require_frames=True)
    assert nonworse(before, copy.deepcopy(before), require_frames=True)
    assert not nonworse(before, metric((1., 1.000000000000001)))  # No extra quality tolerance.


def test_frame_only_test_regression_remains_failed_evidence():
    records = {"layer": {"test": {"baseline": metric(split="test"),
                                    "selected": metric((1., 2.), split="test")}}}
    assert independent_test_summary(records) == {"status": "failed", "regressed_layers": ["layer"]}


@pytest.mark.parametrize("fault", ["empty", "not_list", "not_object", "frame_id", "context", "end",
                                   "positions", "future_position", "nan", "bool", "normalization", "totals"])
def test_invalid_frame_metrics_fail_closed(fault):
    value = metric()
    row = value["requests"][0]
    frame = row["frames"][0]
    if fault == "empty": row["frames"] = []
    elif fault == "not_list": row["frames"] = None
    elif fault == "not_object": row["frames"][0] = None
    elif fault == "frame_id": frame["frame_id"] = False
    elif fault == "context": row["frames"][1]["context"] = 3
    elif fault == "end": frame["current_end"] = 0
    elif fault == "positions": frame["query_positions"] = [3, 0]
    elif fault == "future_position": frame["query_positions"] = [0, 4]
    elif fault == "nan": frame["squared_error"] = float("nan")
    elif fault == "bool": frame["teacher_energy"] = True
    elif fault == "normalization": frame["normalized_error"] = 99.
    elif fault == "totals": frame.update(squared_error=2., normalized_error=.5)
    with pytest.raises(ValueError, match="frame"):
        nonworse(value, copy.deepcopy(value))


@pytest.mark.parametrize("fault", ["positions", "teacher", "missing"])
def test_frame_identity_must_match_even_when_request_identity_matches(fault):
    before, after = metric(), metric((2., .5))
    frames = after["requests"][0]["frames"]
    if fault == "positions": frames[0]["query_positions"] = [1, 3]
    elif fault == "teacher":
        for frame, energy in zip(frames, (3., 5.)):
            frame.update(teacher_energy=energy, normalized_error=frame["squared_error"] / energy)
            terminal = frame["terminal_query"]
            terminal.update(teacher_energy=energy / len(frame["query_positions"]),
                            normalized_error=frame["normalized_error"])
    else: after["requests"][0].pop("frames")
    with pytest.raises(ValueError, match="frame"):
        nonworse(before, after)


def test_old_request_only_metrics_are_readable_but_not_valid_new_frame_evidence():
    before, after = metric(), metric((2., .5))
    for value in (before, after): value["requests"][0].pop("frames")
    assert nonworse(before, after)
    with pytest.raises(ValueError, match="requires per-frame"):
        nonworse(before, after, require_frames=True)


@pytest.mark.parametrize("split", ["validation", "test"])
@pytest.mark.parametrize("choice", ["baseline", "selected"])
def test_new_certificate_requires_frames_for_all_recorded_metrics(tmp_path, split, choice):
    from test_optr_integration import NAME, artifact, configuration
    value = artifact(configuration(tmp_path))
    value["refinement_certificate"]["layers"][NAME][split][choice]["requests"][0].pop("frames")
    with pytest.raises(ValueError, match="requires per-frame"):
        validate_refinement_metadata(value)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_historical_certificates_keep_request_only_interpretation(tmp_path, version):
    from test_optr_integration import NAME, artifact, configuration
    from oscar_ascend.optr_policy import legacy_capture_profile
    config = configuration(tmp_path)
    value = artifact(config)
    value["calibration_profile"] = legacy_capture_profile(value["calibration_profile"], config, version)
    certificate = value["refinement_certificate"]
    certificate["policy"] = value["calibration_profile"]["refinement_policy"]
    for split in ("validation", "test"):
        for choice in ("baseline", "selected"):
            certificate["layers"][NAME][split][choice]["requests"][0].pop("frames")
    assert validate_refinement_metadata(value) is certificate


def test_v3_historical_certificate_still_requires_frames_but_not_new_terminal_metrics(tmp_path):
    from test_optr_integration import NAME, artifact, configuration
    from oscar_ascend.optr_policy import legacy_capture_profile
    config = configuration(tmp_path)
    value = artifact(config)
    value["calibration_profile"] = legacy_capture_profile(value["calibration_profile"], config, "v3")
    certificate = value["refinement_certificate"]
    certificate["policy"] = value["calibration_profile"]["refinement_policy"]
    for split in ("validation", "test"):
        for choice in ("baseline", "selected"):
            for frame in certificate["layers"][NAME][split][choice]["requests"][0]["frames"]:
                frame.pop("terminal_query")
    assert validate_refinement_metadata(value) is certificate
    certificate["layers"][NAME]["test"]["selected"]["requests"][0].pop("frames")
    with pytest.raises(ValueError, match="requires per-frame"):
        validate_refinement_metadata(value)


def test_experimental_frame_regression_is_reported_without_relaxing_validation(tmp_path):
    from test_optr_integration import NAME, artifact, configuration
    value = artifact(configuration(tmp_path))
    certificate = value["refinement_certificate"]
    record = certificate["layers"][NAME]
    record.update(accepted=True, base_constants_sha256="e" * 64,
                  validation={"baseline": metric(), "selected": metric((2., .5))},
                  test={"baseline": metric(split="test"), "selected": metric((1., 2.), split="test")})
    certificate["accepted_layers"] = [NAME]
    with pytest.raises(ValueError, match="worsens test"):
        validate_refinement_metadata(value)
    value["refinement_certificate"] = experimental_certificate(certificate)
    checked = validate_refinement_metadata(value)
    assert checked["independent_test"] == {"status": "failed", "regressed_layers": [NAME]}
    record["validation"]["selected"] = metric((1., 2.))
    with pytest.raises(ValueError, match="worsens validation"):
        validate_refinement_metadata(value)


def test_terminal_prefill_regression_cannot_hide_in_lower_frame_error():
    before, after = metric(), metric((3., .5))
    terminal = after["requests"][0]["frames"][0]["terminal_query"]
    terminal.update(squared_error=2.5, normalized_error=1.25)
    assert after["squared_error"] < before["squared_error"]
    assert after["requests"][0]["frames"][0]["squared_error"] < 4.
    assert not nonworse(before, after, require_terminal=True)
    terminal.update(squared_error=2.000000000000001, normalized_error=1.0000000000000004)
    assert not nonworse(before, after, require_terminal=True)  # No added tolerance.
    terminal.update(squared_error=2., normalized_error=1.)
    assert nonworse(before, after, require_terminal=True)


@pytest.mark.parametrize("fault", ["missing", "none", "position", "bool", "nan",
                                   "normalization", "error_total", "energy_total", "single_query"])
def test_invalid_terminal_evidence_fails_closed(fault):
    value = metric()
    frame = value["requests"][0]["frames"][0]
    terminal = frame["terminal_query"]
    if fault == "missing": frame.pop("terminal_query")
    elif fault == "none": frame["terminal_query"] = None
    elif fault == "position": terminal["position"] = 2
    elif fault == "bool": terminal["position"] = True
    elif fault == "nan": terminal["squared_error"] = float("nan")
    elif fault == "normalization": terminal["normalized_error"] = 99.
    elif fault == "error_total": terminal.update(squared_error=5., normalized_error=2.5)
    elif fault == "energy_total": terminal.update(teacher_energy=5., normalized_error=.4)
    else:
        terminal = value["requests"][0]["frames"][1]["terminal_query"]
        terminal.update(squared_error=.5, normalized_error=.125)
    with pytest.raises(ValueError, match="terminal"):
        nonworse(value, copy.deepcopy(value), require_terminal=True)


def test_old_frame_evidence_is_optional_only_when_both_sides_agree():
    before, after = metric(), metric((2., .5))
    for value in (before, after):
        for frame in value["requests"][0]["frames"]: frame.pop("terminal_query")
    assert nonworse(before, after, require_frames=True)
    with pytest.raises(ValueError, match="requires terminal"):
        nonworse(before, after, require_terminal=True)
    with pytest.raises(ValueError, match="terminal query coverage"):
        nonworse(before, metric((2., .5)))
    for value in (before, after): value["requests"][0].pop("frames")
    with pytest.raises(ValueError, match="requires per-frame"):
        nonworse(before, after, require_terminal=True)


def test_terminal_teacher_identity_is_preserved_when_frame_teacher_sum_matches():
    before, after = metric(), metric((2., .5))
    terminal = after["requests"][0]["frames"][0]["terminal_query"]
    terminal.update(teacher_energy=1., normalized_error=1.)
    with pytest.raises(ValueError, match="terminal query position/teacher"):
        nonworse(before, after)


@pytest.mark.parametrize("split", ["validation", "test"])
@pytest.mark.parametrize("choice", ["baseline", "selected"])
@pytest.mark.parametrize("version", ["v4", "v5", "v6", "v7"])
def test_certificate_requires_terminal_query_for_every_frame(tmp_path, split, choice, version):
    from test_optr_integration import NAME, artifact, configuration
    from oscar_ascend.optr_policy import legacy_capture_profile
    config = configuration(tmp_path)
    value = artifact(config)
    if version != "v7":
        value["calibration_profile"] = legacy_capture_profile(value["calibration_profile"], config, version)
    policy = value["calibration_profile"]["refinement_policy"]
    value["refinement_certificate"]["policy"] = copy.deepcopy(policy)
    frame = value["refinement_certificate"]["layers"][NAME][split][choice]["requests"][0]["frames"][0]
    frame.pop("terminal_query", None)
    with pytest.raises(ValueError, match="requires terminal query"):
        validate_refinement_metadata(value)


def test_terminal_only_independent_test_regression_is_report_only(tmp_path):
    from test_optr_integration import NAME, artifact, configuration
    value = artifact(configuration(tmp_path))
    certificate = value["refinement_certificate"]
    record = certificate["layers"][NAME]
    before, after = metric(split="test"), metric((3., .5), split="test")
    after["requests"][0]["frames"][0]["terminal_query"].update(squared_error=2.5, normalized_error=1.25)
    record.update(accepted=True, base_constants_sha256="e" * 64,
                  validation={"baseline": metric(), "selected": metric((2., .5))},
                  test={"baseline": before, "selected": after})
    certificate["accepted_layers"] = [NAME]
    value["refinement_certificate"] = experimental_certificate(certificate)
    assert validate_refinement_metadata(value)["independent_test"] == {
        "status": "failed", "regressed_layers": [NAME]}
    record["validation"] = {"baseline": metric(), "selected": metric((3., .5))}
    record["validation"]["selected"]["requests"][0]["frames"][0]["terminal_query"].update(
        squared_error=2.5, normalized_error=1.25)
    with pytest.raises(ValueError, match="worsens validation"):
        validate_refinement_metadata(value)
