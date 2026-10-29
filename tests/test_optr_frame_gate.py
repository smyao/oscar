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
