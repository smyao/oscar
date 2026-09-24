# 档案 #70-#73/#85/#94/#95/#125/#126/#129-#146：同配置配对测量；
# 当前签名构建、真 NPU 数值门、逐次释放与失败证据先于性能服务。
"""Verify current AscendC operators, then run one 20/23/27/30K K4 pair.

This one-command synthetic diagnostic builds on source drift, gates real-NPU
CV/rotation accuracy, and checks resource release. It is not the frozen
multi-case performance acceptance matrix or the user's private dataset.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from benchmarks.compare import canonical_sha256, read_json
from benchmarks.mixed import compare_mixed
from .environment import file_fingerprint
from .npu_resources import (DEFAULT_RELEASE_TOLERANCE, read_npu_resources,
                            wait_for_release)
from .phase import atomic_json, cleanup_group, live_log, run_phase, terminal_line
from .service_probe import SYNTHETIC_MIXED_LENGTHS
from .target_cli import ROOT, target_env


CV_NPU_MIN_CASES = 32  # #126/#144-146: include guard plus profile target cases.
ROTATION_NPU_MIN_CASES = 104  # 26 goldens on each selected card.


class OperatorGateError(RuntimeError):
    """A pre-service build or real-device gate failed with its actual phase rc."""

    def __init__(self, phase: str, message: str, *, returncode: int = 1,
                 evidence: dict | None = None):
        super().__init__(message)
        self.phase = phase
        self.returncode = returncode if returncode else 1
        self.evidence = evidence or {}


def _operator_gate_identity(manifest: dict, config: dict, acceptance: dict) -> dict:
    """Bind prior NPU evidence to the built bits, selected cards and oracle."""
    tracked = [ROOT / "tests/test_cv_contracts.py", ROOT / "tests/test_rotation_npu.py",
               ROOT / "tools/generate_rotation_cpu_cases.py"]
    if any(not path.is_file() for path in tracked):
        raise RuntimeError("real NPU accuracy test or golden generator is missing")
    oracle = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in tracked}
    runtime_source = file_fingerprint(ROOT / "oscar_ascend")
    if not runtime_source:
        raise RuntimeError("OSCAR runtime source fingerprint is empty")
    return {"build_signature": manifest["signature"],
            "source_sha256": canonical_sha256(manifest["configuration"]["source"]),
            "artifact_sha256": manifest["sha256"],
            "target_config_sha256": canonical_sha256(config),
            "acceptance_sha256": canonical_sha256(acceptance),
            "oracle_sha256": canonical_sha256(oracle),
            "runtime_sha256": canonical_sha256(runtime_source),
            "devices": config["devices"]}


def _real_npu_junit(path: Path, devices: list[int]) -> dict:
    """Pytest rc=0 alone can include skipped NPU cases; inspect the real gate."""
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    cv = [case for case in cases if case.get("name", "").startswith("test_npu_")]
    rotation = [case for case in cases
                if case.get("name", "").startswith("test_rotation_ascendc_real_npu")]
    selected = cv + rotation
    if len(cv) < CV_NPU_MIN_CASES or len(rotation) < ROTATION_NPU_MIN_CASES:
        raise RuntimeError(f"real NPU CV/rotation cases missing: cv={len(cv)} rotation={len(rotation)}")
    if any(case.find(kind) is not None for case in selected
           for kind in ("skipped", "failure", "error")):
        raise RuntimeError("real NPU CV/rotation case skipped or failed in JUnit")
    observed = {value for case in rotation for prop in case.findall("./properties/property")
                if prop.get("name") == "physical_device"
                for value in [prop.get("value")]}
    if observed != {str(device) for device in devices}:
        raise RuntimeError(f"rotation gate did not complete on all selected physical cards: {sorted(observed)}")
    return {"cv_cases": len(cv), "rotation_cases": len(rotation),
            "physical_devices": sorted(observed),
            "junit": str(path.resolve()), "junit_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _prior_npu_gate(path: Path, identity: dict, devices: list[int]) -> dict | None:
    """Reuse only an intact, exact prior device gate; never call it fresh proof."""
    try:
        previous = read_json(path)
        if previous.get("status") != "passed" or previous.get("identity") != identity:
            return None
        verified = _real_npu_junit(Path(previous["junit"]).resolve(), devices)
        if verified["junit_sha256"] != previous.get("junit_sha256"):
            return None
        if previous.get("resource_release") != "passed":
            return None
        return previous
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ET.ParseError, RuntimeError):
        return None


def _normalize_synthetic(report: dict, variant: str) -> dict:
    """Accept the stand-alone diagnostic or full-service performance evidence."""
    if report.get("mode") == "synthetic_mixed":
        if report.get("variant") != variant:
            raise ValueError(f"{variant} report has variant={report.get('variant')!r}")
        return report
    performance = report.get("performance")
    if not isinstance(performance, dict):
        raise ValueError(f"{variant} report has no synthetic_mixed or performance evidence")
    if report.get("server", {}).get("variant") != variant:
        raise ValueError(f"{variant} full-service report has wrong server variant")
    return {"mode": "synthetic_mixed", "variant": variant,
            "status": "measured" if report.get("status") == "passed" and
            performance.get("status") == "measured" else "failed",
            "synthetic_mixed": performance}


def compare_synthetic_reports(native: dict, oscar: dict, acceptance: dict) -> dict:
    """Apply frozen ratios to exact paired prompts; never claim full acceptance.

    Archive #73/#137-#139: a prior native log is not a matching baseline, and
    one K4 batch cannot satisfy the frozen warmup/repeat and coverage matrix.
    """
    left = _normalize_synthetic(native, "native")
    right = _normalize_synthetic(oscar, "oscar")
    warmup_pairing = ("fresh_service_before_batch" if native.get("mode") == "synthetic_mixed"
                      and oscar.get("mode") == "synthetic_mixed" else "unpaired")
    diagnostic = compare_mixed(left, right)
    policy = acceptance.get("performance")
    if not isinstance(policy, dict) or acceptance.get("frozen_before_measurement") is not True:
        raise ValueError("frozen performance acceptance policy is required")
    max_latency = policy.get("max_latency_ratio")
    min_throughput = policy.get("min_throughput_ratio")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
           for v in (max_latency, min_throughput)):
        raise ValueError("performance latency/throughput ratios must be finite and positive")
    issues = list(diagnostic["issues"])
    unresolved = False
    if diagnostic["status"] == "diagnostic_measured":
        native_section, oscar_section = left["synthetic_mixed"], right["synthetic_mixed"]
        for name, section in (("native", native_section), ("oscar", oscar_section)):
            identity = section.get("pair_identity")
            if not isinstance(identity, dict) or canonical_sha256(identity) != section.get("pair_sha256"):
                issues.append(f"{name} pair identity is missing or does not hash to pair_sha256")
        if native_section.get("pair_identity") != oscar_section.get("pair_identity"):
            issues.append("native and OSCAR pair identity fields differ")
        for name, side in (("native", left), ("oscar", right)):
            if side["synthetic_mixed"].get("prompt_lengths") != list(SYNTHETIC_MIXED_LENGTHS):
                issues.append(f"{name} prompt lengths differ from fixed 20/23/27/30K")
        if len(diagnostic["batches"]) != 1:
            issues.append("the diagnostic requires exactly one batch per variant")
        for batch in diagnostic["batches"]:
            if len(batch["requests"]) != len(SYNTHETIC_MIXED_LENGTHS):
                issues.append("paired batch lacks one of the four requests")
            for metric in ("prompt_throughput_ratio", "generation_throughput_ratio"):
                ratio = batch.get(metric)
                if type(ratio) not in (int, float) or not math.isfinite(ratio):
                    issues.append(f"{metric} ratio missing or nonfinite")
                elif ratio < min_throughput:
                    issues.append(f"{metric}={ratio:.4f} < {min_throughput:.4f}")
            for request in batch["requests"]:
                request_id = request["request_id"]
                source = {row["request_id"]: row for row in native_section["sample"]["requests"]}
                candidate = {row["request_id"]: row for row in oscar_section["sample"]["requests"]}
                n_salt = source[request_id].get("cache_salt_sha256")
                o_salt = candidate[request_id].get("cache_salt_sha256")
                if not isinstance(n_salt, str) or len(n_salt) != 64 or n_salt != o_salt:
                    issues.append(f"{request_id} cache salt identity missing or differs")
                for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
                    ratio = request["oscar_over_native"].get(metric)
                    if type(ratio) not in (int, float) or not math.isfinite(ratio):
                        if metric == "tpot_ms" and (source[request_id].get(metric) == 0
                                                    or candidate[request_id].get(metric) == 0):
                            issues.append(f"{request_id} tpot_unresolved_SSE_burst")
                            unresolved = True
                        else:
                            issues.append(f"{request_id} {metric} ratio missing or nonfinite")
                    elif ratio > max_latency:
                        issues.append(f"{request['request_id']} {metric}={ratio:.4f} > {max_latency:.4f}")
    status = ("passed" if not issues and diagnostic["status"] == "diagnostic_measured"
              else "needs_evidence" if unresolved and all("tpot_unresolved_SSE_burst" in item
                                                           for item in issues) else "failed")
    return {"status": status,
            "scope": "single-batch client-side synthetic K4 directional ratio screen",
            "warmup_pairing": warmup_pairing,
            "warmup_note": ("OSCAR synthetic follows functional probes while native synthetic follows startup"
                            if warmup_pairing == "unpaired" else
                            "both synthetic batches follow fresh service startup"),
            "performance_acceptance": "not_run",
            "max_latency_ratio": max_latency, "min_throughput_ratio": min_throughput,
            "issues": issues, "pair_sha256": diagnostic.get("pair_sha256"),
            "batches": diagnostic.get("batches", []),
            "diagnostic_status": diagnostic["status"]}


def _observe_resources(config: dict, directory: Path, *, before=None):
    """Retain observer logs without printing their device import chatter."""
    directory.mkdir(parents=True, exist_ok=True)
    with live_log(directory / "observer-console.log", mode="compact") as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            if before is None:
                return read_npu_resources(config, log_dir=directory,
                    timeout=float(config.get("resource_observation_timeout_seconds", 30)))
            return wait_for_release(config, before, log_dir=directory,
                timeout=float(config.get("resource_release_timeout_seconds", 30)),
                tolerance_bytes=config.get("resource_release_tolerance_bytes",
                                           DEFAULT_RELEASE_TOLERANCE))


def measure_cv_hotshape(variant: str, config_path: Path, config: dict,
                        acceptance_path: Path, log_dir: Path) -> dict:
    """#144: measure the old signed artifact BEFORE rebuild, then the candidate.

    The reference is an isolated operator benchmark pinned to this project's
    deployed source, never a production fallback or a replay of user data.
    """
    from .probe_cv_hotshape import baseline_availability
    manifest = ROOT / "build/ascendc/build_manifest.json"
    if variant == "baseline":
        availability = baseline_availability(manifest, config_path)
        if availability["status"] != "available":
            return availability
    phase_name = f"cv-hotshape-{variant}"
    output = log_dir / f"{phase_name}-report.json"
    env = target_env(config)
    env.update(OSCAR_TARGET_CONFIG=str(config_path), OSCAR_TERMINAL_LOG_MODE="compact")
    before = _observe_resources(config, log_dir / f"resources-before-{phase_name}")
    result = None
    try:
        result = run_phase(phase_name,
            [sys.executable, "-m", "tools.probe_cv_hotshape", "--variant", variant,
             "--target", str(config_path), "--acceptance", str(acceptance_path),
             "--manifest", str(manifest), "--output", str(output)],
            cwd=ROOT, log_dir=log_dir, env=env,
            timeout=min(300.0, float(config.get("phase_timeout_seconds", 1800))),
            grace=float(config.get("shutdown_timeout_seconds", 30)), heartbeat=60)
    finally:
        release = _observe_resources(config, log_dir / f"resources-after-{phase_name}", before=before)
        atomic_json(log_dir / f"{phase_name}-release.json", release)
    evidence = {"status": "failed", "report": str(output), "resource_release": release["status"],
                "returncode": result.returncode, "log": result.log}
    if result.returncode != 0 or not result.cleanup_complete or release["status"] != "passed":
        raise OperatorGateError(phase_name, f"{phase_name} failed; log={result.log}",
                                returncode=result.returncode, evidence=evidence)
    checked = read_json(output)
    cases = checked.get("cases")
    if (checked.get("status") != "passed" or not isinstance(cases, dict) or not cases
            or any(not isinstance(row.get("sample_oracle"), dict) or
                   row["sample_oracle"].get("status") != "passed" or
                   type(row.get("median_ms")) not in (int, float) or
                   not math.isfinite(row["median_ms"]) or row["median_ms"] <= 0
                   for row in cases.values())):
        raise OperatorGateError(phase_name, f"{phase_name} lacks completed numerical/timing evidence", evidence=evidence)
    if variant == "candidate":
        for name, row in cases.items():
            profile = row.get("profile")
            if (not isinstance(profile, dict) or profile.get("status") != "passed" or
                    profile.get("profiling_only") is not True or
                    profile.get("mode") != "instrumented_kernel_diagnostic" or
                    profile.get("normal_profile_frozen_close") is not True or
                    not isinstance(profile.get("normal_profile_max_abs"), dict) or
                    any(type(profile["normal_profile_max_abs"].get(field)) not in (int, float) or
                        not math.isfinite(profile["normal_profile_max_abs"][field])
                        for field in ("partial", "lse")) or
                    not isinstance(profile.get("sample_oracle"), dict) or
                    profile["sample_oracle"].get("status") != "passed" or
                    not isinstance(profile.get("raw_shape"), list) or
                    len(profile["raw_shape"]) != 4 or
                    profile["raw_shape"][1:] != [3, 4, 20] or
                    type(profile.get("outer_event_ms")) not in (int, float) or
                    not math.isfinite(profile["outer_event_ms"]) or profile["outer_event_ms"] <= 0 or
                    type(profile.get("outer_event_over_normal_median")) not in (int, float) or
                    not math.isfinite(profile["outer_event_over_normal_median"]) or
                    profile["outer_event_over_normal_median"] <= 0 or
                    not isinstance(profile.get("raw_counters"), dict) or
                    not isinstance(profile["raw_counters"].get("sources"), dict) or
                    not {"history", "window", "current"}.issubset(profile["raw_counters"]["sources"])):
                raise OperatorGateError(phase_name,
                    f"{phase_name}/{name} lacks completed diagnostic kernel/accuracy evidence", evidence=evidence)
    evidence.update(status="passed", measurement=checked)
    return evidence


def compare_cv_hotshapes(baseline: dict, candidate: dict, acceptance: dict) -> dict:
    """Same-input operator A/B only; end-to-end native pairing remains required."""
    if baseline.get("status") != "passed":
        return {"status": "not_comparable", "reason": baseline.get("reason", "baseline artifact unavailable"),
                "candidate": candidate, "performance_acceptance": "not_established"}
    before, after = baseline["measurement"]["cases"], candidate["measurement"]["cases"]
    issues, rows = [], {}
    if set(before) != set(after):
        issues.append("operator cases differ")
    for name in sorted(set(before) & set(after)):
        left, right = before[name], after[name]
        if not left.get("fixture_sha256") or left.get("fixture_sha256") != right.get("fixture_sha256"):
            issues.append(f"{name}: operator input fingerprints differ")
        ratio = right["median_ms"] / left["median_ms"]
        rows[name] = {"baseline_ms": left["median_ms"], "candidate_ms": right["median_ms"],
                      "candidate_over_baseline": ratio}
        if ratio > acceptance["performance"]["max_latency_ratio"]:
            issues.append(f"{name}: candidate operator regressed ({ratio:.4f})")
    return {"status": "failed" if issues else "passed", "issues": issues, "cases": rows,
            "scope": "synthetic signed CV operator A/B; not native model speed acceptance"}


def cv_profile_terminal_rows(name: str, case: dict) -> list[str]:
    """Three source lines, with raw critical-core ticks kept apart from ms."""
    profile = case["profile"]
    sources = profile["raw_counters"]["sources"]
    lines = []
    for source_name in ("history", "window", "current"):
        engines = sources[source_name]
        aic = engines["aic"]
        aiv = (engines["aiv0"], engines["aiv1"])
        max_field = lambda field: max(engine[field]["max"] for engine in aiv)
        lines.append(
            f"[oscar] PERF_CV_PROFILE case={name} source={source_name} physical=0 "
            f"tasks_aic={aic['tasks']['sum_across_cores']} "
            f"kv_tiles_aic={aic['kv_tiles']['sum_across_cores']} "
            f"empty_aic={aic['empty_tasks']['sum_across_cores']} "
            f"qk_max_raw_ticks={aic['aic_qk']['max']} "
            f"pv_max_raw_ticks={aic['aic_pv']['max']} "
            f"aiv_load_max_raw_ticks={max_field('aiv_load_publish')} "
            f"aiv_waitqk_max_raw_ticks={max_field('aiv_wait_qk')} "
            f"aiv_softmax_max_raw_ticks={max_field('aiv_softmax_total')} "
            f"mask_finite_subset_max_raw_ticks={max_field('aiv_mask_finite')} "
            f"v2_stats_subset_max_raw_ticks={max_field('aiv_v2')} "
            f"aiv_waitpv_max_raw_ticks={max_field('aiv_wait_pv')} "
            f"profile_event_ms={profile['outer_event_ms']:.3f} "
            f"over_normal={profile['outer_event_over_normal_median']:.2f} "
            "scope=instrumented_kernel_diagnostic")
    return lines


def ensure_native_current_attention(config_path: Path, config: dict,
                                    acceptance_path: Path, log_dir: Path) -> dict:
    """Test native current output/LSE and the three-source merge on real NPU.

    Archive #55–69/#126: an available FIA symbol or placeholder LSE is not
    sufficient. This runs before either model service and never uses CPU as
    the operator under test.
    """
    output = log_dir / "native-current-fia-report.json"
    output.unlink(missing_ok=True)
    env = target_env(config)
    env.update(OSCAR_TARGET_CONFIG=str(config_path), OSCAR_TERMINAL_LOG_MODE="compact")
    before = _observe_resources(config, log_dir / "resources-before-current")
    result = None
    phase_error = None
    try:
        result = run_phase("native-current-fia", [sys.executable, "-m", "tools.probe_native_current_fia",
            "--target", str(config_path), "--acceptance", str(acceptance_path), "--output", str(output)],
            cwd=ROOT, log_dir=log_dir, env=env,
            timeout=min(300.0, float(config.get("phase_timeout_seconds", 1800))),
            grace=float(config.get("shutdown_timeout_seconds", 30)), heartbeat=60)
    except Exception as error:
        phase_error = error
    finally:
        try:
            release = _observe_resources(config, log_dir / "resources-after-current", before=before)
        except Exception as error:
            release = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
        atomic_json(log_dir / "native-current-release.json", release)
    code = result.returncode if result is not None else 1
    log = result.log if result is not None else str(log_dir / "native-current-fia.log")
    evidence = {"status": "failed", "report": str(output), "resource_release": release["status"],
                "returncode": code, "log": log}
    if (phase_error is not None or result is None or code != 0
            or not result.cleanup_complete or release["status"] != "passed"):
        raise OperatorGateError("native-current-fia", f"native-current NPU oracle/release failed; log={log}",
                                returncode=code, evidence=evidence) from phase_error
    try:
        checked = read_json(output)
        merged = checked["mixed_history_window_current_merge"]
        task_contract = checked["production_task_contract"]
        if (checked.get("status") != "current_partial_probe_passed" or merged.get("oracle") != "passed"
                or merged.get("history_window") != "production_NPU_attention_cv_out"
                or merged.get("prepare") != "production_NPU_prepare_attention_tasks_out"
                or merged.get("source2") != "production_NPU_suppress_current_source_tasks"
                or checked.get("long_case", {}).get("sampled_oracle") != "passed"
                or task_contract.get("source2_range_rewrite") != "passed"
                or task_contract.get("slot_guard") != "passed"
                or task_contract.get("metadata_error_preserved") is not True
                or task_contract.get("padding_excluded") is not True
                or not checked.get("small_cases")
                or any(case.get("oracle") != "passed" for case in checked["small_cases"])):
            raise ValueError("native-current output/LSE/three-source merge proof is incomplete")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise OperatorGateError("native-current-fia", f"native-current NPU report invalid: {error}; report={output}",
                                evidence=evidence) from error
    evidence.update(status="passed", device_completion="fresh_native_current_and_merge",
                    device_event_median_ms=checked["long_case"].get("device_event_median_ms"))
    terminal_line(f"[oscar] PERF_CURRENT_FIA_GATE accuracy=passed mixed_sources=passed task_contract=passed "
                  f"current16k_device_ms={evidence['device_event_median_ms']} release=passed")
    return evidence


def ensure_current_operators(config_path: Path, config: dict, acceptance: dict,
                             log_dir: Path, *, require_fresh_npu: bool = False) -> dict:
    """Verify the current signed build and true NPU oracle before either server."""
    from .build_ops import reusable_build

    evidence = {"status": "running", "build": "not_run", "accuracy": "not_run",
                "resource_release": "not_run", "devices": config["devices"]}
    evidence_path = log_dir / "operator-gate.json"
    atomic_json(evidence_path, evidence)
    env = target_env(config)
    env.update(OSCAR_TARGET_CONFIG=str(config_path.resolve()), OSCAR_RUN_NPU_TESTS="1",
               OSCAR_TERMINAL_LOG_MODE="compact")
    build_report = ROOT / "reports/build.json"
    build_report.unlink(missing_ok=True)  # #94/#95: no previous success can satisfy this run.
    try:
        build_result = run_phase("operator-build",
            [sys.executable, "-m", "tools.build_ops", "--soc", config["soc_version"],
             "--log-dir", str(log_dir / "build")], cwd=ROOT, log_dir=log_dir,
            timeout=float(config.get("phase_timeout_seconds", 1800)), env=env,
            grace=float(config.get("shutdown_timeout_seconds", 30)), heartbeat=60)
    except Exception as error:
        evidence.update(status="failed", build="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(evidence_path, evidence)
        raise OperatorGateError("operator-build", f"AscendC build runner failed: {error}; "
                                f"log={log_dir / 'operator-build.log'}", evidence=evidence) from error
    evidence["build_phase"] = {"returncode": build_result.returncode, "log": build_result.log,
                               "cleanup_complete": build_result.cleanup_complete}
    if build_result.returncode != 0 or not build_result.cleanup_complete:
        evidence.update(status="failed", build="failed")
        atomic_json(evidence_path, evidence)
        raise OperatorGateError("operator-build", f"current AscendC build failed; log={build_result.log}",
                                returncode=build_result.returncode, evidence=evidence)
    try:
        manifest = read_json(build_report)
        signature = manifest["signature"]
        verified = reusable_build(ROOT / "build/ascendc", signature)
        if (manifest.get("build") != "passed" or type(manifest.get("reused")) is not bool
                or verified is None or verified.get("sha256") != manifest.get("sha256")
                or verified.get("configuration") != manifest.get("configuration")
                or manifest.get("configuration", {}).get("source") != file_fingerprint(ROOT / "csrc")):
            raise RuntimeError("completed AscendC artifact or current-source signature is missing/mismatched")
        if manifest["configuration"].get("soc") != config["soc_version"]:
            raise RuntimeError("AscendC SOC signature differs from target.json")
        identity = _operator_gate_identity(manifest, config, acceptance)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as error:
        evidence.update(status="failed", build="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(evidence_path, evidence)
        raise OperatorGateError("operator-build", f"build evidence invalid: {error}; log={build_result.log}",
                                evidence=evidence) from error
    evidence.update(build="reused" if manifest["reused"] else "rebuilt",
                    build_signature=signature, source_sha256=identity["source_sha256"],
                    identity=identity)
    atomic_json(evidence_path, evidence)

    gate_path = ROOT / "build/ascendc/oscar_fast_probe_cv_gate.json"
    previous = (_prior_npu_gate(gate_path, identity, config["devices"])
                if manifest["reused"] and not require_fresh_npu else None)
    if previous is not None:
        evidence.update(status="passed", accuracy="reused_prior_evidence",
                        resource_release="reused_prior_evidence", accuracy_log=previous["log"],
                        junit=previous["junit"], cv_cases=previous["cv_cases"],
                        rotation_cases=previous["rotation_cases"])
        atomic_json(evidence_path, evidence)
        terminal_line(f"[oscar] PERF_OPERATOR_GATE build=reused npu_cv=reused_prior_evidence "
                      f"signature={signature[:12]} devices={','.join(map(str, config['devices']))}")
        return evidence

    # #125/#126 and H15: completed numerical assertions do not excuse an NPU
    # context leak that could contaminate the following native baseline.
    try:
        before = _observe_resources(config, log_dir / "resources-before-cv")
    except Exception as error:
        evidence.update(status="failed", accuracy="not_run",
                        error=f"pre-probe NPU resource snapshot failed: {type(error).__name__}: {error}")
        atomic_json(evidence_path, evidence)
        raise OperatorGateError("operator-cv-npu-resources", f"pre-probe NPU resource snapshot failed: "
                                f"{error}; log={log_dir / 'resources-before-cv'}",
                                evidence=evidence) from error
    atomic_json(log_dir / "resources-before-cv.json", before)
    junit = log_dir / "cv-npu.xml"
    junit.unlink(missing_ok=True)
    gate_result = None
    gate_error = None
    try:
        gate_result = run_phase("operator-cv-npu",
            [sys.executable, "-m", "pytest", "-q", "--maxfail=1",
             str(ROOT / "tests/test_cv_contracts.py"), str(ROOT / "tests/test_rotation_npu.py"),
             "--junitxml=" + str(junit)], cwd=ROOT, log_dir=log_dir,
            timeout=float(config.get("phase_timeout_seconds", 1800)), env=env,
            grace=float(config.get("shutdown_timeout_seconds", 30)), heartbeat=60)
    except Exception as error:
        gate_error = error
    finally:
        try:
            release = _observe_resources(config, log_dir / "resources-after-cv", before=before)
        except Exception as error:
            release = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
        atomic_json(log_dir / "resources-after-cv.json", release)
        evidence["resource_release"] = release["status"]
        evidence["resource_release_evidence"] = str(log_dir / "resources-after-cv.json")
    if gate_result is not None:
        evidence["accuracy_phase"] = {"returncode": gate_result.returncode, "log": gate_result.log,
                                      "cleanup_complete": gate_result.cleanup_complete}
    if gate_error is not None or gate_result is None or gate_result.returncode != 0 or not gate_result.cleanup_complete:
        rc = gate_result.returncode if gate_result is not None else 1
        log = gate_result.log if gate_result is not None else str(log_dir / "operator-cv-npu.log")
        evidence.update(status="failed", accuracy="failed")
        atomic_json(evidence_path, evidence)
        raise OperatorGateError("operator-cv-npu", f"real NPU CV/rotation gate failed; log={log}; "
                                f"resource_release={release['status']}", returncode=rc,
                                evidence=evidence) from gate_error
    try:
        case_counts = _real_npu_junit(junit, config["devices"])
    except (OSError, ValueError, ET.ParseError, RuntimeError) as error:
        evidence.update(status="failed", accuracy="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(evidence_path, evidence)
        raise OperatorGateError("operator-cv-npu", f"real NPU JUnit evidence invalid: {error}; "
                                f"log={gate_result.log}", evidence=evidence) from error
    if release["status"] != "passed":
        evidence.update(status="failed", accuracy="passed", error="NPU memory not released after accuracy gate")
        atomic_json(evidence_path, evidence)
        raise OperatorGateError("operator-cv-npu-release", f"NPU memory did not return after CV gate; "
                                f"evidence={log_dir / 'resources-after-cv.json'}", evidence=evidence)
    gate = {"status": "passed", "identity": identity, **case_counts,
            "resource_release": "passed", "resource_release_evidence": evidence["resource_release_evidence"],
            "log": gate_result.log, "returncode": gate_result.returncode,
            "scope": "real NPU CV and rotation device completion, not graph or service acceptance"}
    atomic_json(gate_path, gate)
    evidence.update(status="passed", accuracy="fresh_device_completion", accuracy_log=gate_result.log,
                    junit=str(junit), cv_cases=case_counts["cv_cases"],
                    rotation_cases=case_counts["rotation_cases"])
    atomic_json(evidence_path, evidence)
    terminal_line(f"[oscar] PERF_OPERATOR_GATE build={evidence['build']} npu_cv=fresh_device_completion "
                  f"signature={signature[:12]} devices={','.join(map(str, config['devices']))}")
    return evidence


def _run_variant(variant: str, config_path: Path, config: dict, directory: Path,
                 acceptance_path: Path = ROOT / "configs/acceptance.json") -> dict:
    """Capture full service output and always reclaim the runner's process group."""
    directory.mkdir(parents=True, exist_ok=True)
    report_path, console_path = directory / "report.json", directory / "console.log"
    command = [sys.executable, "-m", "tools.service_probe", "--synthetic",
               "--config", str(config_path), "--log-dir", str(directory),
               "--output", str(report_path)]
    if variant == "native":
        command.append("--native")
    else:
        command += ["--native-report", str(directory.parent / "native" / "report.json"),
                    "--acceptance", str(acceptance_path)]
    env = target_env(config)
    env["OSCAR_TERMINAL_LOG_MODE"] = "compact"
    duration = float(config.get("synthetic_timeout_seconds", 1800)) + float(
        config.get("shutdown_timeout_seconds", 30)) + 90
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("paired probe timeout must be finite and positive")
    started = time.monotonic()
    # Archive #125: full child log is durable, but traceback/error lines are
    # forwarded while the request is still running rather than after timeout.
    with live_log(console_path, mode="compact") as stream:
        stream.write("START " + json.dumps({"variant": variant, "command": command}) + "\n")
        stream.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        timed_out, returncode = False, 1
        try:
            try:
                returncode = process.wait(timeout=duration)
            except subprocess.TimeoutExpired:
                timed_out, returncode = True, 124
                stream.write(f"TIMEOUT variant={variant} limit={duration}s\n")
                stream.flush()
        finally:
            cleaned = cleanup_group(process, float(config.get("shutdown_timeout_seconds", 30)))
        if not cleaned and returncode == 0:
            returncode = 125
        stream.write("RESULT " + json.dumps({"returncode": returncode,
            "timed_out": timed_out, "runner_cleanup_complete": cleaned}) + "\n")
    try:
        probe = read_json(report_path)
    except (OSError, ValueError) as error:
        probe = {"status": "failed", "error": f"report missing or invalid: {error}"}
    lifecycle = probe.get("server") if isinstance(probe.get("server"), dict) else {}
    status = ("passed" if returncode == 0 and probe.get("status") == "measured"
              and lifecycle.get("cleanup_complete") is True and cleaned else "failed")
    return {"status": status, "variant": variant, "returncode": returncode,
            "timed_out": timed_out, "runner_cleanup_complete": cleaned,
            "owned_server_cleanup_complete": lifecycle.get("cleanup_complete") is True,
            "probe_status": probe.get("status"), "probe_error": probe.get("error"),
            "report": str(report_path), "log": str(console_path),
            "elapsed_seconds": round(time.monotonic() - started, 6)}


def run_paired(config_path: Path, *, output: Path, log_dir: Path,
               acceptance_path: Path, native_only: bool = False,
               require_fresh_npu: bool = False) -> dict:
    """Run native→OSCAR sequentially, with release proof between owners."""
    config_path, output = config_path.resolve(), output.resolve()
    acceptance_path, log_dir = acceptance_path.resolve(), log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "mode": "native_synthetic_mixed" if native_only else "paired_synthetic_mixed",
              "config": str(config_path), "acceptance": str(acceptance_path),
              "native": "not_run", "oscar": "not_run", "comparison": "not_run",
              "operator_gate": "not_run", "native_current_gate": "not_run",
              "cv_hotshape_baseline": "not_run", "cv_hotshape_candidate": "not_run",
              "performance_acceptance": "not_run", "exit_code": 1,
              "warmup_scope": "fresh_service_before_batch"}
    atomic_json(output, report)
    try:
        config, acceptance = read_json(config_path), read_json(acceptance_path)
        target_env(config)  # Archive #123: reject invalid selection before NPU touch.
        report["config_sha256"] = canonical_sha256(config)
        report["acceptance_sha256"] = canonical_sha256(acceptance)
        report["cv_hotshape_baseline"] = measure_cv_hotshape(
            "baseline", config_path, config, acceptance_path, log_dir)
        atomic_json(output, report)
        # #85/#125/#126/#140: the fast one-command path must not compare a
        # freshly pulled AscendC source tree against stale local .so files.
        report["operator_gate"] = ensure_current_operators(
            config_path, config, acceptance, log_dir,
            require_fresh_npu=require_fresh_npu)
        atomic_json(output, report)
        report["native_current_gate"] = ensure_native_current_attention(
            config_path, config, acceptance_path, log_dir)
        atomic_json(output, report)
        report["cv_hotshape_candidate"] = measure_cv_hotshape(
            "candidate", config_path, config, acceptance_path, log_dir)
        cv_comparison = compare_cv_hotshapes(report["cv_hotshape_baseline"], report["cv_hotshape_candidate"], acceptance)
        report["cv_hotshape_comparison"] = cv_comparison
        atomic_json(log_dir / "cv-hotshape-comparison.json", cv_comparison)
        for name, row in cv_comparison.get("cases", {}).items():
            terminal_line(f"[oscar] PERF_CV_AB case={name} baseline_ms={row['baseline_ms']:.3f} "
                          f"candidate_ms={row['candidate_ms']:.3f} ratio={row['candidate_over_baseline']:.3f} "
                          f"scope=synthetic_operator_only status={cv_comparison['status']}")
        if cv_comparison["status"] == "not_comparable":
            terminal_line(f"[oscar] PERF_CV_AB status=not_comparable reason={cv_comparison['reason']}")
            for name, row in report["cv_hotshape_candidate"]["measurement"]["cases"].items():
                terminal_line(f"[oscar] PERF_CV_OP case={name} candidate_ms={row['median_ms']:.3f} "
                              "sample_oracle=passed baseline=unavailable scope=synthetic_operator_only")
        for name, row in report["cv_hotshape_candidate"]["measurement"]["cases"].items():
            # The real measure_cv_hotshape gate requires this profile. Host
            # flow tests may replace that gate with a minimal fake report.
            if isinstance(row.get("profile"), dict):
                for line in cv_profile_terminal_rows(name, row):
                    terminal_line(line)
        if cv_comparison["status"] == "failed":
            raise OperatorGateError("cv-hotshape-comparison", "; ".join(cv_comparison["issues"]),
                                    evidence=cv_comparison)
        baseline = _observe_resources(config, log_dir / "resources-before-native")
        atomic_json(log_dir / "resources-before-native.json", baseline)
        for variant in (("native",) if native_only else ("native", "oscar")):
            terminal_line(f"[oscar] PERF_VARIANT start={variant} lengths=20K,23K,27K,30K K=4")
            result = _run_variant(variant, config_path, config, log_dir / variant, acceptance_path)
            report[variant] = result
            atomic_json(output, report)
            release = _observe_resources(config, log_dir / f"resources-after-{variant}",
                                         before=baseline)
            result["resource_release"] = release["status"]
            result["release_evidence"] = str(log_dir / f"resources-after-{variant}.json")
            atomic_json(log_dir / f"resources-after-{variant}.json", release)
            atomic_json(output, report)
            if result["status"] != "passed" or release["status"] != "passed":
                reason = result.get("probe_error") or release.get("reason") or "request, cleanup or release gate failed"
                report.update(status="failed", error=f"{variant}: {reason}",
                              exit_code=result["returncode"] if result["returncode"] != 0 else 1)
                return report
            terminal_line(f"[oscar] PERF_VARIANT done={variant} result=4/4 release=passed")
        if native_only:
            report.update(status="passed", exit_code=0)
            return report
        native, oscar = read_json(report["native"]["report"]), read_json(report["oscar"]["report"])
        comparison = compare_synthetic_reports(native, oscar, acceptance)
        report["comparison"] = comparison
        atomic_json(log_dir / "comparison.json", comparison)
        from .performance_summary import format_performance_summary
        for line in format_performance_summary(native, oscar, comparison,
                                               native_path=report["native"]["report"],
                                               oscar_path=report["oscar"]["report"],
                                               comparison_path=log_dir / "comparison.json"):
            terminal_line(line)
        report["status"] = "passed" if comparison["status"] == "passed" else "failed"
        report["exit_code"] = 0 if report["status"] == "passed" else 2
        if comparison["issues"]:
            report["error"] = "; ".join(comparison["issues"][:6])
    except OperatorGateError as error:
        gate_name = ("cv_hotshape_error" if error.phase.startswith("cv-hotshape") else
                     "native_current_gate" if error.phase == "native-current-fia" else "operator_gate")
        report[gate_name] = error.evidence
        report.update(status="failed", failed_phase=error.phase,
                      error=str(error), exit_code=error.returncode)
    except KeyboardInterrupt:
        report.update(status="interrupted", error="interrupted", exit_code=130)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", exit_code=1)
    finally:
        atomic_json(output, report)
        if report["status"] != "passed":
            terminal_line(f"[oscar] PERF_ERROR {report.get('error', 'probe failed')} report={output}", stderr=True)
        terminal_line(f"[oscar] PERF_RESULT status={report['status']} rc={report['exit_code']} report={output}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--acceptance", type=Path, default=ROOT / "configs/acceptance.json")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--native-only", action="store_true",
                        help="one-click pipeline: collect native side; compare with its existing OSCAR service probe")
    parser.add_argument("--require-fresh-npu", action="store_true",
                        help="full one-click flow: rerun real CV/rotation NPU gate even with prior matching evidence")
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    log_dir = args.log_dir or ROOT / "logs" / f"paired-concurrency-{stamp}"
    output = args.output or log_dir / "paired-report.json"
    return run_paired(args.config, output=output, log_dir=log_dir,
                      acceptance_path=args.acceptance, native_only=args.native_only,
                      require_fresh_npu=args.require_fresh_npu)["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
