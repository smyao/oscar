"""Archive #34/#37-49/#94/#125/#127/#128/#156/#157: bounded local replay.

Development-only native AST/CPU simulation, never an NPU fallback or deploy
preflight. Failures retain their exit code and stop subsequent rounds.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from .phase import atomic_json, run_phase
from .serving_variants import runtime_feature_summary, variant_config
from .native_contract_sources import NativeSources, NativeSourceError

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ["tests/test_native_source_contracts.py", "tests/test_runtime_native.py",
             "tests/test_runtime.py", "tests/test_decode_bundle_routing.py",
             "tests/test_serving_variants.py"]
REPLAYS = ["tests/test_mtp_context_limits.py", "tests/test_local_stateful_replay.py"]


def memory_ledger(config):
    """Analytic Qwen3.5-27B TP4/MTP3 worksheet, NOT measured peak HBM.

    Shapes are checked against native model/config methods by the contract
    suite. Fixed pool allocation uses the memory budget; rows below describe
    live block demand without prefix sharing/cached checkpoints/transients.
    """
    from oscar_ascend.layout import HybridPageLayout, SlotLayout, ceil_div
    from oscar_ascend.lifecycle import SnapshotLayout
    from oscar_ascend.runtime import WorkspaceGeometry
    slot = SlotLayout(256)
    layout = HybridPageLayout(slot, 30720, 393216, 768,
                              native_page_size_bytes=817152)
    snapshots = SnapshotLayout(layout, 64, 256, 3)
    workspace = WorkspaceGeometry(config["max_num_batched_tokens"], 6, 1, 256,
                                  cube_cores=20, history_cluster_size=16)
    bytes_per_pool_block = 17 * layout.page_size_bytes
    scenarios = []
    for requests, length in ((1, 20000), (1, 32768), (4, 20000), (32, 20000), (32, 50000)):
        state_blocks = requests * 3 * (1 + 3)
        native_blocks = 1 + state_blocks + requests * ceil_div(length + 3, 768)
        oscar_blocks = 1 + state_blocks + requests * ceil_div(length + 3, layout.block_size)
        pool_saved = (native_blocks - oscar_blocks) * bytes_per_pool_block
        scenarios.append({"requests": requests, "context_tokens": length,
            "native_live_pool_blocks": native_blocks, "oscar_live_pool_blocks": oscar_blocks,
            "native_live_pool_bytes": native_blocks * bytes_per_pool_block,
            "oscar_live_pool_bytes": oscar_blocks * bytes_per_pool_block,
            "pool_demand_saved_bytes": pool_saved,
            "pool_saved_minus_oscar_workspace_bytes": pool_saved - workspace.total_bytes})
    return {"scope": "analytic_live_block_demand_not_HBM_measurement",
        "assumptions": {"model": "Qwen3.5-27B", "tp": 4, "mtp_tokens": 3,
            "prefix_mode": "align", "shared_tensors": 17, "gdn_groups": 3,
            "no_prefix_sharing_in_scenarios": True, "native_workspace_not_subtracted": True,
            "cached_checkpoints_transients_graphs_weights_activations_not_counted": True},
        "full_token_bytes": {"native_bf16": slot.bf16_token_bytes, "oscar_int2": slot.token_bytes},
        "tokens_per_full_page": {"native": 768, "oscar": layout.block_size},
        "native_page_bytes": layout.page_size_bytes,
        "snapshot_bytes_inside_page_padding": snapshots.required_bytes,
        "page_padding_bytes": layout.native_padding_bytes,
        "oscar_workspace_bytes_per_rank": workspace.total_bytes,
        "scenarios": scenarios,
        "caveat": "Fixed-budget pools can allocate similar HBM; capacity gain is not allocation reduction. "
                  "Workspace can outweigh pool demand savings for short/low-concurrency workloads."}


def junit_counts(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    return {key: sum(int(suite.get(key, 0)) for suite in suites)
            for key in ("tests", "failures", "errors", "skipped")}


def run(output: Path, rounds: int):
    output.mkdir(parents=True, exist_ok=True)
    config = variant_config(json.loads((ROOT / "configs/target.json").read_text()), "candidate")
    sources = ["oscar_ascend/integration/metadata.py", "oscar_ascend/integration/impl.py",
               "tools/local_simulation.py", "tools/serving_variants.py", "tools/native_contract_sources.py",
               "tests/native_contract_manifest.json", *CONTRACTS, *REPLAYS]
    report = {"scope": "local_native_AST_and_CPU_only", "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(), "rounds_requested": rounds,
        "python": sys.version, "effective_candidate": runtime_feature_summary(config, "candidate"),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        "memory": memory_ledger(config), "phases": [],
        "not_verified": ["CANN_build", "NPU_completion", "ACL_graph_capture", "ACL_graph_replay",
                         "full_model_quality", "measured_HBM_savings", "latency_vs_native",
                         "archive_157_device_first_error"]}
    summary = output / "report.json"
    atomic_json(summary, report)
    try:
        report["native_sources"] = NativeSources(ROOT).verify()
    except (NativeSourceError, OSError, ValueError) as error:
        report.update(status="failed", exit_code=2, failure_stage="native_source_inputs",
                      error=str(error))
        atomic_json(summary, report)
        print(f"LOCAL_SIMULATION failed rc=2 {error} report={summary}", flush=True)
        return 2
    for source in report["native_sources"]:
        print(f'LOCAL_NATIVE_SOURCE mode={source["mode"]} path={source["path"]} '
              f'commit={source["recorded_commit"]} files={source["files_sha256_verified"]}', flush=True)
    atomic_json(summary, report)
    environment = dict(os.environ, OSCAR_TERMINAL_LOG_MODE="full")
    # Local simulation can never silently opt in to device tests inherited
    # from another shell. Genuine device probes have separate entrypoints.
    environment.pop("OSCAR_RUN_NPU_TESTS", None)
    phases = [("contracts", CONTRACTS)] + [(f"replay-{i + 1}", REPLAYS) for i in range(rounds)]
    for name, tests in phases:
        xml = output / f"{name}-junit.xml"
        xml.unlink(missing_ok=True)
        command = [sys.executable, "-m", "pytest", "-q", "--tb=short", f"--junitxml={xml}", *tests]
        result = run_phase(name, command, cwd=ROOT, log_dir=output,
                           timeout=300, env=environment)
        phase = asdict(result)
        code = result.returncode or (0 if result.cleanup_complete else 1)
        try:
            phase["junit"] = junit_counts(xml)
            if phase["junit"]["failures"] or phase["junit"]["errors"]:
                code = code or 1
            if (phase["junit"]["tests"] <= phase["junit"]["skipped"]
                    or (name.startswith("replay-") and phase["junit"]["skipped"])):
                phase["report_error"] = "no executed tests or skipped required CPU replay"
                code = code or 1
        except (OSError, ValueError, ET.ParseError) as error:
            phase["report_error"] = str(error)
            code = code or 1
        report["phases"].append(phase)
        if code:
            report.update(status="failed", exit_code=code)
            atomic_json(summary, report)
            print(f"LOCAL_SIMULATION failed phase={name} rc={code} report={summary}", flush=True)
            return code
        atomic_json(summary, report)
    report.update(status="passed_local_only", exit_code=0)
    atomic_json(summary, report)
    print(f"LOCAL_SIMULATION passed_local_only rounds={rounds} report={summary}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.output or ROOT / "logs" / f"local-simulation-{stamp}").resolve()
    return run(output, args.rounds)


if __name__ == "__main__":
    raise SystemExit(main())
