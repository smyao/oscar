"""Archive #94/#125/#148/#154: one reusable BF16 gate with real exit status.

This subprocess shares the observe entrypoint's source/artifact/configuration,
numerical, graph, performance and resource-release checks. It cannot serve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

from .phase import atomic_json, terminal_line
from .serving_variants import variant_config
from .target_cli import target_env


def run(config_path: Path, log_dir: Path) -> int:
    from .observe_serve import _bf16_history_gate
    log_dir = log_dir.resolve()
    status = {"status": "running", "phases": []}
    atomic_json(log_dir / "status.json", status)
    try:
        original = config_path.read_bytes()
        config = variant_config(json.loads(original), "candidate")
        # The deployment parent must bind every child to this same effective file.
        if config != json.loads(original):
            raise ValueError("BF16 gate requires the normalized effective target configuration")
        status["effective_config_sha256"] = hashlib.sha256(original).hexdigest()
        env = target_env(config)
        env["OSCAR_TARGET_CONFIG"] = str(config_path.resolve())
        _bf16_history_gate(config_path.resolve(), config, env, log_dir, status)
        status["status"] = "passed"
        return 0
    except Exception as error:
        code = getattr(error, "returncode", 1) or 1
        status.update(status="failed", failed_phase=getattr(error, "phase", "bf16-attention-gate"),
                      error=f"{type(error).__name__}: {error}", returncode=code)
        traceback.print_exc()
        terminal_line(f"[oscar] FAILED BF16 attention gate rc={code}: {error}", stderr=True)
        return code
    finally:
        atomic_json(log_dir / "status.json", status)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    return run(args.config, args.log_dir)


if __name__ == "__main__":
    sys.exit(main())
