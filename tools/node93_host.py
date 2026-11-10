"""Archive #74/#75/#94/#125/#175/#176: explicit user-specified host preparation.

Only the --prepare-host deployment option invokes this helper. It does not
initialize SDKs, delete cache contents, or create a DirStore configuration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

HUGEPAGES_PATH = Path("/proc/sys/vm/nr_hugepages")
HUGEPAGES_VALUE = "200000\n"
HOST_COMMANDS = (
    ("mount", "-o", "remount,size=8G", "/dev/shm"),
    ("mkdir", "-p", "/home/kvcache"),
)


def prepare_host() -> None:
    print(f"[oscar] write {HUGEPAGES_PATH}=200000", flush=True)
    HUGEPAGES_PATH.write_text(HUGEPAGES_VALUE)
    for command in HOST_COMMANDS:
        print("[oscar] " + " ".join(command), flush=True)
        subprocess.run(command, check=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="show the three host operations without executing them")
    args = parser.parse_args(argv)
    if args.plan:
        print(json.dumps({"write": {"path": str(HUGEPAGES_PATH), "value": HUGEPAGES_VALUE},
                          "commands": HOST_COMMANDS, "executed": False}, indent=2))
        return 0
    try:
        prepare_host()
    except subprocess.CalledProcessError as exc:
        print(f"[oscar] FAILED node93 host preparation: command={exc.cmd!r} rc={exc.returncode}",
              file=sys.stderr, flush=True)
        return exc.returncode if exc.returncode >= 0 else 128 - exc.returncode
    except OSError as exc:
        print(f"[oscar] FAILED node93 host preparation: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
