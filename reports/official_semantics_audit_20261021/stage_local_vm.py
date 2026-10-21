"""Archive #79-85/#117: stage this checkout into a fresh local Lima directory.

No target-machine access. Reuse only the existing VM Python executable, not
another workspace's source. Stage source files explicitly and retain logs.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
GUEST = "/tmp/oscar-official-math-20261021"


def main():
    paths = list((ROOT / "csrc").glob("*.cpp")) + [ROOT / "csrc/CMakeLists.txt", ROOT / "pyproject.toml"]
    for name in ("csrc/kernels", "csrc/include", "csrc/experiments/current_only", "csrc/experiments/official_math_20261021", "oscar_ascend", "tools", "configs"):
        paths += [p for p in (ROOT / name).rglob("*") if p.is_file() and not p.is_symlink()
                  and "__pycache__" not in p.parts and not p.name.startswith("._")]
    paths = sorted(set(paths))
    checks = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    out = ROOT / "reports/official_semantics_audit_20261021/source_stage.json"
    out.write_text(json.dumps({"guest": GUEST, "sha256": checks}, indent=2) + "\n")
    subprocess.run(["limactl", "shell", "--workdir", "/tmp", "oscar", "--", "mkdir", "-p", GUEST], check=True, timeout=30)
    with tempfile.TemporaryFile() as archive:
        with tarfile.open(fileobj=archive, mode="w") as bundle:
            for p in paths:
                bundle.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
        archive.seek(0)
        subprocess.run(["limactl", "shell", "--workdir", GUEST, "oscar", "--", "tar", "-xf", "-"],
                       stdin=archive, check=True, timeout=60)
    print(json.dumps({"guest": GUEST, "files": len(paths), "manifest": str(out)}))


if __name__ == "__main__":
    main()
