"""Archive #169: wrapper failure propagation and evidence packaging, no NPU."""
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("rc", [0, 2, 124])
def test_probe_exit_and_bundle(tmp_path, rc):
    (tmp_path / "scripts").mkdir()
    script = tmp_path / "scripts/diagnose_bf16.sh"
    shutil.copyfile(ROOT / "scripts/diagnose_bf16.sh", script)
    run = tmp_path / "logs/20261020T064430.498396Z"
    run.mkdir(parents=True)
    (run / "effective-target.json").write_text('{"devices":[4,5,6,7]}')
    (run / "serve.log").write_text("original failure\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    git = bindir / "git"
    git.write_text("#!/bin/sh\nprintf 'test-source\\n'\n")
    git.chmod(0o755)
    python = bindir / "probe-python"
    python.write_text("""#!/bin/bash
set -eu
[[ "$*" == *"tools.phase"* && "$*" == *"tools.probe_bf16_history"* ]]
while [[ "$1" != "--log-dir" ]]; do shift; done
out="$2"
printf 'device probe stdout and stderr\\n' | tee "$out/bf16-probe.log"
printf '{"status":"fixture"}\\n' > "$out/result.json"
exit "$FAKE_PROBE_RC"
""")
    python.chmod(0o755)
    env = {**os.environ, "PATH": str(bindir) + os.pathsep + os.environ["PATH"],
           "OSCAR_PYTHON": str(python), "FAKE_PROBE_RC": str(rc)}
    result = subprocess.run(["bash", str(script)], cwd=tmp_path.parent,
                            env=env, text=True, capture_output=True)
    assert result.returncode == rc, result.stdout + result.stderr
    latest = tmp_path / "logs/bf16_diag_latest"
    assert latest.is_symlink()
    assert (latest / "exit-code.txt").read_text().strip() == str(rc)
    assert f"script_exit_code={rc}" in (latest / "summary.txt").read_text()
    archives = list((tmp_path / "logs").glob("*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0]) as archive:
        names = archive.getnames()
        assert "logs/20261020T064430.498396Z/serve.log" in names
        assert any(name.endswith("/bf16-probe.log") for name in names)
        assert any(name.endswith("/result.json") for name in names)
    assert "device probe stdout and stderr" in result.stdout
    assert (run / "serve.log").read_text() == "original failure\n"


def test_missing_run_is_explicit_failure(tmp_path):
    (tmp_path / "scripts").mkdir()
    script = tmp_path / "scripts/diagnose_bf16.sh"
    shutil.copyfile(ROOT / "scripts/diagnose_bf16.sh", script)
    result = subprocess.run(["bash", str(script)], text=True, capture_output=True)
    assert result.returncode == 1
    assert "缺少配置" in result.stdout
    assert not (tmp_path / "logs").exists()
