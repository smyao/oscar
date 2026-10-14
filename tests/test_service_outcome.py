"""Archive #159: the supplied fatal service log ends with process returncode 0."""
import json
import sys

import pytest

from tools import install_serve, phase
from tools.service_outcome import fatal_service_evidence


@pytest.mark.parametrize("message", [
    "(Worker_TP1 pid=186817) ERROR 10-14 06:16:30 [multiproc_executor.py:987] WorkerProc hit an exception.",
    "(EngineCore pid=186769) ERROR 10-14 06:16:47 [core.py:1197] EngineCore encountered a fatal error.",
    "(APIServer pid=186739) ERROR 10-14 06:16:47 [serving.py:932] vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.",
    "(Worker_TP0 pid=186797) Exception in thread WorkerAsyncOutputCopy:",
])
def test_real_fatal_signatures_are_detected_with_provenance(tmp_path, message):
    log = tmp_path / "server.log"
    log.write_text("normal line\n" + message + "\n")
    result = fatal_service_evidence(log)
    assert result["fatal_line_count"] == 1
    assert result["excerpts"] == [{"line": 2, "text": message}]


def test_recovered_aot_warnings_and_request_errors_are_not_service_fatal(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("""(Worker_TP0 pid=1) WARNING 10-14 06:07:49 [decorators.py:321] Compiling model again due to a load failure
(Worker_TP0 pid=1) WARNING 10-14 06:07:49 [decorators.py:321] Original traceback:
(APIServer pid=2) ERROR 10-14 06:08:00 [serving.py:932] ValueError: invalid max_tokens
(APIServer pid=2) INFO 10-14 06:08:01 [test.py:1] description: EngineCore encountered a fatal error.
""")
    assert fatal_service_evidence(log)["fatal_line_count"] == 0


def test_long_lines_and_repeated_fatal_messages_have_bounded_evidence(tmp_path):
    log = tmp_path / "server.log"
    with log.open("w") as stream:
        stream.write("x" * 140000 + "\n")
        for pid in range(100):
            stream.write(f"(EngineCore pid={pid}) ERROR 10-14 06:16:47 [core.py:1197] EngineCore encountered a fatal error.\n")
    result = fatal_service_evidence(log)
    assert result["fatal_line_count"] == 100
    assert len(result["excerpts"]) == 8 and result["excerpts"][0]["line"] == 2


def test_zero_exit_with_engine_death_is_failed_without_rewriting_child_code(monkeypatch, tmp_path, capsys):
    config = json.loads((install_serve.ROOT / "configs/target.json").read_text())
    config["shutdown_timeout_seconds"] = .1
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    logs = tmp_path / "run"
    def runner(name, command, **kwargs):
        if name != "server":
            return phase.PhaseResult(name, command, 0, 0., False, False, str(logs / name), True)
        message = "(EngineCore pid=186769) ERROR 10-14 06:16:47 [core.py:1197] EngineCore encountered a fatal error."
        child = "print(" + repr(message) + "); raise SystemExit(0)"
        return phase.run_phase(name, [sys.executable, "-c", child], **kwargs)
    monkeypatch.setattr(install_serve, "run_phase", runner)
    assert install_serve.main(["--config", str(config_path), "--log-dir", str(logs), "--variant", "candidate"]) == 1
    output = capsys.readouterr()
    assert "PASSED phase=server" not in output.out
    assert "service status=failed child_rc=0 exit_code=1" in output.err
    status = json.loads((logs / "status.json").read_text())
    assert status["status"] == "failed" and status["returncode"] == 1 and status["child_returncode"] == 0
    assert status["server"]["returncode"] == 0 and status["fatal_log_evidence"]["fatal_line_count"] == 1
    result = json.loads((logs / "server.json").read_text())
    assert result["returncode"] == 0 and result["service_returncode"] == 1
