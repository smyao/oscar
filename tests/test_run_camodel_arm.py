"""Archive #94/#95/#129: real exits/logs and a bounded owned child lifetime."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/experiments/run_camodel_arm.py"


class RunCAModelArmTests(unittest.TestCase):
    def run_arm(self, source, timeout=1.0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / "control.json"
            env_file = root / "env.json"
            env_file.write_text(json.dumps({"ARM_TEST": "child-only"}))
            env = dict(os.environ)
            env.pop("LD_PRELOAD", None)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--output", str(root / "logs"),
                 "--control-path", str(control), "--timeout", str(timeout),
                 "--cwd", str(root), "--env-json", str(env_file), "--",
                 sys.executable, "-u", "-c", source],
                env=env, capture_output=True, text=True, timeout=5,
            )
            state = json.loads(control.read_text())
            stdout = (root / "logs/stdout.log").read_text()
            stderr = (root / "logs/stderr.log").read_text()
            return result, state, stdout, stderr

    def test_success_keeps_logs_and_child_environment(self):
        result, state, out, err = self.run_arm(
            "import os,sys; print(os.environ['ARM_TEST']); print('stderr-kept',file=sys.stderr)")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["child_returncode"], 0)
        self.assertTrue(state["cleanup_complete"])
        self.assertEqual(out, "child-only\n")
        self.assertEqual(err, "stderr-kept\n")

    def test_nonzero_exit_is_not_rewritten(self):
        result, state, out, err = self.run_arm(
            "import sys; print('before-failure'); print('cause',file=sys.stderr); sys.exit(17)")
        self.assertEqual(result.returncode, 17)
        self.assertEqual(state["child_returncode"], 17)
        self.assertEqual(state["supervisor_returncode"], 17)
        self.assertEqual(state["status"], "failed")
        self.assertTrue(state["cleanup_complete"])
        self.assertEqual(out, "before-failure\n")
        self.assertEqual(err, "cause\n")

    def test_short_deadline_stops_real_child_and_preserves_log(self):
        result, state, out, _ = self.run_arm(
            "import time; print('started',flush=True); time.sleep(20)", timeout=0.2)
        self.assertEqual(result.returncode, 124)
        self.assertTrue(state["timed_out"])
        self.assertTrue(state["cleanup_complete"])
        self.assertEqual(state["child_returncode"], -signal.SIGTERM)
        self.assertLess(state["elapsed_seconds"], 1)
        self.assertEqual(state["cleanup_grace_seconds"], 12)
        self.assertEqual(out, "started\n")
        with self.assertRaises(ProcessLookupError):
            os.kill(state["child_pid"], 0)

    def test_interrupt_reaps_owned_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / "control.json"
            env = dict(os.environ)
            env.pop("LD_PRELOAD", None)
            supervisor = subprocess.Popen(
                [sys.executable, str(SCRIPT), "--output", str(root / "logs"),
                 "--control-path", str(control), "--cwd", str(root), "--",
                 sys.executable, "-u", "-c", "import time; time.sleep(20)"],
                env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if control.exists():
                        state = json.loads(control.read_text())
                        if state.get("status") == "running":
                            break
                    time.sleep(0.02)
                else:
                    self.fail("supervisor did not publish running child")
                supervisor.send_signal(signal.SIGTERM)
                self.assertEqual(supervisor.wait(timeout=3), 143)
                state = json.loads(control.read_text())
                self.assertEqual(state["interrupt_signal"], signal.SIGTERM)
                self.assertFalse(state["timed_out"])
                self.assertTrue(state["cleanup_complete"])
                self.assertEqual(state["child_returncode"], -signal.SIGTERM)
            finally:
                if supervisor.poll() is None:
                    supervisor.send_signal(signal.SIGTERM)
                    supervisor.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
