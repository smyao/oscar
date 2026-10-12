"""Archive #132-135/#145/#156: reject incomplete diagnostic evidence.

Host file/contract tests only; these do not execute or certify an NPU kernel.
"""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from tools.experiments import decode_completion_probe as probe


class CompletionFiles(unittest.TestCase):
    def fixture(self, root):
        (root / "log_ca").mkdir()
        samples = []
        for engine in probe.ENGINES:
            for kind in ("instr_log", "instr_popped_log"):
                name = f"core0.{engine}.{kind}.dump"
                data = f"[info] [00000001] {engine} complete instruction\n".encode()
                (root / "log_ca" / name).write_bytes(data)
                samples.append({"source": name, "source_bytes": len(data), "captured_bytes": len(data),
                                "complete_for_actor": True, "prefix_capped": False})
        return {"profile_counters": False, "instruction_samples": samples}

    def test_all_six_full_files_survive_lossless_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self.fixture(root)
            original = {p.name: p.read_bytes() for p in (root / "log_ca").iterdir()}
            result = probe.capture(root, report, b"")
            self.assertEqual(result["status"], "passed")
            self.assertEqual(len(result["files"]), 6)
            for row in result["files"]:
                with gzip.open(root / row["path"], "rb") as stream:
                    self.assertEqual(stream.read(), original[row["source"]])
                self.assertFalse((root / "log_ca" / row["source"]).exists())

    def test_capped_prefix_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self.fixture(root)
            report["instruction_samples"][0]["prefix_capped"] = True
            with self.assertRaisesRegex(ValueError, "cap"):
                probe.capture(root, report, b"")

    def test_rotation_and_partial_line_rejected(self):
        for rotated in (False, True):
            with self.subTest(rotated=rotated), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                report = self.fixture(root)
                first = root / "log_ca" / report["instruction_samples"][0]["source"]
                if rotated:
                    first.with_name(first.name + ".1").write_bytes(b"old chunk\n")
                else:
                    first.write_bytes(first.read_bytes().rstrip(b"\n"))
                with self.assertRaises(ValueError):
                    probe.capture(root, report, b"")

    def test_wrong_gzip_witness_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.gz"
            data = b"complete\n"
            path.write_bytes(gzip.compress(data))
            with self.assertRaises(ValueError):
                probe.verify_gzip(path, len(data) - 1, hashlib.sha256(data).hexdigest())

    def test_counts_bind_real_task_bytes_and_profile_work(self):
        rows = []
        for token in range(8):
            for source in range(3):
                row = [0] * 16
                row[1] = 4 if token in (0, 4) else 0
                row[3], row[4] = ((64, 1089), (0, 319), (0, 4))[source]
                row[7], row[8] = source, 1341
                rows.append(struct.pack("<16q", *row))
        tasks = b"".join(rows)
        values = [0] * 288
        for engine in range(3):
            for source, work in enumerate(((2, 10, 2050), (2, 4, 638), (2, 2, 8), (6, 16, 2696))):
                offset = (engine * 4 + source) * 24
                values[offset:offset + 3] = work
        profile = struct.pack("<288q", *values)
        report = {"case": probe.CASE, "mode": "q4", "cores": 1,
                  "attrs": [8, 6, 1, 256, 2, 0, 24, 1],
                  "all_sources_and_tasks": True, "task_filtering": False,
                  "input_sha256": {"tasks": hashlib.sha256(tasks).hexdigest()},
                  "outputs": {"profile": {"sha256": hashlib.sha256(profile).hexdigest()}}}
        counts = probe.expected_counts(report, tasks, profile)
        self.assertEqual([sum(counts[name].values()) for name in probe.ENGINES], [107, 247, 231])
        with self.assertRaisesRegex(ValueError, "identity"):
            probe.expected_counts(report, tasks[:-1] + b"\x01", profile)


if __name__ == "__main__":
    unittest.main()
