"""Archive #132-135/#150: reject misleading same-shape/precision evidence.

Stdlib-only evidence-gate tests. Synthetic report dictionaries are not kernel,
CPU-debug, CAModel, NPU, or performance validation.
"""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from csrc.experiments.request_window_decode_profile import compare as gate


class InstrumentationEvidenceGate(unittest.TestCase):
    def reports(self):
        normal = {
            "mode": "q4", "kernel": "oscar_bundle_normal_q4_kernel", "profile_counters": False,
            "workspace_bytes_per_core": 1146880, "phase_profile_schema": None,
            "attrs": [128, 6, 1, 256, 32, 9, 1152, 512, 96, 64, 69632, 82688, 323, 64, 256, 3, 3],
            "cores": 20, "case": "q4_r32_n128_c1025_s3", "scale": .0625,
            "input_sha256": {"q": "same"}, "oracle_sha256": {"output": "same"},
            "cpu_baseline_sha256": {}, "library_sha256": "same-library",
            "simulator_configuration": {"same": True}, "cpu_reference": "not_available",
            "cpu_reference_policy": "optional", "task_ticks": 100000,
            "outputs": {key: {"sha256": key} for key in ("partial", "lse", "status")},
        }
        instrumented = copy.deepcopy(normal)
        instrumented.update(kernel="oscar_bundle_profile_q4_kernel", profile_counters=True,
                            phase_profile_schema=2, task_ticks=104000)
        host = {"cores": 20, "hard_limit_seconds": 600}
        return normal, instrumented, host

    def evaluate(self, normal, profile, host):
        # checked_arm's real completion/hash/oracle tests remain in the reused
        # helper. Mock only that boundary to isolate our permitted ABI delta.
        with patch.object(gate, "checked_arm", side_effect=[(normal, host), (profile, host)]):
            return gate.compare(Path("unused"), normal["case"])

    def test_only_profile_abi_difference_and_signed_perturbation(self):
        normal, profile, host = self.reports()
        result = self.evaluate(normal, profile, host)
        self.assertEqual(result["perturbation_gate"], "passed")
        self.assertEqual(result["profile_over_normal"], 1.04)
        profile["task_ticks"] = 94000
        self.assertEqual(self.evaluate(normal, profile, host)["perturbation_gate"], "failed")

    def test_padding_one_request_cannot_masquerade_as_r32(self):
        normal, profile, host = self.reports()
        normal["attrs"][4] = profile["attrs"][4] = 1
        with self.assertRaisesRegex(ValueError, "real R32"):
            self.evaluate(normal, profile, host)

    def test_bitwise_status_and_library_identity_are_required(self):
        for mutation, expected in ((lambda p: p["outputs"]["status"].update(sha256="different"), "status bits"),
                                   (lambda p: p.update(library_sha256="other-library"), "library_sha256"),
                                   (lambda p: p.update(kernel="oscar_decode_simd_profile_q4_kernel"), "symbol or ABI")):
            normal, profile, host = self.reports()
            mutation(profile)
            with self.assertRaisesRegex(ValueError, expected):
                self.evaluate(normal, profile, host)


if __name__ == "__main__":
    unittest.main()
