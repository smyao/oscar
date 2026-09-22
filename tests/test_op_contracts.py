# Archive #34/#36/#87/#108: executable byte-boundary and incomplete-capability checks.
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from oscar_ascend.ops.contracts import (
    PackedStorage, OperatorContractError, SOURCE_CAPABILITIES,
    missing_capabilities, merge_shapes,
)
from oscar_ascend.ops.loader import OperatorUnavailable, require_production_ops, validate_build_artifacts


class OperatorContractTests(unittest.TestCase):
    def storage(self, **updates):
        kwargs = dict(physical_block_tokens=256, physical_num_blocks=3,
                      heads=2, head_dim=256, raw_ssm_offset=192,
                      physical_page_stride=70000, raw_num_bytes=210192)
        kwargs.update(updates)
        return PackedStorage(**kwargs)

    def test_subpages_map_without_crossing_conv_or_next_ssm_page(self):
        storage = self.storage()
        self.assertEqual(storage.slot_offset(-1), None)
        self.assertEqual(storage.slot_offset(0), 192)
        self.assertEqual(storage.slot_offset(128), 192+128*272)
        self.assertEqual(storage.slot_offset(256), 70192)
        self.assertLessEqual(storage.slot_offset(255,1)+136,70192)
        with self.assertRaises(OperatorContractError):
            storage.slot_offset(768)

    def test_exact_pr_layout(self):
        self.assertEqual(tuple(self.storage().field_offsets.values()),(0,64,66,68,132,134))
        self.assertEqual(self.storage().head_bytes,136)

    def test_underallocation_is_rejected(self):
        for updates in ({"physical_page_stride": 69631},
                        {"raw_num_bytes": 210191},
                        {"physical_block_tokens": 127},
                        {"head_dim": 80}):
            with self.subTest(updates=updates), self.assertRaises(OperatorContractError):
                self.storage(**updates)

    def test_bounded_split_shape_and_empty_batch(self):
        self.assertEqual(merge_shapes(0,3,256)["output"],(0,256))
        with self.assertRaises(OperatorContractError):
            merge_shapes(1,129,256)

    def test_two_components_cannot_claim_complete_production(self):
        primitive_only={"store_int2_out","merge_lse_out"}
        self.assertIn("attention_cv_out",missing_capabilities(primitive_only))
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory)/"manifest.json"
            manifest.write_text(json.dumps({"abi_version":1,
                "execution_interface":"direct_launch",
                "source_capabilities":sorted(primitive_only)}))
            # Fails before importing torch_npu or attempting nonexistent .so.
            with self.assertRaisesRegex(OperatorUnavailable,"not implemented.*attention_cv_out"):
                require_production_ops(manifest)

    def fixture_build(self, directory):
        root = Path(directory).resolve()
        extension, kernels = root/"_oscar_ascend_ops.so", root/"liboscar_ascend_kernels.so"
        extension.write_bytes(b"test-extension-content")
        kernels.write_bytes(b"test-kernel-content")
        configuration = {"source": {}, "soc": "ascend910b4"}
        signature = hashlib.sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest()
        state = {"build":"passed", "configuration":configuration, "signature":signature,
                 "extension":str(extension), "kernel_library":str(kernels),
                 "sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (extension,kernels)}}
        manifest = {"abi_version":1,"execution_interface":"direct_launch",
                    "source_capabilities":sorted(SOURCE_CAPABILITIES),
                    **{k:v for k,v in state.items() if k!="configuration"}}
        state_path = root/"oscar_build_signature.json"
        manifest_path = root/"build_manifest.json"
        state_path.write_text(json.dumps(state))
        manifest_path.write_text(json.dumps(manifest))
        return manifest_path, state_path, extension, manifest, state

    def test_configure_manifest_is_not_build_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, _, manifest, _ = self.fixture_build(directory)
            del manifest["build"]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(OperatorUnavailable,"not completed"):
                validate_build_artifacts(path)

    def test_artifact_replacement_with_preserved_mtime_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, extension, _, _ = self.fixture_build(directory)
            validate_build_artifacts(path)
            before = extension.stat()
            extension.write_bytes(b"changed-extension-content")
            os.utime(extension, ns=(before.st_atime_ns,before.st_mtime_ns))
            with self.assertRaisesRegex(OperatorUnavailable,"SHA256 mismatch"):
                validate_build_artifacts(path)

    def test_configuration_digest_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path, state_path, _, _, state = self.fixture_build(directory)
            state["configuration"]["soc"] = "ascend910b"
            state_path.write_text(json.dumps(state))
            with self.assertRaisesRegex(OperatorUnavailable,"configuration signature"):
                validate_build_artifacts(path)

    def test_disagreeing_runtime_manifest_and_build_state_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, _, manifest, _ = self.fixture_build(directory)
            manifest["signature"] = "a"*64
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(OperatorUnavailable,"signature differs"):
                validate_build_artifacts(path)


if __name__ == "__main__":
    unittest.main()
