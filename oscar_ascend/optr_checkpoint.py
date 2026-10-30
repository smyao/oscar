"""Archive #97/#120/#165/#172: atomic, authenticated offline fit checkpoints.

These are resumable work products, never serving artifacts. A checkpoint
cannot bypass validation/test or substitute for a complete recipe5 PT.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

from .optr_artifacts import file_sha256

def _identity_directory(directory, identity):
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return Path(directory) / hashlib.sha256(encoded.encode()).hexdigest()


def fitting_implementation_digest():
    root = Path(__file__).resolve().parents[1]
    names = ("oscar_ascend/optr_calibration.py", "oscar_ascend/optr_sequential.py", "oscar_ascend/optr_traces.py",
             "oscar_ascend/optr_selection.py", "oscar_ascend/optr_shared.py", "oscar_ascend/optr_shared_cache.py",
             "oscar_ascend/optr_artifacts.py", "oscar_ascend/optr_checkpoint.py",
             "oscar_ascend/optr_oscar_seed.py", "tools/prepare_optr_seed.py",
             "tools/prepare_optr_rotations.py", "tools/optr_parallel.py", "tools/optr_fit_worker.py")
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode() + b"\0" + (root / name).read_bytes())
    return digest.hexdigest()


class FitCheckpoints:
    def __init__(self, directory, identity):
        # The implementation identity belongs here, not in the native trace
        # profile: fixing execution must reuse already completed sampling.
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.identity = json.loads(encoded)
        self.root = _identity_directory(directory, self.identity)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key):
        if not isinstance(key, str) or not re.fullmatch(r"layer-\d+(?:-head-\d+|-shared)?", key):
            raise ValueError("invalid offline checkpoint owner")
        return self.root / (key + ".json")

    def load(self, key):
        manifest_path = self._path(key)
        if manifest_path.is_file():
            return self._load_from(manifest_path, key, self.identity)
        # v4 changes the objective and selection. Even reviewed v3 serial or
        # parallel fits cannot supply completed work to this implementation;
        # only authenticated raw captures are compatible across that change.
        return None

    @staticmethod
    def _load_from(manifest_path, key, identity):
        import torch
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("schema_version") != 1 or manifest.get("key") != key or
                manifest.get("identity") != identity):
            raise ValueError("offline checkpoint manifest identity differs")
        name = manifest.get("file")
        if not isinstance(name, str) or not re.fullmatch(re.escape(key) + r"-[0-9a-f]{32}\.pt", name):
            raise ValueError("offline checkpoint file does not belong to its owner")
        path = manifest_path.parent / name
        if not path.is_file() or file_sha256(path) != manifest.get("sha256"):
            raise ValueError("offline checkpoint file hash differs")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(saved, dict) or saved.get("identity") != identity or
                saved.get("key") != key or not isinstance(saved.get("payload"), dict)):
            raise ValueError("offline checkpoint payload identity differs")
        return saved["payload"]

    def save(self, key, payload):
        import torch
        from tools.phase import atomic_json
        manifest_path = self._path(key)
        if not isinstance(payload, dict):
            raise ValueError("offline checkpoint payload must be a mapping")
        name = key + "-" + uuid.uuid4().hex + ".pt"
        fd, temporary = tempfile.mkstemp(prefix=".pending-", suffix=".pt", dir=self.root)
        os.close(fd)
        try:
            torch.save({"identity": self.identity, "key": key, "payload": payload}, temporary)
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
            digest = file_sha256(temporary)
            os.replace(temporary, self.root / name)
            # Only this atomic pointer makes a generation reusable. A process
            # interrupted before it leaves an unreferenced file, not success.
            atomic_json(manifest_path, {"schema_version": 1, "key": key,
                "identity": self.identity, "file": name, "sha256": digest})
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
