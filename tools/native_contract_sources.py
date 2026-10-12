"""Archive #74-78/#127/#128: pinned native sources for local CPU contracts.

Read existing reference checkouts without modifying them. A fresh Git checkout
uses the byte-identical, hash-checked bundle shipped under references/ instead.
Neither mode imports installed vLLM nor participates in production deployment.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class NativeSourceError(ValueError):
    pass


class NativeSources:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "tests/native_contract_manifest.json").read_text())
        self.files = {f'{ref["path"]}/{path}': (ref, expected)
                      for ref in self.manifest["references"]
                      for path, expected in ref["files"].items()}
        self._archive = None

    def mode(self, reference):
        # A present but incomplete/wrong checkout is an error, never hidden
        # by silently selecting the bundle for individual missing files.
        return "checkout" if (self.root / reference["path"]).exists() else "bundled_snapshot"

    def _bundle(self):
        if self._archive is None:
            info = self.manifest["source_bundle"]
            path = self.root / info["path"]
            try:
                payload = path.read_bytes()
            except OSError as error:
                raise NativeSourceError(f"missing native CPU source bundle: {path}; "
                    "update the complete project checkout, including references/native_cpu_contracts.zip") from error
            if hashlib.sha256(payload).hexdigest() != info["sha256"]:
                raise NativeSourceError(f"native source bundle hash mismatch: {path}")
            try:
                archive = zipfile.ZipFile(io.BytesIO(payload))
                names = archive.namelist()
                if len(names) != len(set(names)) or set(names) != set(self.files):
                    raise NativeSourceError("native source bundle members differ from the pinned manifest")
                self._archive = archive
            except zipfile.BadZipFile as error:
                raise NativeSourceError(f"invalid native source bundle: {path}") from error
        return self._archive

    def read_bytes(self, path):
        if path not in self.files:
            raise NativeSourceError(f"unrecorded native contract source: {path}")
        reference, expected = self.files[path]
        if self.mode(reference) == "checkout":
            try:
                source = (self.root / path).read_bytes()
            except OSError as error:
                raise NativeSourceError(f"incomplete reference checkout: {self.root / path}") from error
        else:
            archive = self._bundle()
            if archive.getinfo(path).file_size > 8 * 1024 * 1024:
                raise NativeSourceError(f"oversized native CPU source: {path}")
            try:
                source = archive.read(path)
            except (OSError, zipfile.BadZipFile) as error:
                raise NativeSourceError(f"unreadable native CPU source: {path}") from error
        if hashlib.sha256(source).hexdigest() != expected["sha256"]:
            raise NativeSourceError(f"native source hash mismatch: {path}; review the pinned revision")
        return source

    def read_text(self, path):
        return self.read_bytes(path).decode("utf-8")

    def verify(self):
        evidence = []
        for reference in self.manifest["references"]:
            mode = self.mode(reference)
            if mode == "checkout":
                tree = self.root / reference["path"]
                try:
                    def git(*args):
                        return subprocess.check_output(["git", "-C", str(tree), *args],
                            text=True, stderr=subprocess.STDOUT).strip()
                    if (git("rev-parse", "HEAD") != reference["observed_commit"]
                            or git("rev-parse", "HEAD^{tree}") != reference["git_tree"]
                            or git("status", "--porcelain")):
                        raise NativeSourceError(f"reference checkout revision/tree/status changed: {tree}")
                except (OSError, subprocess.CalledProcessError) as error:
                    raise NativeSourceError(f"cannot verify reference checkout: {tree}: {error}") from error
            for path in reference["files"]:
                self.read_bytes(f'{reference["path"]}/{path}')
            evidence.append({"path": reference["path"], "mode": mode,
                "recorded_commit": reference["observed_commit"],
                "recorded_git_tree": reference["git_tree"],
                "git_checkout_verified": mode == "checkout",
                "files_sha256_verified": len(reference["files"])})
        return evidence


@lru_cache(maxsize=1)
def native_sources():
    return NativeSources()
