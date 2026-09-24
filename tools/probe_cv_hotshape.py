# Archive #70-73/#85/#125/#126/#143-#146 and startup D.4: this is a diagnostic
# comparison of one bounded CV operator, not a whole-service speed certificate.
# D.4 four questions: (1) measure only attention_cv_out for a long mixed
# prefill shape; (2) the old failed route spent ~6.5 s restoring full history,
# while its FIA phase was ~19 ms; (3) production CV still reads packed INT2
# pages and exact BF16 windows, and only this offline oracle decodes history;
# (4) report target Event times and frozen numerical error separately, never
# infer service performance from this isolated measurement.
"""One-NPU, one-artifact CV hot-shape diagnostic for paired old/new builds.

Run baseline in a separate child *before* rebuilding. The baseline artifact is
accepted only if its signed source map equals the pinned same-project revision.
Candidate uses the normal production loader and current source fingerprint.
Neither variant substitutes CPU execution for an NPU operator.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

from .environment import file_fingerprint
from .phase import atomic_json

ROOT = Path(__file__).resolve().parents[1]
BASELINE_REVISION = "fe0e925e7ef78bfb64217a300031502fc4a7b7bc"
REQUIRED_OPS = frozenset({"prepare_attention_tasks_out", "attention_cv_out", "merge_lse_out"})
QLENS = (4, 4, 4, 11492)
CONTEXTS = (20032, 23032, 27032, 18508)
DECODE_QLENS = (4,) * 32
DECODE_CONTEXTS = (20000, 23000, 27000, 30000) * 8
HEADS, KV_HEADS, DIM = 6, 1, 256
BLOCK_TOKENS, SINK, RECENT, SPECULATIVE = 512, 64, 256, 3
PREFIX_BYTES = 64
SEED = 46783
WARMUP, REPEATS = 2, 5
OLD_QUERY_ROWS, OLD_KV_ROWS = 128, 256
PROFILE_ENGINES = ("aic", "aiv0", "aiv1")
PROFILE_SOURCES = ("history", "window", "current", "total")
PROFILE_FIELDS = (
    "tasks", "kv_tiles", "kv_rows", "aiv_load_publish", "aiv_wait_qk",
    "aiv_softmax_total", "aiv_mask_finite", "aiv_v2", "aiv_wait_pv",
    "aiv_pv_add", "aiv_final", "aic_wait_kv", "aic_qk", "aic_wait_p",
    "aic_pv", "aic_wait_consume", "aic_rotation", "process_span",
    "empty_tasks", "errors",
)
PROFILE_COUNT_FIELDS = frozenset({"tasks", "kv_tiles", "kv_rows", "empty_tasks", "errors"})


class HotShapeError(RuntimeError):
    pass


class BaselineUnavailable(HotShapeError):
    pass


def git_csrc_fingerprint(root: Path = ROOT, revision: str = BASELINE_REVISION) -> dict[str, str]:
    """SHA256 of committed csrc blobs; no checkout/copy/source execution."""
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root,
                         capture_output=True, text=True, check=True).stdout.strip()
    if Path(top).resolve() != root.resolve():
        raise BaselineUnavailable("pinned baseline revision is outside this project")
    resolved = subprocess.run(["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
                              cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    if resolved != revision:
        raise BaselineUnavailable("pinned baseline revision did not resolve exactly")
    entries = subprocess.run(["git", "ls-tree", "-r", "-z", revision, "--", "csrc"],
                             cwd=root, capture_output=True, check=True).stdout
    result: dict[str, str] = {}
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        descriptor, path = entry.split(b"\t", 1)
        mode, kind, object_id = descriptor.split()
        if kind != b"blob" or mode == b"120000" or not path.startswith(b"csrc/"):
            raise BaselineUnavailable("pinned baseline csrc tree contains an unsupported entry")
        name = path[len(b"csrc/"):].decode("utf-8")
        blob = subprocess.run(["git", "cat-file", "blob", object_id.decode("ascii")],
                              cwd=root, capture_output=True, check=True).stdout
        result[name] = hashlib.sha256(blob).hexdigest()
    if not result:
        raise BaselineUnavailable("pinned baseline has no csrc fingerprint")
    return result


def _fingerprint_digest(source: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()


def _source_check(configuration: dict, variant: str, *, root: Path = ROOT) -> str:
    source = configuration.get("source")
    if not isinstance(source, dict):
        raise HotShapeError("artifact has no signed source fingerprint")
    expected = git_csrc_fingerprint(root) if variant == "baseline" else file_fingerprint(root / "csrc")
    if source != expected:
        exception = BaselineUnavailable if variant == "baseline" else HotShapeError
        raise exception(f"{variant} artifact source does not match its required csrc fingerprint")
    return _fingerprint_digest(source)


def _select_device(target: dict) -> str:
    devices = target.get("devices")
    if devices != [0, 1, 2, 3] or any(type(device) is not int for device in devices):
        raise HotShapeError("hot-shape target must select explicit physical devices 0,1,2,3")
    selected = ",".join(map(str, devices))
    inherited = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if inherited is not None and inherited != selected:
        raise HotShapeError(f"inherited NPU selection {inherited!r} differs from target {selected!r}")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = selected
    if target.get("soc_version") != "ascend910b4":
        raise HotShapeError("target SOC must be ascend910b4")
    return selected


def _verify_artifact(variant: str, manifest_path: Path, target: dict) -> tuple[dict, dict, str]:
    from oscar_ascend.ops.loader import OperatorUnavailable, validate_build_artifacts
    if manifest_path.resolve() != (ROOT / "build/ascendc/build_manifest.json").resolve():
        raise HotShapeError("hot-shape probe accepts only this project's build manifest")
    try:
        manifest = validate_build_artifacts(manifest_path)
    except (OperatorUnavailable, OSError, ValueError, KeyError) as exc:
        if variant == "baseline":
            raise BaselineUnavailable(str(exc)) from exc
        raise
    configuration = json.loads((manifest_path.parent / "oscar_build_signature.json").read_text())["configuration"]
    if manifest.get("soc") != target["soc_version"] or configuration.get("soc") != target["soc_version"]:
        exception = BaselineUnavailable if variant == "baseline" else HotShapeError
        raise exception("artifact SOC differs from explicit target SOC")
    required = REQUIRED_OPS | ({"attention_cv_profile_out"} if variant == "candidate" else set())
    if not required.issubset(set(manifest.get("source_capabilities", ()))):
        exception = BaselineUnavailable if variant == "baseline" else HotShapeError
        raise exception("artifact lacks required CV/prepare/merge/profile capabilities")
    source_digest = _source_check(configuration, variant)
    if configuration.get("python") != sys.executable:
        exception = BaselineUnavailable if variant == "baseline" else HotShapeError
        raise exception("artifact Python executable differs from current runtime")
    for package in ("torch", "torch_npu", "pybind11"):
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            exception = BaselineUnavailable if variant == "baseline" else HotShapeError
            raise exception(f"artifact package {package} is not installed") from exc
        if configuration.get("package_versions", {}).get(package) != installed:
            exception = BaselineUnavailable if variant == "baseline" else HotShapeError
            raise exception(f"artifact {package} version differs from current runtime")
    from .build_ops import cann_root
    try:
        cann = cann_root(os.environ).resolve()
    except RuntimeError as exc:
        exception = BaselineUnavailable if variant == "baseline" else HotShapeError
        raise exception(str(exc)) from exc
    if configuration.get("cann") != str(cann):
        exception = BaselineUnavailable if variant == "baseline" else HotShapeError
        raise exception("artifact CANN root differs from current target")
    versions = {}
    for path in (cann / "version.cfg", cann / "version.info", cann / "compiler/version.info"):
        if path.is_file():
            versions[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if configuration.get("cann_versions") != versions:
        exception = BaselineUnavailable if variant == "baseline" else HotShapeError
        raise exception("artifact CANN version fingerprint differs from current target")
    npu_spec = importlib.util.find_spec("torch_npu")
    if npu_spec is None or npu_spec.origin is None or \
            configuration.get("torch_npu") != str(Path(npu_spec.origin).parent):
        exception = BaselineUnavailable if variant == "baseline" else HotShapeError
        raise exception("artifact torch_npu installation differs from current target")
    flags = {key: os.environ.get(key) for key in ("CXX", "CC", "CXXFLAGS")}
    if configuration.get("flags") != flags:
        exception = BaselineUnavailable if variant == "baseline" else HotShapeError
        raise exception("artifact compiler flags differ from current target")
    return manifest, configuration, source_digest


def baseline_availability(manifest_path: Path = ROOT / "build/ascendc/build_manifest.json",
                          target_path: Path = ROOT / "configs/target.json") -> dict:
    """Read-only artifact gate for the parent; does not import torch or touch NPU."""
    try:
        target = json.loads(target_path.read_text())
        if (target.get("devices") != [0, 1, 2, 3]
                or any(type(device) is not int for device in target["devices"])
                or target.get("soc_version") != "ascend910b4"):
            raise BaselineUnavailable("target devices/SOC are not the signed 0-3/A2 configuration")
        manifest, _, source_digest = _verify_artifact("baseline", manifest_path, target)
    except (OSError, ValueError, KeyError, TypeError, AttributeError,
            RuntimeError, subprocess.SubprocessError) as exc:
        return {"status": "not_available", "reason": str(exc),
                "source_revision": BASELINE_REVISION}
    return {"status": "available", "source_revision": BASELINE_REVISION,
            "source_sha256": source_digest, "build_signature": manifest["signature"],
            "artifact_sha256": manifest["sha256"]}


def _load_baseline_extension(manifest: dict):
    """Diagnostic-only counterpart of production loading, after hash checks."""
    from oscar_ascend.ops.contracts import ABI_VERSION
    importlib.import_module("torch_npu")
    kernels = Path(manifest["kernel_library"])
    extension = Path(manifest["extension"])
    handle = ctypes.CDLL(str(kernels), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    spec = importlib.util.spec_from_file_location("_oscar_ascend_ops", extension)
    if spec is None or spec.loader is None:
        raise HotShapeError("baseline extension cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.abi_version() != ABI_VERSION or set(module.capabilities()) != set(manifest["source_capabilities"]):
        raise HotShapeError("baseline extension ABI/capabilities differ from signed manifest")
    return handle, module


def _load_ops(variant: str, manifest: dict, manifest_path: Path):
    if variant == "baseline":
        handle, module = _load_baseline_extension(manifest)
        return handle, module
    from oscar_ascend.ops.loader import require_capabilities
    module = require_capabilities(REQUIRED_OPS | {"attention_cv_profile_out"}, manifest_path)
    return None, module


def splits_for_shape(tokens: int, cores: int, *, query_rows: int, capacity: int) -> int:
    if tokens <= 0 or cores <= 0 or query_rows < HEADS // KV_HEADS or capacity < tokens:
        raise HotShapeError("invalid CV split geometry")
    query_tile = query_rows // (HEADS // KV_HEADS)
    groups = (tokens + query_tile - 1) // query_tile * KV_HEADS
    cube_parallelism = (cores + groups - 1) // groups
    return max(1, min(32, cube_parallelism, capacity // tokens))


def workspace_per_core_bytes(dim: int, current_rows: int, current_kv_rows: int) -> int:
    old = ((2 * OLD_QUERY_ROWS + 2 * OLD_KV_ROWS) * dim
           + OLD_QUERY_ROWS * OLD_KV_ROWS) * 4
    current = ((2 * current_rows + 2 * current_kv_rows) * dim
               + current_rows * current_kv_rows) * 4
    return max(old, current)


def sample_local_indices(length: int) -> tuple[int, ...]:
    candidates = (0, 1, 2, 3, 20, 21, 22, 63, 64, 127, 128,
                  255, 256, 257, 2047, 2048, length // 2, length - 1)
    return tuple(sorted({value for value in candidates if 0 <= value < length}))


def _physical_pages(lengths: tuple[int, ...], contexts: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    budgets = [math.ceil((length + context) / BLOCK_TOKENS)
               for length, context in zip(lengths, contexts)]
    pages = list(range(sum(budgets)))
    random.Random(SEED).shuffle(pages)
    result = []
    cursor = 0
    for budget in budgets:
        result.append(tuple(pages[cursor:cursor + budget]))
        cursor += budget
    if sorted(page for request in result for page in request) != list(range(sum(budgets))):
        raise HotShapeError("hot-shape physical pages are not disjoint")
    return tuple(result)


def _tensor_hash(named: dict) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(named.items()):
        if not tensor.is_contiguous() or tensor.device.type != "cpu":
            raise HotShapeError(f"fixture tensor {name} is not contiguous CPU data")
        hasher.update(name.encode() + b"\0")
        hasher.update(str(tuple(tensor.shape)).encode() + b"\0")
        hasher.update(str(tensor.dtype).encode() + b"\0")
        # CPU-only C copy; no NumPy dependency or Python per-byte iteration.
        # data_ptr starts at the logical view, including a storage offset.
        hasher.update(ctypes.string_at(tensor.data_ptr(), tensor.numel() * tensor.element_size()))
    return hasher.hexdigest()


def _hadamard(torch, dim: int):
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < dim:
        matrix = torch.cat((torch.cat((matrix, matrix), dim=1),
                            torch.cat((matrix, -matrix), dim=1)), dim=0) / math.sqrt(2)
    return matrix.contiguous()


def _oracle_rows(torch, reference_attention, q, old_k, old_v, packed, rk, rv,
                 context: int, qbegin: int, qlength: int, scale: float):
    from oscar_ascend.ops.reference import decode_kv
    decoded_k, decoded_v = decode_kv(packed, DIM)
    restored_k, restored_v = decoded_k @ rk.T, decoded_v @ rv.T
    actual_k, actual_v = old_k.float(), old_v.float()
    expected = {}
    previous_cut = None
    selected_k = selected_v = None
    for local in sample_local_indices(qlength):
        cut = min(context, max(SINK, context + local + 1 - RECENT))
        if cut != previous_cut:
            selected_k, selected_v = actual_k.clone(), actual_v.clone()
            selected_k[SINK:cut] = restored_k[SINK:cut]
            selected_v[SINK:cut] = restored_v[SINK:cut]
            previous_cut = cut
        index = qbegin + local
        result = reference_attention(q[index:index + 1], selected_k, selected_v,
                                     scale=scale, causal=False)
        expected[index] = (result.output[0], result.lse[0])
    return expected


def build_fixture(torch, *, scale: float):
    """Synthetic paired inputs; no user's dataset or complete dense Q×K scores."""
    from oscar_ascend.ops.reference import attention, encode_kv
    lengths, contexts = QLENS, CONTEXTS
    pages_by_request = _physical_pages(lengths, contexts)
    blocks = sum(map(len, pages_by_request))
    tokens = sum(lengths)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    q = torch.randn((tokens, HEADS, DIM), generator=generator).to(torch.bfloat16)
    ck = torch.randn((tokens, KV_HEADS, DIM), generator=generator).to(torch.bfloat16)
    cv = torch.randn((tokens, KV_HEADS, DIM), generator=generator).to(torch.bfloat16)
    rk = _hadamard(torch, DIM)
    rv = rk.flip(1).contiguous()
    qr = (q.float() @ rk).contiguous()
    slot_bytes = DIM // 2 + 8
    stride = BLOCK_TOKENS * KV_HEADS * slot_bytes
    raw = torch.full((PREFIX_BYTES + blocks * stride,), 0xA5, dtype=torch.uint8)
    raw_pages = raw[PREFIX_BYTES:].view(blocks, BLOCK_TOKENS, KV_HEADS, slot_bytes)
    window_rows = SINK + RECENT + SPECULATIVE
    wk = torch.full((blocks, window_rows, KV_HEADS, DIM), float("nan"), dtype=torch.bfloat16)
    wv = torch.full_like(wk, float("nan"))
    tags = torch.full((blocks, window_rows), -1, dtype=torch.int64)
    columns = math.ceil(max(context + length for context, length in zip(contexts, lengths)) / 128)
    table = torch.full((len(lengths), columns), -1, dtype=torch.int32)
    starts = [0]
    lens = []
    slots = torch.empty((tokens,), dtype=torch.int64)
    expected = {}
    qbegin = 0
    for request, (length, context, pages) in enumerate(zip(lengths, contexts, pages_by_request)):
        old_k = torch.randn((context, KV_HEADS, DIM), generator=generator).to(torch.bfloat16)
        old_v = torch.randn((context, KV_HEADS, DIM), generator=generator).to(torch.bfloat16)
        packed = encode_kv(old_k.float() @ rk, old_v.float() @ rv)
        for logical_page, physical in enumerate(pages):
            first = logical_page * BLOCK_TOKENS
            count = min(BLOCK_TOKENS, context - first)
            if count > 0:
                raw_pages[physical, :count] = packed[first:first + count]
            for quarter in range(BLOCK_TOKENS // 128):
                column = logical_page * (BLOCK_TOKENS // 128) + quarter
                if column < columns:
                    table[request, column] = physical * (BLOCK_TOKENS // 128) + quarter
        # The exact source reads only sink and most recent 256 old positions.
        for position in sorted(set(range(min(SINK, context)))
                               | set(range(max(0, context - RECENT), context))):
            physical = pages[position // BLOCK_TOKENS]
            inpage = position % BLOCK_TOKENS
            row = position if position < SINK else SINK + inpage % (RECENT + SPECULATIVE)
            wk[physical, row] = old_k[position]
            wv[physical, row] = old_v[position]
            tags[physical, row] = inpage
        for local in range(length):
            position = context + local
            physical = pages[position // BLOCK_TOKENS]
            slots[qbegin + local] = physical * BLOCK_TOKENS + position % BLOCK_TOKENS
        expected.update(_oracle_rows(torch, attention, q, old_k, old_v, packed,
                                     rk, rv, context, qbegin, length, scale))
        qbegin += length
        starts.append(qbegin)
        lens.append(context + length)
    tensors = {"q": q, "qr": qr, "ck": ck, "cv": cv, "rv": rv, "raw": raw,
               "table": table, "wk": wk, "wv": wv, "tags": tags,
               "starts": torch.tensor(starts, dtype=torch.int32),
               "lens": torch.tensor(lens, dtype=torch.int32), "slots": slots}
    fixture_hash = _tensor_hash(tensors)
    return {"cpu": tensors, "expected": expected, "hash": fixture_hash,
            "pages": pages_by_request, "blocks": blocks, "stride": stride,
            "tokens": tokens, "columns": columns, "qlens": lengths,
            "contexts": contexts, "source2_suppressed": True,
            "page_policy": "disjoint_physical_pages"}


def build_decode_fixture(torch, *, scale: float):
    """32 independent old histories and disjoint physical pages, q4 each.

    Only one request's BF16 old K/V and decoded offline oracle are materialized
    at a time. The retained packed arena and exact window are what production
    CV reads. This is eager operator evidence, not graph replay or user data.
    """
    from oscar_ascend.ops.reference import attention, decode_kv, encode_kv
    lengths, contexts = DECODE_QLENS, DECODE_CONTEXTS
    pages_by_request = _physical_pages(lengths, contexts)
    blocks = sum(map(len, pages_by_request))
    tokens = sum(lengths)
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    q = torch.randn((tokens, HEADS, DIM), generator=generator).to(torch.bfloat16)
    ck = torch.randn((tokens, KV_HEADS, DIM), generator=generator).to(torch.bfloat16)
    cv = torch.randn((tokens, KV_HEADS, DIM), generator=generator).to(torch.bfloat16)
    rk = _hadamard(torch, DIM)
    rv = rk.flip(1).contiguous()
    qr = (q.float() @ rk).contiguous()
    slot_bytes = DIM // 2 + 8
    stride = BLOCK_TOKENS * KV_HEADS * slot_bytes
    raw = torch.full((PREFIX_BYTES + blocks * stride,), 0xA5, dtype=torch.uint8)
    raw_pages = raw[PREFIX_BYTES:].view(blocks, BLOCK_TOKENS, KV_HEADS, slot_bytes)
    window_rows = SINK + RECENT + SPECULATIVE
    wk = torch.full((blocks, window_rows, KV_HEADS, DIM), float("nan"), dtype=torch.bfloat16)
    wv = torch.full_like(wk, float("nan"))
    tags = torch.full((blocks, window_rows), -1, dtype=torch.int64)
    columns = math.ceil(max(context + length for context, length in zip(contexts, lengths)) / 128)
    table = torch.full((len(lengths), columns), -1, dtype=torch.int32)
    starts = [0]
    lens = []
    slots = torch.empty(tokens, dtype=torch.int64)
    expected = {}
    qbegin = 0
    for request, (length, context, pages) in enumerate(zip(lengths, contexts, pages_by_request)):
        old_k = torch.randn((context, KV_HEADS, DIM), generator=generator).to(torch.bfloat16)
        old_v = torch.randn((context, KV_HEADS, DIM), generator=generator).to(torch.bfloat16)
        packed = encode_kv(old_k.float() @ rk, old_v.float() @ rv)
        decoded_k, decoded_v = decode_kv(packed, DIM)
        restored_k, restored_v = decoded_k @ rk.T, decoded_v @ rv.T
        for logical_page, physical in enumerate(pages):
            first = logical_page * BLOCK_TOKENS
            count = min(BLOCK_TOKENS, context - first)
            if count > 0:
                raw_pages[physical, :count] = packed[first:first + count]
            for quarter in range(BLOCK_TOKENS // 128):
                column = logical_page * (BLOCK_TOKENS // 128) + quarter
                if column < columns:
                    table[request, column] = physical * (BLOCK_TOKENS // 128) + quarter
        for position in sorted(set(range(min(SINK, context)))
                               | set(range(max(0, context - RECENT), context))):
            physical = pages[position // BLOCK_TOKENS]
            inpage = position % BLOCK_TOKENS
            row = position if position < SINK else SINK + inpage % (RECENT + SPECULATIVE)
            wk[physical, row] = old_k[position]
            wv[physical, row] = old_v[position]
            tags[physical, row] = inpage
        for local in range(length):
            position = context + local
            physical = pages[position // BLOCK_TOKENS]
            slots[qbegin + local] = physical * BLOCK_TOKENS + position % BLOCK_TOKENS
        old_k, old_v = old_k.float(), old_v.float()
        for local in range(length):
            cut = min(context, max(SINK, context + local + 1 - RECENT))
            selected_k, selected_v = old_k.clone(), old_v.clone()
            selected_k[SINK:cut] = restored_k[SINK:cut]
            selected_v[SINK:cut] = restored_v[SINK:cut]
            index = qbegin + local
            dense_k = torch.cat((selected_k, ck[qbegin:index + 1].float()))
            dense_v = torch.cat((selected_v, cv[qbegin:index + 1].float()))
            result = attention(q[index:index + 1], dense_k, dense_v,
                               scale=scale, causal=False)
            expected[index] = (result.output[0], result.lse[0])
        qbegin += length
        starts.append(qbegin)
        lens.append(context + length)
    if sorted(page for pages in pages_by_request for page in pages) != list(range(blocks)):
        raise HotShapeError("decode fixture physical pages overlap")
    tensors = {"q": q, "qr": qr, "ck": ck, "cv": cv, "rv": rv, "raw": raw,
               "table": table, "wk": wk, "wv": wv, "tags": tags,
               "starts": torch.tensor(starts, dtype=torch.int32),
               "lens": torch.tensor(lens, dtype=torch.int32), "slots": slots}
    return {"cpu": tensors, "expected": expected, "hash": _tensor_hash(tensors),
            "pages": pages_by_request, "blocks": blocks, "stride": stride,
            "tokens": tokens, "columns": columns, "qlens": lengths,
            "contexts": contexts, "source2_suppressed": False,
            "page_policy": "32_independent_disjoint_physical_histories"}


def _actual_cores(torch, target: dict) -> int:
    # Same driver query as pinned triton_utils.py:49-53. This standalone
    # operator child must not bootstrap vLLM's model/plugin import graph (#78).
    import triton
    properties = triton.runtime.driver.active.utils.get_device_properties(torch.npu.current_device())
    aic = int(properties.get("num_aicore", -1))
    aiv = int(properties.get("num_vectorcore", -1))
    if aic <= 0 or aiv < 2 * aic:
        raise HotShapeError(f"A2 CV requires two Vector cores per Cube: AIC={aic}, AIV={aiv}")
    requested = target.get("cube_cores")
    cores = min(aic, 32) if requested is None else requested
    if type(cores) is not int or not 1 <= cores <= min(aic, 32):
        raise HotShapeError("target cube_cores is invalid for measured device")
    return cores


def _device_buffers(torch, fixture: dict, device, *, splits: int, cores: int,
                    current_rows: int, current_kv_rows: int):
    tensors = {name: tensor.to(device) for name, tensor in fixture["cpu"].items()}
    tokens = fixture["tokens"]
    count = tokens * KV_HEADS * 3 * splits
    buffers = {
        "tasks": torch.empty((count, 16), dtype=torch.int64, device=device),
        "positions": torch.empty(tokens, dtype=torch.int64, device=device),
        "partial": torch.empty((tokens, HEADS, 3 * splits, DIM), dtype=torch.float32, device=device),
        "lse": torch.empty((tokens, HEADS, 3 * splits), dtype=torch.float32, device=device),
        "status": torch.empty((count, 2), dtype=torch.int32, device=device),
        "workspace": torch.empty(cores * workspace_per_core_bytes(DIM, current_rows, current_kv_rows),
                                 dtype=torch.uint8, device=device),
    }
    return tensors, buffers


def _prepare_ops(torch, ops, tensors: dict, buffers: dict, fixture: dict, *, splits: int):
    from oscar_ascend.integration.current_attention import suppress_current_source_tasks
    ops.prepare_attention_tasks_out(tensors["starts"], tensors["lens"], tensors["slots"],
                                   buffers["tasks"], buffers["positions"], HEADS, KV_HEADS,
                                   SINK, RECENT, splits)
    if fixture["source2_suppressed"]:
        suppress_current_source_tasks(buffers["tasks"], int(tensors["q"].shape[0]), KV_HEADS, splits)
    torch.npu.synchronize()
    expected_positions = torch.cat([torch.arange(context, context + length)
                                    for context, length in zip(fixture["contexts"], fixture["qlens"])])
    torch.testing.assert_close(buffers["positions"].cpu(), expected_positions, atol=0, rtol=0)
    return _tensor_hash({"tasks": buffers["tasks"].cpu()})


def _run_cv(ops, tensors: dict, buffers: dict, fixture: dict, *, splits: int, cores: int,
            scale: float, profile=None):
    arguments = (tensors["q"], tensors["qr"], tensors["ck"], tensors["cv"],
        tensors["rv"], tensors["raw"], tensors["table"], tensors["wk"], tensors["wv"],
        tensors["tags"], buffers["tasks"], buffers["partial"], buffers["lse"],
        buffers["status"], buffers["workspace"], BLOCK_TOKENS, fixture["blocks"],
        PREFIX_BYTES, fixture["stride"], SINK, RECENT, SPECULATIVE, splits, scale, cores)
    if profile is None:
        ops.attention_cv_out(*arguments)
    else:
        ops.attention_cv_profile_out(*arguments, profile)


def _nearest_rank(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def summarize_raw_profile(profile_cpu, *, cores: int) -> dict:
    """Distributions across cores, never an additive raw-tick wall time."""
    if tuple(profile_cpu.shape) != (cores, 3, 4, len(PROFILE_FIELDS)) or \
            str(profile_cpu.dtype) != "torch.int64" or profile_cpu.device.type != "cpu":
        raise HotShapeError("profile kernel returned an invalid int64 [cores,3,4,20] buffer")
    raw = profile_cpu.tolist()
    if any(value < 0 for core in raw for engine in core for source in engine for value in source):
        raise HotShapeError("profile kernel returned negative raw counters")
    if any(raw[core][engine][source][17] != 0
           for core in range(cores) for engine in range(3) for source in range(3)):
        raise HotShapeError("source-local profile process_span must be zero; only total records it")
    if any(raw[core][engine][3][field] != sum(raw[core][engine][source][field]
                                            for source in range(3))
           for core in range(cores) for engine in range(3)
           for field in (*range(17), 18, 19)):
        raise HotShapeError("profile total counters differ from per-source counters")
    if any(len({raw[core][engine][source][field] for engine in range(3)}) != 1
           for core in range(cores) for source in range(4) for field in (0, 1, 2)):
        raise HotShapeError("profile Cube/AIV task, KV tile or KV row counters differ")
    if sum(raw[core][0][3][0] for core in range(cores)) <= 0 or \
            sum(raw[core][0][3][1] for core in range(cores)) <= 0:
        raise HotShapeError("profile recorded no valid tasks or KV work")
    if any(raw[core][engine][3][19] != 0 for core in range(cores) for engine in range(3)):
        raise HotShapeError("profile kernel reported task errors")
    if any(not any(raw[core][engine][3][17] > 0 for core in range(cores)) for engine in range(3)):
        raise HotShapeError("profile kernel did not record a process span on each engine type")
    sources = {}
    for source_index, source_name in enumerate(PROFILE_SOURCES):
        engines = {}
        for engine_index, engine_name in enumerate(PROFILE_ENGINES):
            fields = {}
            for field_index, field_name in enumerate(PROFILE_FIELDS):
                values = [int(raw[core][engine_index][source_index][field_index])
                          for core in range(cores)]
                nonzero = [value for value in values if value > 0]
                unit = "count" if field_name in PROFILE_COUNT_FIELDS else "raw_SYS_CNT"
                item = {"unit": unit, "nonzero_cores": len(nonzero),
                        "max": max(nonzero, default=0),
                        "p50": _nearest_rank(nonzero, 0.50) if nonzero else 0,
                        "p95": _nearest_rank(nonzero, 0.95) if nonzero else 0}
                if unit == "count":
                    item["sum_across_cores"] = sum(values)
                fields[field_name] = item
            engines[engine_name] = fields
        sources[source_name] = engines
    return {"engines": PROFILE_ENGINES, "source_names": PROFILE_SOURCES,
            "field_names": PROFILE_FIELDS,
            "units": "raw SYS_CNT ticks; overlapping cores/lanes never summed as wall time",
            "nested_fields": "aiv_mask_finite and aiv_v2 are subsets of aiv_softmax_total; never add them",
            "field_scope": {
                "aiv_mask_finite": "visibility intervals, batched row finite checks and bitmap Select; excludes score GM-to-UB load, scale and padding",
                "aiv_v2": "includes online-stat copy, device tiling function and SoftmaxFlashV2 call"},
            "sources": sources, "raw": raw}


def _profile_once(torch, ops, tensors: dict, buffers: dict, fixture: dict, *,
                  splits: int, cores: int, scale: float, normal_median_ms: float,
                  tolerance: dict, stream) -> dict:
    """One instrumented invocation after the normal speed and oracle gates."""
    diagnostic = {"tasks": buffers["tasks"],
        "partial": torch.full_like(buffers["partial"], float("nan")),
        "lse": torch.full_like(buffers["lse"], float("nan")),
        "status": torch.full_like(buffers["status"], -99),
        "workspace": torch.empty_like(buffers["workspace"])}
    profile = _aligned_profile_tensor(torch, cores, buffers["partial"].device)
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    begin.record(stream)
    _run_cv(ops, tensors, diagnostic, fixture, splits=splits, cores=cores,
            scale=scale, profile=profile)
    end.record(stream)
    end.synchronize()
    event_ms = float(begin.elapsed_time(end))
    if not math.isfinite(event_ms) or event_ms <= 0:
        raise HotShapeError("profile kernel returned an invalid outer NPU Event duration")
    _check_cv_status(torch, diagnostic)
    torch.testing.assert_close(diagnostic["partial"], buffers["partial"], **tolerance)
    torch.testing.assert_close(diagnostic["lse"], buffers["lse"], **tolerance)
    output_maxdiff = float((diagnostic["partial"] - buffers["partial"]).abs().max())
    finite_lse = torch.isfinite(diagnostic["lse"]) & torch.isfinite(buffers["lse"])
    lse_difference = torch.where(finite_lse,
        (diagnostic["lse"] - buffers["lse"]).abs(), torch.zeros_like(diagnostic["lse"]))
    lse_maxdiff = float(lse_difference.max())
    oracle = _check_oracle(torch, ops, fixture, diagnostic, tolerance)
    raw_summary = summarize_raw_profile(profile.cpu(), cores=cores)
    return {"status": "passed", "profiling_only": True,
            "mode": "instrumented_kernel_diagnostic",
            "normal_profile_frozen_close": True,
            "normal_profile_max_abs": {"partial": output_maxdiff, "lse": lse_maxdiff},
            "frozen_tolerance": tolerance, "sample_oracle": oracle,
            "outer_event_ms": event_ms,
            "outer_event_over_normal_median": event_ms / normal_median_ms,
            "normal_median_ms": normal_median_ms,
            "raw_shape": [cores, 3, 4, len(PROFILE_FIELDS)],
            "rank_scope": "standalone_logical_device0_not_TP_rank",
            "physical_device": 0, "raw_counters": raw_summary}


def _aligned_profile_tensor(torch, cores: int, device):
    """Allocate a 64-byte aligned contiguous diagnostic output view."""
    count = cores * 3 * 4 * len(PROFILE_FIELDS)
    storage = torch.empty((count + 8,), dtype=torch.int64, device=device)
    offset_words = ((-storage.data_ptr()) % 64) // 8
    profile = storage[offset_words:offset_words + count].view(
        cores, 3, 4, len(PROFILE_FIELDS))
    if profile.data_ptr() % 64 or not profile.is_contiguous():
        raise HotShapeError("profile buffer cannot satisfy diagnostic 64-byte alignment")
    profile.zero_()  # Outside the timing Event; C++ owners write disjoint rows.
    return profile


def _check_cv_status(torch, buffers: dict):
    actual = buffers["status"].cpu()
    if not bool((actual == 0).all()):
        bad = int((actual != 0).sum())
        raise HotShapeError(f"CV wrote {bad} nonzero/unwritten task status words")
    if bool(torch.isnan(buffers["lse"]).any()):
        raise HotShapeError("CV left NaN LSE rows")


def _check_oracle(torch, ops, fixture: dict, buffers: dict, tolerance: dict):
    tokens = fixture["tokens"]
    output = torch.empty((tokens * HEADS, DIM), dtype=torch.float32, device=buffers["partial"].device)
    output_lse = torch.empty((tokens * HEADS,), dtype=torch.float32, device=output.device)
    status = torch.full((tokens * HEADS,), -1, dtype=torch.int32, device=output.device)
    ops.merge_lse_out(buffers["partial"].view(tokens * HEADS, -1, DIM),
                      buffers["lse"].view(tokens * HEADS, -1), output, output_lse, status)
    torch.npu.synchronize()
    if not bool((status.cpu() == 0).all()):
        raise HotShapeError("NPU merge did not complete every output row")
    indexes = sorted(fixture["expected"])
    actual_out = output.view(tokens, HEADS, DIM)[indexes].cpu()
    actual_lse = output_lse.view(tokens, HEADS)[indexes].cpu()
    expected_out = torch.stack([fixture["expected"][index][0] for index in indexes])
    expected_lse = torch.stack([fixture["expected"][index][1] for index in indexes])
    torch.testing.assert_close(actual_out, expected_out, **tolerance)
    torch.testing.assert_close(actual_lse, expected_lse, **tolerance)
    return {"sampled_queries": len(indexes),
            "sampled_indices": indexes,
            "max_output_abs": float((actual_out - expected_out).abs().max()),
            "max_lse_abs": float((actual_lse - expected_lse).abs().max()),
            "frozen_tolerance": tolerance, "status": "passed"}


def run_probe(variant: str, manifest_path: Path, target_path: Path,
              acceptance_path: Path = ROOT / "configs/acceptance.json") -> dict:
    if variant not in ("baseline", "candidate"):
        raise HotShapeError("variant must be baseline or candidate")
    target = json.loads(target_path.read_text())
    _select_device(target)
    acceptance = json.loads(acceptance_path.read_text())
    if (acceptance.get("frozen_before_measurement") is not True or
            acceptance.get("performance", {}).get("warmup") != WARMUP or
            acceptance.get("performance", {}).get("repeats") != REPEATS or
            acceptance.get("performance", {}).get("statistic") != "median" or
            not isinstance(acceptance.get("fused_attention"), dict)):
        raise HotShapeError("hot-shape probe requires frozen 2+5 median/event and fused-attention acceptance")
    manifest, configuration, source_digest = _verify_artifact(variant, manifest_path, target)
    from .probe_native_current_fia import _target_geometry
    if _target_geometry(target) != (HEADS, KV_HEADS, DIM):
        raise HotShapeError("target model TP4 head geometry differs from fixed Hq6/Hkv1/D256 fixture")
    import torch
    import torch_npu  # noqa: F401 - enables the real PrivateUse1 backend
    # Isolated child only: bound CPU fixture/oracle overhead and make paired
    # baseline/candidate construction use the same host thread count.
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.npu.set_device(0)
    from .build_ops import normalize_soc
    if normalize_soc(torch.npu.get_device_name(0)) != target["soc_version"]:
        raise HotShapeError("selected device SOC differs from signed target")
    if not torch.npu.is_available():
        raise HotShapeError("real NPU unavailable; CPU execution is forbidden")
    keepalive, _module = _load_ops(variant, manifest, manifest_path)
    _ = keepalive  # Keep baseline kernel dependency loaded for this process.
    ops = torch.ops.oscar_ascend_ops
    device = torch.device("npu:0")
    cores = _actual_cores(torch, target)
    from oscar_ascend.runtime import ATTENTION_QUERY_ROWS, ATTENTION_KV_ROWS
    query_rows = OLD_QUERY_ROWS if variant == "baseline" else ATTENTION_QUERY_ROWS
    capacity = int(target["max_num_batched_tokens"])
    scale = DIM ** -0.5
    tolerance = acceptance["fused_attention"]
    cases = {}
    stream = torch.npu.current_stream()
    for name, make_fixture in (("main", build_fixture), ("decode32", build_decode_fixture)):
        build_started = time.monotonic()
        fixture = make_fixture(torch, scale=scale)
        fixture_build_seconds = time.monotonic() - build_started
        splits = splits_for_shape(fixture["tokens"], cores, query_rows=query_rows, capacity=capacity)
        tensors, buffers = _device_buffers(torch, fixture, device, splits=splits, cores=cores,
                                           current_rows=ATTENTION_QUERY_ROWS,
                                           current_kv_rows=ATTENTION_KV_ROWS)
        task_hash = _prepare_ops(torch, ops, tensors, buffers, fixture, splits=splits)
        durations = []
        for repetition in range(WARMUP + REPEATS):
            buffers["partial"].fill_(float("nan"))
            buffers["lse"].fill_(float("nan"))
            buffers["status"].fill_(-99)
            if repetition >= WARMUP:
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                begin.record(stream)
            _run_cv(ops, tensors, buffers, fixture, splits=splits, cores=cores, scale=scale)
            if repetition >= WARMUP:
                end.record(stream)
                end.synchronize()
                duration = float(begin.elapsed_time(end))
                if not math.isfinite(duration) or duration <= 0:
                    raise HotShapeError(f"invalid NPU Event duration {duration!r} ms")
                durations.append(duration)
            else:
                torch.npu.synchronize()
            _check_cv_status(torch, buffers)
        oracle = _check_oracle(torch, ops, fixture, buffers, tolerance)
        cases[name] = {"fixture_sha256": fixture["hash"], "task_sha256": task_hash,
            "fixture_build_seconds": fixture_build_seconds,
            "query_lengths": fixture["qlens"], "contexts": fixture["contexts"],
            "tokens": fixture["tokens"], "requests": len(fixture["qlens"]),
            "query_heads": HEADS, "kv_heads": KV_HEADS, "head_dim": DIM,
            "physical_pages": fixture["blocks"], "page_policy": fixture["page_policy"],
            "source2_suppressed": fixture["source2_suppressed"],
            "source_splits": splits, "cube_cores": cores,
            "workspace_bytes_per_core": workspace_per_core_bytes(DIM, ATTENTION_QUERY_ROWS, ATTENTION_KV_ROWS),
            "warmup": WARMUP, "repeats": REPEATS, "device_event_ms": durations,
            "median_ms": statistics.median(durations), "sample_oracle": oracle,
            "device_completion": "passed", "timing_evidence": "NPU_Event_attention_cv_out_only"}
        if variant == "candidate":
            cases[name]["profile"] = _profile_once(
                torch, ops, tensors, buffers, fixture, splits=splits, cores=cores,
                scale=scale, normal_median_ms=cases[name]["median_ms"],
                tolerance=tolerance, stream=stream)
        del fixture, tensors, buffers
        gc.collect()
        torch.npu.empty_cache()
    return {"status": "passed", "variant": variant,
            "scope": "operator_only_attention_cv_out; synthetic_main_and_decode32; no_user_dataset",
            "device": str(device), "physical_devices": target["devices"],
            "device_name": torch.npu.get_device_name(0), "device_completion": "passed",
            "fixture_cpu_threads": torch.get_num_threads(),
            "source": {"revision": BASELINE_REVISION if variant == "baseline" else "current_checkout",
                       "sha256": source_digest, "signature": manifest["signature"]},
            "source_sha256": source_digest, "build_signature": manifest["signature"],
            "artifact_sha256": manifest["sha256"], "cases": cases,
            "service_performance": "not_measured"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("baseline", "candidate"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "build/ascendc/build_manifest.json")
    parser.add_argument("--target", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--acceptance", type=Path, default=ROOT / "configs/acceptance.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_probe(args.variant, args.manifest, args.target, args.acceptance)
        atomic_json(args.output, report)
        print("CV_HOTSHAPE " + json.dumps({
            "variant": args.variant, "status": report["status"],
            "signature": report["source"]["signature"],
            "cases": {name: {"fixture_sha256": case["fixture_sha256"],
                             "source_splits": case["source_splits"],
                             "median_ms": case["median_ms"],
                             "sample_oracle": case["sample_oracle"]["status"]}
                      for name, case in report["cases"].items()},
        }, sort_keys=True), flush=True)
        return 0
    except BaselineUnavailable as exc:
        report = {"status": "not_available", "variant": args.variant,
                  "reason": str(exc), "device_completion": "not_run"}
        atomic_json(args.output, report)
        print("CV_HOTSHAPE " + json.dumps(report, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        report = {"status": "failed", "variant": args.variant, "error_type": type(exc).__name__,
                  "error": str(exc), "device_completion": "not_established"}
        atomic_json(args.output, report)
        traceback.print_exc()
        print("CV_HOTSHAPE " + json.dumps(report, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
