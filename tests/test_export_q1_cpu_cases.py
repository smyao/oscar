"""Q1 CPU-debug goldens preserve distinct histories and padded task geometry.

Archive #126/#129/#148-150 and startup D.4: these offline checks do not
establish target NPU numerical completion or performance.
"""

import json

import torch

from tools.export_q1_cpu_cases import export_q1_cpu_cases


def _tensor(path, dtype, shape):
    data = bytearray(path.read_bytes())
    return torch.frombuffer(data, dtype=dtype).reshape(shape)


def test_q1_goldens_have_distinct_physical_histories_and_exact_padding(tmp_path):
    root = export_q1_cpu_cases(tmp_path / "goldens")
    cases = json.loads((root / "cases.json").read_text())
    assert [entry["op"] for entry in cases] == [
        "q1_schedule", "q1_schedule_bad_meta", "q1_schedule",
        "q1_schedule", "q1_schedule"]

    for entry in cases:
        if entry["op"] != "q1_schedule":
            continue
        case = root / entry["path"].split("/")[-1]
        n, hq, hk, dim, context_hint, sink, recent, speculative, block_tokens, blocks, \
            prefix, stride, cores, requests, splits = map(int, (case / "shape.txt").read_text().split())
        assert hq == 6 and hk == 1 and context_hint == 0
        assert (sink, recent, speculative, block_tokens, prefix) == (4, 32, 3, 512, 64)
        assert stride == 512 * (dim // 2 + 8) and blocks == 2 * requests
        assert len((case / "raw.bin").read_bytes()) == prefix + blocks * stride
        assert len((case / "wk.bin").read_bytes()) == blocks * 39 * dim * 2
        assert len((case / "q.bin").read_bytes()) == n * hq * dim * 2
        starts = _tensor(case / "starts.bin", torch.int32, (requests + 1,))
        lens = _tensor(case / "lens.bin", torch.int32, (requests,))
        slots = _tensor(case / "slots.bin", torch.int64, (n,))
        positions = _tensor(case / "expected_positions.bin", torch.int64, (n,))
        table = _tensor(case / "table.bin", torch.int32, (requests, 8))
        output = _tensor(case / "expected_output.bin", torch.float32, (n, hq, dim))
        lse = _tensor(case / "expected_lse.bin", torch.float32, (n, hq))
        contexts = lens.to(torch.int64) - 1
        assert torch.equal(starts, torch.arange(requests + 1, dtype=torch.int32))
        assert len(set(contexts.tolist())) == requests
        assert torch.equal(positions[:requests], contexts)
        assert torch.equal(slots[:requests] % block_tokens, contexts)
        assert torch.equal(slots[:requests] // block_tokens,
                           2 * torch.arange(requests) + 1)
        assert torch.equal(table[:, 0], 4 * (2 * torch.arange(requests) + 1))
        assert bool((slots[requests:] == -1).all())
        assert bool((positions[requests:] == -1).all())
        assert bool((output[requests:] == 0).all())
        assert bool(torch.isneginf(lse[requests:]).all())
        assert bool(torch.isfinite(output[:requests]).all())
        assert bool(torch.isfinite(lse[:requests]).all())
        if "empty" in case.name:
            assert int(contexts[0]) == 0 and bool((contexts[1:] > 0).all())
        else:
            assert bool((contexts > 0).all())
        if case.name.startswith("q1_32"):
            assert (n, requests, cores, dim) == (128, 32, 20, 64)
            assert splits == (1 if "_s1_" in case.name else 3)
        else:
            assert (n, requests, cores, dim, splits) == (8, 4, 2, 256, 1)


def test_q1_bad_metadata_offset_hits_live_history_scale(tmp_path):
    root = export_q1_cpu_cases(tmp_path / "goldens")
    case = root / "q1_32_s3_d64"
    raw = (case / "raw.bin").read_bytes()
    offset = int((case / "bad_meta_offset.txt").read_text())
    assert offset == 64 + 20480 + 20 * 40 + 16
    assert raw[offset:offset + 2] != b"\x00\x00"
    assert offset + 2 < len(raw)
    lens = _tensor(case / "lens.bin", torch.int32, (32,))
    context = int(lens[0]) - 1
    assert 4 <= 20 < context + 1 - 32  # compressed live history, not sink/window
