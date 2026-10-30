"""Archive #170/#172/#175/#176: bounded staging preserves the OptR objective.

CPU/meta checks cover tensor ownership and arithmetic, not NPU peak or speed.
The independent official loss/gradient tests remain in upstream_parity.
"""
import weakref

import pytest
import torch

from oscar_ascend import optr_shared_cache as cache
from oscar_ascend import optr_sequential as sequential
from test_optr_calibration import trace
from test_optr_upstream_parity import options


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def capture():
    splits = []
    for split in ("train", "held"):
        heads = []
        for head in range(2):
            rows = []
            for request, (length, queries) in enumerate(((81, 7), (93, 5))):
                sample = trace(83 + head * 7 + request, f"{split}_{request}",
                               length=length, queries=queries, group=3)
                # Include prefill, several subsequent forward boundaries, and
                # differing causal tails within the same captured request.
                positions = [2, 8, 11, 30, 42, 60, 80] if queries == 7 else [2, 11, 30, 60, 92]
                contexts = [0, 0, 8, 16, 40, 40, 75] if queries == 7 else [0, 8, 16, 40, 85]
                sample["positions"] = torch.tensor(positions)
                sample["contexts"] = torch.tensor(contexts)
                sample["wo"].mul_(1e-3)
                if rows:
                    sample["wo"] = rows[0]["wo"]
                if splits:
                    sample["wo"] = splits[0][head][0]["wo"]
                rows.append(sample)
            heads.append(rows)
        splits.append(heads)
    return *splits, torch.zeros(2, 8), options()


def prepare(train, held, mean, policy):
    return cache.prepare_shared_rows(train, held, mean, policy, "cpu", lambda *a, **kw: None)


def force_stream(monkeypatch, train, held):
    monkeypatch.setattr(cache, "RESIDENT_CACHE_BYTES", 0)
    # Every input has at least two chunks, with nonuniform final chunk sizes.
    budget = max(cache.sample_cache_bytes(s, queries=2)
                 for heads in (train, held) for samples in heads for s in samples)
    monkeypatch.setattr(cache, "STREAMED_ROW_CACHE_BYTES", budget)


def loss_and_backward(rows, rotation, phase, policy):
    total = 0.
    for row, weight in rows:
        loss = sequential.phase_loss(rotation, row, phase, policy)
        total += float(loss.detach()) * weight
        (loss * weight).backward()
        del row, loss
    return total


def test_fixed_byte_estimate_counts_real_staged_tensors_and_shared_weights_once():
    train, held, mean, policy = capture()
    training, validation, record = prepare(train, held, mean, policy)
    tensors = {id(value): value for rows in (training, validation)
               for row, _ in rows for value in row.values() if isinstance(value, torch.Tensor)}
    actual = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    assert actual == record["estimated_resident_fixed_bytes"]
    assert record["mode"] == "resident"
    for row, _ in training:
        sample = next(s for s in train[0] if s["request_id"] == row["request_id"])
        assert cache.sample_cache_bytes(sample) == sum(
            v.numel() * v.element_size() for v in row.values() if isinstance(v, torch.Tensor))


def test_resident_stages_once_and_caches_fixed_k_once_without_new_transfers(monkeypatch):
    train, held, mean, policy = capture()
    preparations, probabilities = [], []
    original_prepare, original_key = cache.prepare_samples, cache._key_probabilities

    def stage(samples, *args, **kwargs):
        preparations.append(samples)
        return original_prepare(samples, *args, **kwargs)

    def key(*args):
        probabilities.append(1)
        return original_key(*args)

    monkeypatch.setattr(cache, "prepare_samples", stage)
    monkeypatch.setattr(cache, "_key_probabilities", key)
    training, validation, record = prepare(train, held, mean, policy)
    assert preparations == [*train, *held]
    assert len(preparations) == 4  # Unchanged head-wise prepare_samples calls.
    assert not record["extra_transfers_per_iteration"]
    assert not record["teacher_recomputed_per_iteration"]
    for _ in range(2):
        for row, _ in training:
            sequential.phase_loss(torch.eye(8), row, "k", policy)
    for rows in (training, validation):
        rows.prepare_value(torch.eye(8))
        for row, _ in rows:
            assert not {"q", "k", "exact", "causal", "teacher_p", "teacher_logp", "teacher_logits"} & row.keys()
        for _ in range(2):
            for row, _ in rows:
                sequential.phase_loss(torch.eye(8), row, "v", policy)
    assert len(preparations) == 4
    assert len(probabilities) == len(training) + len(validation)
    assert training.rows[0][0]["wo_t"] is validation.rows[0][0]["wo_t"]
    assert training.rows[0][0]["wo_t"] is training.rows[1][0]["wo_t"]


@pytest.mark.parametrize("phase", ["k", "v"])
def test_query_chunks_preserve_losses_gradients_and_adam_with_real_frame_masks(monkeypatch, phase):
    train, held, mean, policy = capture()
    resident, _, _ = prepare(train, held, mean, policy)
    force_stream(monkeypatch, train, held)
    streamed, _, record = prepare(train, held, mean, policy)
    assert len(streamed) > len(resident)
    assert record["largest_staged_fixed_row_bytes"] <= cache.STREAMED_ROW_CACHE_BYTES
    if phase == "v":
        rotation = torch.linalg.qr(torch.randn(8, 8, generator=torch.Generator().manual_seed(9))).Q
        resident.prepare_value(rotation)
        streamed.prepare_value(rotation)
    parameters = [torch.eye(8, requires_grad=True) for _ in range(2)]
    optimizers = [torch.optim.Adam([p], lr=policy["lr"]) for p in parameters]
    for _ in range(3):
        losses = []
        for rows, parameter, optimizer in zip((resident, streamed), parameters, optimizers, strict=True):
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss_and_backward(rows, parameter, phase, policy))
        assert losses[1] == pytest.approx(losses[0], rel=2e-5, abs=2e-8)
        torch.testing.assert_close(parameters[1].grad, parameters[0].grad, rtol=3e-5, atol=2e-7)
        for optimizer in optimizers:
            optimizer.step()
        torch.testing.assert_close(parameters[1], parameters[0], rtol=3e-5, atol=2e-6)


def test_streamed_rows_preserve_complete_history_query_order_masks_and_weights(monkeypatch):
    train, held, mean, policy = capture()
    resident, _, _ = prepare(train, held, mean, policy)
    force_stream(monkeypatch, train, held)
    streamed, _, _ = prepare(train, held, mean, policy)
    offset, row_index, weight_sum = 0, 0, 0.
    for row, weight in streamed:
        expected, original_weight = resident.rows[row_index]
        count = row["q"].shape[1]
        for key in ("q", "teacher_p", "teacher_logp", "teacher_logits"):
            torch.testing.assert_close(row[key], expected[key][:, offset:offset + count], rtol=2e-6, atol=2e-6)
        for key in ("exact", "causal"):
            assert torch.equal(row[key], expected[key][offset:offset + count])
        for key in ("k", "v", "wo_t"):
            assert torch.equal(row[key], expected[key])
        assert row["request_id"] == expected["request_id"]
        offset += count
        weight_sum += weight
        if offset == expected["q"].shape[1]:
            assert weight_sum == pytest.approx(original_weight)
            offset, weight_sum = 0, 0.
            row_index += 1
        del row
    assert row_index == len(resident)


def test_streamed_previous_device_row_is_released_before_next_staging(monkeypatch):
    train, held, mean, policy = capture()
    force_stream(monkeypatch, train, held)
    references, original = [], cache.prepare_samples

    def stage(*args, **kwargs):
        assert all(reference() is None for reference in references)
        rows = original(*args, **kwargs)
        references[:] = [weakref.ref(rows[0][key]) for key in ("q", "k", "v", "teacher_p")]
        return rows

    monkeypatch.setattr(cache, "prepare_samples", stage)
    training, _, _ = prepare(train, held, mean, policy)
    for _ in range(2):
        parameter = torch.eye(8, requires_grad=True)
        loss_and_backward(training, parameter, "k", policy)
        assert all(reference() is None for reference in references)
    assert len(training.weights) == 1


def test_streamed_v_freezes_selected_k_and_releases_k_side_per_chunk(monkeypatch):
    train, held, mean, policy = capture()
    resident, _, _ = prepare(train, held, mean, policy)
    force_stream(monkeypatch, train, held)
    streamed, _, _ = prepare(train, held, mean, policy)
    selected = torch.eye(8, requires_grad=True)
    resident.prepare_value(selected)
    streamed.prepare_value(selected)
    with torch.no_grad():
        selected.mul_(2)  # The prepared provider owns its frozen snapshot.
    expected = [row["selected_k_quant_p"] for row, _ in resident]
    offset, row_index = 0, 0
    rv = torch.eye(8, requires_grad=True)
    for row, weight in streamed:
        assert not {"q", "k", "teacher_p", "teacher_logp", "teacher_logits", "exact", "causal"} & row.keys()
        probability = row["selected_k_quant_p"]
        assert not probability.requires_grad
        count = probability.shape[1]
        torch.testing.assert_close(probability, expected[row_index][:, offset:offset + count], rtol=2e-6, atol=2e-6)
        loss = sequential.phase_loss(rv, row, "v", policy) * weight
        loss.backward()
        offset += count
        if offset == expected[row_index].shape[1]:
            row_index += 1
            offset = 0
        del row, probability, loss
    assert selected.grad is None
    assert rv.grad is not None and bool(torch.isfinite(rv.grad).all())
    with pytest.raises(ValueError, match="already prepared"):
        streamed.prepare_value(selected)


def shaped_capture(lengths, queries_per_frame, decode_tokens):
    splits = []
    weights = [torch.empty((6, 5120, 256), device="meta") for _ in range(4)]
    for split in ("train", "held"):
        heads = []
        for head in range(4):
            samples = []
            for length in lengths:
                count = ((length + 16383) // 16384) * queries_per_frame + decode_tokens
                samples.append({"request_id": f"{split}_{length}",
                    "q": torch.empty((count, 6, 256), dtype=torch.bfloat16, device="meta"),
                    "k": torch.empty((length + decode_tokens, 256), dtype=torch.bfloat16, device="meta"),
                    "v": torch.empty((length + decode_tokens, 256), dtype=torch.bfloat16, device="meta"),
                    "wo": weights[head]})
            heads.append(samples)
        splits.append(heads)
    return splits


def test_default_capture_stays_resident_without_query_splitting_or_restaging(monkeypatch):
    train, held = shaped_capture([4096, 32768, 65536], 8, 8)
    calls = []

    def no_allocation(samples, *args, **kwargs):
        calls.append(samples)
        return [{"request_id": s["request_id"]} for s in samples]

    monkeypatch.setattr(cache, "prepare_samples", no_allocation)
    training, validation, record = prepare(train, held, torch.zeros(4, 256), options())
    assert record["mode"] == "resident"
    assert 3 * 1024 ** 3 < record["estimated_resident_fixed_bytes"] < 4 * 1024 ** 3
    assert len(training) == len(validation) == 12
    assert len(calls) == 8
    for _ in range(3):
        assert len(list(training)) == len(list(validation)) == 12
    assert len(calls) == 8


def test_legal_large_capture_is_planned_under_bound_without_device_allocation(monkeypatch):
    train, held = shaped_capture([65533, 65534, 65535, 65536], 64, 64)
    monkeypatch.setattr(cache, "prepare_samples", lambda *a, **kw: pytest.fail("preflight must not stage"))
    training, validation, record = prepare(train, held, torch.zeros(4, 256), options())
    assert record["mode"] == "streamed"
    assert record["estimated_resident_fixed_bytes"] > 50 * 1024 ** 3
    assert record["largest_staged_fixed_row_bytes"] <= 512 * 1024 ** 2
    assert len(training) > 16 and len(validation) > 16
    assert record["extra_transfers_per_iteration"]
    assert record["selected_k_probabilities_recomputed_per_v_iteration"]


def test_streamed_preflight_rejects_single_unfittable_complete_history_before_staging(monkeypatch):
    train, held, mean, policy = capture()
    monkeypatch.setattr(cache, "RESIDENT_CACHE_BYTES", 0)
    monkeypatch.setattr(cache, "STREAMED_ROW_CACHE_BYTES", cache.sample_cache_bytes(train[0][0], queries=0))
    monkeypatch.setattr(cache, "prepare_samples", lambda *a, **kw: pytest.fail("preflight must not stage"))
    with pytest.raises(ValueError, match="complete K/V/W_O plus one query.*histories are never truncated"):
        prepare(train, held, mean, policy)
