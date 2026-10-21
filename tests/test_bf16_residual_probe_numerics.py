"""CPU residual arithmetic on NPU probe fixtures; no C4 atomic/device emulation."""
import json

import pytest
import torch

from oscar_ascend.ops.reference import AttentionResult, decode_kv, merge_attention
from tools import probe_bf16_history as probe
from tools.experiments import bf16_residual_screen as screen
from tools.striped_fixture import striped_slots_to_canonical
from test_probe_bf16_history import task_fixture

ACCEPTANCE = json.loads((probe.ROOT / 'configs/acceptance.json').read_text())


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def residual_partials(fixture, tasks):
    spec = fixture['spec']
    n, h, splits = fixture['tokens'], spec.heads, 3 * spec.splits
    partial = torch.zeros(n, h, splits, 256)
    lse = torch.full((n, h, splits), -torch.inf)
    raw = fixture['cpu']['raw'][probe.reuse.PREFIX:].view(
        fixture['blocks'], probe.reuse.BLOCK_TOKENS, spec.kv_heads, 136)
    rotated = {}
    for request, (_, q, k, *_rest) in fixture['logical'].items():
        # Decode actual serialized fixture bytes through unchanged codec oracle;
        # do not derive candidate operands from original-space expected values.
        pages = fixture['page_assignments'][request]
        slots = raw[list(pages)].reshape(-1, spec.kv_heads, 136)[:len(k)]
        rotated[request] = (decode_kv(striped_slots_to_canonical(slots), 256) if len(k)
                            else (torch.empty(0, spec.kv_heads, 256),
                                  torch.empty(0, spec.kv_heads, 256)))
    for descriptor in tasks.tolist():
        token, count, hk, first, last, request, part, kind, context, begin, error = descriptor[:11]
        assert error == 0
        if count <= 0:
            continue
        heads = slice(hk * 6, (hk + 1) * 6)
        _, q, k, v, ck, cv, _dk, _dv = fixture['logical'][request]
        positions = context + torch.arange(token, token + count) - begin
        cut = torch.maximum(torch.tensor(probe.reuse.SINK), positions + 1 - spec.recent_tokens)
        if kind == 0:
            ids = torch.arange(first, last)
            dk, dv = rotated[request]
            key, value = dk[ids, hk:hk+1], dv[ids, hk:hk+1]
            query = fixture['cpu']['qr'][token:token+count, heads]
            mask = (ids[None, :] >= probe.reuse.SINK) & (ids[None, :] < cut[:, None])
        elif kind == 1:
            tail = min(context, max(probe.reuse.SINK, context + token - begin + 1 - spec.recent_tokens))
            mapping = torch.cat((torch.arange(min(probe.reuse.SINK, context)), torch.arange(tail, context)))
            ids = mapping[first:last]
            key, value = k[ids, hk:hk+1], v[ids, hk:hk+1]
            query = q[token-begin:token-begin+count, heads]
            mask = (ids[None, :] < probe.reuse.SINK) | (ids[None, :] >= cut[:, None])
        else:
            ids = torch.arange(first, last)
            key, value = ck[ids-context, hk:hk+1], cv[ids-context, hk:hk+1]
            query = q[token-begin:token-begin+count, heads]
            mask = torch.ones(count, len(ids), dtype=torch.bool)
        mask &= ids[None, :] <= positions[:, None]
        result = screen.online(query, key, value, mask, 'residual4', history=kind == 0)
        partial[token:token+count, heads, part] = result.output @ fixture['cpu']['rv'].T if kind == 0 else result.output
        lse[token:token+count, heads, part] = result.lse
    return partial, lse


def check_case(case, *, negate=False):
    fixture = probe.make_fixture(torch, case)
    if negate:
        fixture['cpu']['q'].neg_()
        fixture['cpu']['qr'].neg_()
        for request, values in list(fixture['logical'].items()):
            begin, q, *other = values
            fixture['logical'][request] = (begin, -q, *other)
        probe._refresh_dense_oracle(torch, fixture)
    tasks = task_fixture(fixture)
    expected_partial = probe.partial_oracle(torch, fixture, tasks)
    partial, lse = residual_partials(fixture, tasks)
    merged = merge_attention([AttentionResult(partial[:, :, i], lse[:, :, i])
                              for i in range(3 * case.shape.splits)])
    live = fixture['actual_tokens']
    gold = torch.stack([fixture['expected'][i][0] for i in range(live)])
    gold_lse = torch.stack([fixture['expected'][i][1] for i in range(live)])
    checks = {'partial': (partial, expected_partial['partial']),
              'partial_lse': (lse, expected_partial['lse']),
              'output': (merged.output[:live], gold),
              'output_lse': (merged.lse[:live], gold_lse),
              'final_bf16': (merged.output[:live].bfloat16().float(), gold)}
    for name, (actual, expected) in checks.items():
        metric = screen.compare(actual, expected, ACCEPTANCE['fused_attention'],
                                allow_negative_infinity='lse' in name)
        assert metric['passed'], (case.shape.name, negate, name, metric)
    metric = screen.relative_l2(merged.output[:live], gold, ACCEPTANCE['prefill_window']['relative_l2_lt'])
    assert metric['passed'], (case.shape.name, negate, metric)
    assert torch.equal(torch.isneginf(lse), torch.isneginf(expected_partial['lse']))
    assert not bool(partial[torch.isneginf(lse)].any())


@pytest.mark.parametrize('case', probe.case_plan()[:12], ids=lambda c: c.shape.name)
def test_npu_fixture_cpu_residual_arithmetic(case):
    check_case(case)


def test_q1_changed_q_qr_negative_query_state_cpu_arithmetic():
    check_case(probe.case_plan()[1], negate=True)
