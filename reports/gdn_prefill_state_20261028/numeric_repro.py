"""Archive #163/#174: finite numerical witness for warm-prefill state misuse.

Execute the pinned native fold, pure-torch recurrent golden, and actual native
prefill SSM assignment. The conv witness is source-bound CPU convolution
mathematics, not an execution of the AscendC conv kernel. No model or NPU run,
no INT2 arithmetic, no native source changes and no performance measurement.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch

ROOT = Path(__file__).resolve().parents[2]
BUILDER = "references/vllm-ascend/vllm_ascend/ops/gdn_attn_builder.py"
GDN = "references/vllm-ascend/vllm_ascend/ops/gdn.py"
GOLDEN = "references/vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops/test_recurrent_gated_delta_rule.py"
RECURRENT = "references/vllm-ascend/csrc/attention/recurrent_gated_delta_rule/op_kernel/recurrent_gated_delta_rule.h"
CONV = "references/vllm-ascend/csrc/moe/causal_conv1d/op_kernel/causal_conv1d.h"


def _native_function(path, name):
    tree = ast.parse((ROOT / path).read_text())
    function = copy.deepcopy(next(node for node in ast.walk(tree)
                                  if isinstance(node, ast.FunctionDef) and node.name == name))
    # Drop import-time annotations only; executable function statements remain.
    function.returns = None
    for arg in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs):
        arg.annotation = None
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 path, "exec"), namespace)
    return namespace[name]


def _native_prefill_store(states, indices, final_state):
    tree = ast.parse((ROOT / GDN).read_text())
    assignment = next(node for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name) and target.value.id == "ssm_state"
            and isinstance(target.slice, ast.Name) and target.slice.id == "prefill_state_indices"
            for target in node.targets))
    namespace = {"ssm_state": states, "prefill_state_indices": indices,
                 "last_recurrent_state": final_state}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), GDN, "exec"), namespace)


def reproduce(stale_value=-42.0):
    torch.set_num_threads(1)
    fold = _native_function(BUILDER, "_fold_spec_sized_prefill_chunks_into_spec")
    golden = _native_function(GOLDEN, "golden_recurrent_gated_delta_rule")
    common = NS(is_prefilling=torch.tensor([True]),
                query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
                seq_lens_cpu_upper_bound=torch.tensor([5000], dtype=torch.int32))
    original_mask = torch.tensor([False])
    original_count = torch.tensor([1], dtype=torch.int32)
    folded_mask, folded_count = fold(NS(num_spec=3), common, original_mask, original_count)
    assert folded_mask.tolist() == [True] and folded_count.tolist() == [4]

    # These are the actual table identities from the paired align lifecycle
    # replay at computed4996 -> scheduled5000. No state copy occurs there.
    table = torch.tensor([7, 8, 9, 10], dtype=torch.int32)
    dim = 128
    states = torch.zeros(11, 1, dim, dim, dtype=torch.bfloat16)
    states[7, :, :, 0] = .25
    states[10, :, :, 0] = stale_value
    query = torch.zeros(4, 1, dim, dtype=torch.bfloat16)
    query[:, :, 0] = 1  # Exact unit vectors; native Q/K l2norm changes nothing.
    key = query.clone()
    value = torch.ones_like(query)
    beta = torch.full((4, 1), .5, dtype=torch.bfloat16)
    gate = torch.zeros(4, 1, dtype=torch.float32)
    scale = dim ** -.5

    recurrent_source = (ROOT / RECURRENT).read_text()
    assert "stateTokenIdx = seq0 + acceptedTokenNum - 1" in recurrent_source
    assert "stateOffset = ssmStateIndicesGm_.GetValue(stateTokenIdx)" in recurrent_source
    assert "ssmStateIndicesGm_.GetValue(seq_i)" in recurrent_source

    old, old_states = golden(query, key, value, states, beta, scale, [4], table, gate, folded_count)
    current_correct, selector_only_states = golden(
        query, key, value, states, beta, scale, [4], table, gate, original_count)
    # Independent closed form for these unit-vector inputs: every active state
    # coordinate satisfies s_n = (s_(n-1) + 1) / 2, with s_0=.25.
    exact_states = 1.0 - .75 * .5 ** torch.arange(1, 5, dtype=torch.float32)
    expected_output = (exact_states * scale).bfloat16()[:, None, None].expand_as(current_correct)
    assert torch.equal(current_correct, expected_output)
    # The sequential golden computes the final state; the actual prefill
    # assignment commits that final state to column0, not speculative column3.
    committed = states.clone()
    _native_prefill_store(committed, table[:1].long(), selector_only_states[table[-1:]].clone())

    def next_output(current_states):
        return golden(query[:1], key[:1], value[:1], current_states, beta[:1], scale,
                      [1], table[:1], gate[:1], original_count)[0]

    next_correct = next_output(committed)
    next_selector_only = next_output(selector_only_states)
    next_old = next_output(old_states)
    assert torch.isfinite(old).all() and torch.isfinite(current_correct).all()
    assert not torch.equal(old, current_correct)
    assert not torch.equal(next_correct, next_selector_only)

    conv_source = (ROOT / CONV).read_text()
    prefill_store = conv_source.split("CAUSAL_CONV1D_CLASS::WriteBackState(int32_t", 1)[1].split(
        "CAUSAL_CONV1D_CLASS::WriteBackStateSpec", 1)[0]
    assert "pos < (width - 1)" in prefill_store
    assert "stateBaseOffset + static_cast<int64_t>(pos) * dim" in prefill_store
    assert "stateTokenOffset = accepted - 1" in conv_source
    assert "const int32_t srcPos0 = stateTokenOffset + 1" in conv_source
    assert "const int32_t srcPos1 = stateTokenOffset + 2" in conv_source
    conv_state = torch.full((6,), stale_value, dtype=torch.float32)
    conv_state[:3] = .25
    current = torch.tensor([1., 2., 3., 4.])

    def conv(history, values):
        sequence = torch.cat((history, values))
        return torch.nn.functional.conv1d(sequence.view(1, 1, -1),
                                          torch.ones(1, 1, 4)).view(-1)

    correct_conv = conv(conv_state[:3], current)
    old_conv = conv(conv_state[3:], current)
    # accepted1 alone writes [old history tail2, four new tokens] in spec mode;
    # next accepted1 reads its first3, whereas prefill commits last3 new tokens.
    selector_conv_state = torch.cat((conv_state[1:3], current))
    correct_conv_state = current[-3:]
    correct_conv_next = conv(correct_conv_state, torch.tensor([5.]))
    selector_conv_next = conv(selector_conv_state[:3], torch.tensor([5.]))
    assert not torch.equal(old_conv, correct_conv)
    assert not torch.equal(correct_conv_next, selector_conv_next)

    return {
        "scope": "Pinned native CPU golden and AST contracts; finite synthetic values; no model, NPU or performance evidence",
        "alignment_lifecycle_reference": "align_repro.py: prompt5000 -> 4996+4, same table[7,8,9,10], no preprocess copy",
        "initial_committed_state": .25, "initial_stale_state": stale_value,
        "original_accepted": original_count.tolist(), "folded_accepted": folded_count.tolist(),
        "committed_state_column": 0, "incorrect_initial_state_column": 3,
        "gdn_scale": scale,
        "gdn_old_output": old[:, 0, 0].tolist(),
        "gdn_correct_output": current_correct[:, 0, 0].tolist(),
        "gdn_selector_only_current_output_equals_correct": torch.equal(current_correct, expected_output),
        "independent_correct_state_sequence": exact_states.tolist(),
        "gdn_correct_final_col0": float(committed[7, 0, 0, 0]),
        "gdn_selector_only_col0": float(selector_only_states[7, 0, 0, 0]),
        "gdn_selector_only_col3": float(selector_only_states[10, 0, 0, 0]),
        "next_gdn_old": float(next_old[0, 0, 0]),
        "next_gdn_selector_only": float(next_selector_only[0, 0, 0]),
        "next_gdn_correct": float(next_correct[0, 0, 0]),
        "conv_width": 4, "conv_state_len": 6,
        "conv_old_pre_activation": old_conv.tolist(),
        "conv_correct_pre_activation": correct_conv.tolist(),
        "conv_old_silu": torch.nn.functional.silu(old_conv).tolist(),
        "conv_correct_silu": torch.nn.functional.silu(correct_conv).tolist(),
        "next_conv_selector_only": float(selector_conv_next[0]),
        "next_conv_correct": float(correct_conv_next[0]),
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                          for path in (BUILDER, GDN, GOLDEN, RECURRENT, CONV)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    result = reproduce()
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
