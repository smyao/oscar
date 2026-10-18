"""Archive #26/#37-49/#127/#128: native APC ownership and OSCAR exact pages.

Execute pinned native coordinator/managers/pool unchanged, including MTP's
lookahead drop, native hash aggregation, refcounts, eviction and reset. This
is host evidence; the byte checks do not stand in for an NPU attention run.
"""
import json
import logging
from types import SimpleNamespace

import pytest
import torch

from tools import install_serve, observe_serve
from tools.serving_variants import variant_config, runtime_feature_summary
from tools.target_cli import serve_argv
from oscar_ascend.lifecycle import SnapshotLayout
from test_runtime_native import (definitions, native_specs, native_groups,
                                _native_coordinator)


@pytest.mark.parametrize("variant", ["native", "baseline", "candidate"])
@pytest.mark.parametrize("prefix", [None, False, True])
def test_explicit_prefix_policy_reaches_native_argv_without_budget_override(variant, prefix):
    config = json.loads((install_serve.ROOT / "configs/target.json").read_text())
    config.pop("enable_prefix_caching", None)
    if prefix is not None:
        config["enable_prefix_caching"] = prefix
    selected = variant_config(config, variant)
    argv = serve_argv(selected)
    assert ("--enable-prefix-caching" in argv) is (prefix is True)
    assert ("--no-enable-prefix-caching" in argv) is (prefix is not True)
    assert argv[argv.index("--max-num-batched-tokens") + 1] == "16384"
    assert "--scheduler-cls" not in argv
    assert "--mamba-cache-mode" not in argv  # Native model config derives align.


def test_raw_service_callers_keep_stable_prefix_off_without_variant_normalization():
    config = json.loads((install_serve.ROOT / "configs/target.json").read_text())
    assert config["enable_prefix_caching"] is False
    config.pop("enable_prefix_caching")
    assert "--no-enable-prefix-caching" in serve_argv(config)
    assert runtime_feature_summary(config, None)["prefix_caching"] is False


@pytest.mark.parametrize("entry", [install_serve, observe_serve])
def test_both_entrypoints_keep_explicit_stable_prefix_off(entry, tmp_path, capsys):
    log_dir = tmp_path / "not_created"
    assert entry.main(["--variant", "candidate", "--plan", "--log-dir", str(log_dir)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["runtime_selection"]["prefix_caching"] is False
    assert not log_dir.exists()


@pytest.mark.parametrize("value", ["true", "false", 1, 0, None])
def test_prefix_flag_cannot_be_silently_coerced(value):
    config = json.loads((install_serve.ROOT / "configs/target.json").read_text())
    config["enable_prefix_caching"] = value
    with pytest.raises(ValueError, match="explicit boolean"):
        variant_config(config, "candidate")
    with pytest.raises(ValueError, match="explicit boolean"):
        serve_argv(config)


@pytest.mark.parametrize("entry", [install_serve, observe_serve])
@pytest.mark.parametrize("enabled", [True, False])
def test_both_candidate_entrypoints_expose_the_same_native_switch(entry, enabled, tmp_path, capsys):
    log_dir = tmp_path / "not_created"
    switch = "--enable-prefix-caching" if enabled else "--no-enable-prefix-caching"
    assert entry.main(["--variant", "candidate", "--plan", "--log-dir", str(log_dir), switch]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["runtime_selection"]["prefix_caching"] is enabled
    assert not log_dir.exists()


def _coordinator_with_hashing(native_groups, monkeypatch):
    instance = _native_coordinator(native_groups, monkeypatch)
    utils = native_groups[3]
    helpers = definitions(
        "references/vllm/vllm/v1/core/kv_cache_utils.py", "native_prefix_hash_contract",
        {"make_block_hash_with_group_id", "get_block_hash", "get_group_id"},
        {"BlockHash": bytes, "BlockHashWithGroupId": bytes}, monkeypatch)
    pool_globals = type(instance.block_pool).__init__.__globals__
    pool_globals.update(
        make_block_hash_with_group_id=helpers.make_block_hash_with_group_id,
        BlockHashListWithBlockSize=utils.BlockHashListWithBlockSize,
        logger=logging.getLogger("native-prefix-test"))
    instance.find_longest_cache_hit.__func__.__globals__["BlockHashListWithBlockSize"] = utils.BlockHashListWithBlockSize
    return instance


def _cache_seed(instance, *, request_id="seed"):
    full = instance.single_type_managers[0]
    total = 4 * full.block_size
    hashes = [i.to_bytes(4, "big") for i in range(total // instance.hash_block_size)]
    req = SimpleNamespace(request_id=request_id, block_hashes=hashes, num_prompt_tokens=total)
    # Native align checkpoints remain 768 tokens even though FULL pages are
    # 2304. Do not change GDN or inflate the native per-step token budget.
    for count in range(instance.hash_block_size, total + 1, instance.hash_block_size):
        for manager in instance.single_type_managers:
            manager.new_step_starts()
            manager.allocate_new_blocks(request_id, count + 3, count)
            manager.cache_blocks(req, count)
    return req, total


def test_native_hybrid_prefix_hit_forks_refcounts_and_reset(native_groups, monkeypatch):
    instance = _coordinator_with_hashing(native_groups, monkeypatch)
    req, total = _cache_seed(instance)
    full, *gdn = instance.single_type_managers
    assert full.block_size == 2304
    assert all(manager.block_size == 768 and manager.mamba_cache_mode == "align" for manager in gdn)
    seed_ids = [block.block_id for block in full.req_to_blocks[req.request_id]]
    for manager in instance.single_type_managers:
        manager.free(req.request_id)
    computed, hit = instance.find_longest_cache_hit(req.block_hashes, total)
    assert hit == total - 2304  # Actual FullAttentionManager MTP last-block drop.
    assert [block.block_id for block in computed[0]] == seed_ids[:3]
    assert all(len(group) * manager.block_size == hit
               for group, manager in zip(computed, instance.single_type_managers))
    for fork in ("left", "right"):
        for manager, blocks in zip(instance.single_type_managers, computed):
            manager.allocate_new_computed_blocks(fork, blocks, hit, 0)
            manager.allocate_new_blocks(fork, hit + 4, hit + 1)
    left, right = (full.req_to_blocks[name] for name in ("left", "right"))
    assert left[:3] == right[:3] and all(block.ref_cnt == 2 for block in left[:3])
    assert left[3].block_id != right[3].block_id
    assert left[3].ref_cnt == right[3].ref_cnt == 1
    assert not instance.block_pool.reset_prefix_cache()
    for manager in instance.single_type_managers:
        manager.free("left")  # same native free used on preemption/abort
    assert all(block.ref_cnt == 1 for block in right[:3])
    for manager in instance.single_type_managers:
        manager.free("right")
    assert instance.block_pool.reset_prefix_cache()
    assert instance.find_longest_cache_hit(req.block_hashes, total)[1] == 0


def test_partial_prefix_and_evicted_page_cannot_reappear(native_groups, monkeypatch):
    instance = _coordinator_with_hashing(native_groups, monkeypatch)
    req, total = _cache_seed(instance)
    full = instance.single_type_managers[0]
    for manager in instance.single_type_managers:
        manager.free(req.request_id)
    _, full_hit = instance.find_longest_cache_hit(req.block_hashes, total)
    changed = req.block_hashes[:6] + [b"different"] * (len(req.block_hashes) - 6)
    _, partial_hit = instance.find_longest_cache_hit(changed, total)
    assert partial_hit == full.block_size and 0 < partial_hit < full_hit
    _, bounded_hit = instance.find_longest_cache_hit(req.block_hashes, total - 1)
    assert bounded_hit == 2 * full.block_size  # prompt's final token is recomputed
    computed, _ = instance.find_longest_cache_hit(req.block_hashes, total)
    victim = computed[0][0]
    instance.block_pool.evict_blocks({victim.block_id})
    assert victim.block_hash is None
    assert instance.find_longest_cache_hit(req.block_hashes, total)[1] == 0
    # Exhaust free FIFO once: every old ID can be reused, but old hashes stay
    # evicted. This tests allocator identity, not a guessed request-row owner.
    recycled = instance.block_pool.get_new_blocks(instance.block_pool.get_num_free_blocks())
    assert victim in recycled
    assert instance.find_longest_cache_hit(req.block_hashes, total)[1] == 0
    instance.block_pool.free_blocks(recycled)


def test_shared_prefix_exact_snapshot_survives_both_forks(native_groups, monkeypatch):
    instance = _coordinator_with_hashing(native_groups, monkeypatch)
    req, total = _cache_seed(instance)
    full = instance.single_type_managers[0]
    for manager in instance.single_type_managers:
        manager.free(req.request_id)
    computed, hit = instance.find_longest_cache_hit(req.block_hashes, total)
    for fork in ("left", "right"):
        full.allocate_new_computed_blocks(fork, computed[0], hit, 0)
        full.allocate_new_blocks(fork, hit + 4, hit + 1)
    physical = [block.block_id for block in computed[0]]
    tails = [full.req_to_blocks[name][-1].block_id for name in ("left", "right")]
    count = max(physical + tails) + 1
    snap = SnapshotLayout(full.kv_cache_spec.layout, 64, 256, 3)
    raw = torch.full((count * snap.page.page_size_bytes,), 0x55, dtype=torch.uint8)
    keys, values, tags = snap.views(raw, count)
    for logical_page, page in enumerate(physical):
        for pos in range(logical_page * 2304, (logical_page + 1) * 2304):
            row = snap.row_for_position(pos)
            keys[page, row].fill_(pos % 127)
            values[page, row].fill_(-(pos % 127))
            tags[page, row] = pos % 2304
    immutable = [(page, raw[slice(*snap.interval(count, page))].clone()) for page in physical]
    for fork_index, page in enumerate(tails):
        for pos in range(hit, hit + 4):
            row = snap.row_for_position(pos)
            keys[page, row].fill_(20 + fork_index)
            values[page, row].fill_(-20 - fork_index)
            tags[page, row] = pos % 2304
    for page, expected in immutable:
        assert torch.equal(raw[slice(*snap.interval(count, page))], expected)
    for pos in list(range(64)) + list(range(hit - 256, hit)):
        page = physical[pos // 2304]
        row = snap.row_for_position(pos)
        assert tags[page, row] == pos % 2304
        assert torch.all(keys[page, row] == pos % 127)
        assert torch.all(values[page, row] == -(pos % 127))
