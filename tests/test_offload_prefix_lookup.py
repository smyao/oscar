"""Archive #127/#128/#157/#163/#171/#182/#183: bounded complete-prefix queries.

Count host calls rather than timing them. Native salted hashes, all-TP state,
eviction races and explicit corruption/errors remain part of the contract.
"""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from oscar_ascend.offload.plans import expected_keys, prefix_identity, prefix_plan
from oscar_ascend.offload.transport import TransferError
from test_offload_native_bridge import bridge_module, connector_pair, scheduler_for, scheduled


class Hashes(list):
    reads = 0

    def __getitem__(self, index):
        self.reads += 1
        return super().__getitem__(index)


class Metadata:
    def __init__(self):
        self.manifests = {}
        self.probes = []
        self.reads = []

    def exists(self, keys):
        self.probes.append(tuple(keys))
        assert len(keys) <= 256 and len(set(keys)) == len(keys)
        return tuple(key in self.manifests for key in keys)

    def get(self, key):
        self.reads.append(key)
        return self.manifests.get(key)


def lookup_case(bridge_module, candidates=3):
    sender = bridge_module.OscarAscendStoreConnector.__new__(bridge_module.OscarAscendStoreConnector)
    sender.namespace = "native-salted-model-layout"
    sender.world_size = 2
    sender.geometry = (
        {"cache_family": "oscar_full", "layer_name": "full0", "storage_block_size": 4},
        {"cache_family": "oscar_full", "layer_name": "full1", "storage_block_size": 8},
        {"cache_family": "gdn", "layer_name": "gdn", "components": ("conv", "ssm")},
    )
    sender._prefix_view = "shared"
    sender._prefix_matches = {}
    sender.metadata_client = Metadata()
    payloads = set()
    sender.transport = NS(exists=lambda keys: tuple(key in payloads for key in keys))
    scheduler = NS(_oscar_prefix_checkpoint_policy=NS(
        alignment_tokens=8, terminal_position=lambda n: (n - 1) // 8 * 8),
        kv_cache_manager=NS(coordinator=NS(hash_block_size=2)))
    sender.scheduler_state = NS(scheduler=scheduler, epoch=5,
        match_resume=lambda request, local: None, observe_request=lambda request: 3)
    request = NS(request_id="reader", num_prompt_tokens=candidates * 8 + 1,
        block_hashes=Hashes(i.to_bytes(8, "big") for i in range(candidates * 4)))
    return sender, request, payloads


def publish(sender, request, count, payloads):
    identity = prefix_plan(request, 3, 5, count, (), sender.geometry, 2,
                           include_page_hashes=False)
    index = sender._prefix_index_key(identity.base_key)
    stored = replace(identity, base_key=f"prefix/data/writer/{count}")
    keys = expected_keys(sender.namespace, sender.world_size, stored, sender.geometry)
    sender.metadata_client.manifests[index] = {
        "namespace": sender.namespace, "prefix_view": sender._prefix_view,
        "index_key": index, "committed_tokens": count, "base_key": stored.base_key,
        "keys": list(keys),
    }
    payloads.update(keys)
    return index, keys


def test_cold_lookup_batches_indices_without_page_hash_walks(bridge_module):
    sender, request, _ = lookup_case(bridge_module, candidates=513)
    assert sender.get_num_new_matched_tokens(request, 0) == (0, False)
    assert [len(keys) for keys in sender.metadata_client.probes] == [256, 256]
    # One longest-candidate metadata GET RPC plus two batch-exists RPCs.
    # host_get's tested existence guard makes that cold RPC issue no SDK GET.
    assert len(sender.metadata_client.reads) == 1
    assert request.block_hashes.reads == 513


def test_cold_miss_does_not_hide_a_later_publication(bridge_module):
    sender, request, payloads = lookup_case(bridge_module)
    assert sender.get_num_new_matched_tokens(request, 0) == (0, False)
    index, _ = publish(sender, request, 24, payloads)
    assert sender.get_num_new_matched_tokens(request, 0) == (24, False)
    assert sender.metadata_client.reads == [index, index]
    assert len(sender.metadata_client.probes) == 1


def test_warm_longest_hit_uses_one_hash_and_one_rpc_without_shorter_probes(bridge_module):
    sender, request, payloads = lookup_case(bridge_module, candidates=513)
    index, _ = publish(sender, request, 513 * 8, payloads)
    request.block_hashes.reads = 0
    assert sender.get_num_new_matched_tokens(request, 0) == (513 * 8, False)
    assert sender.metadata_client.probes == []
    assert sender.metadata_client.reads == [index]
    assert request.block_hashes.reads == 1


@pytest.mark.parametrize("failure", ["evicted_index", "missing_last_tp_gdn"])
def test_lookup_selects_largest_complete_checkpoint_after_race(bridge_module, failure):
    sender, request, payloads = lookup_case(bridge_module)
    longer, keys = publish(sender, request, 24, payloads)
    shorter, _ = publish(sender, request, 16, payloads)
    if failure == "missing_last_tp_gdn":
        payloads.remove(keys[-1])
    else:
        original_get = sender.metadata_client.get
        def get(key):
            if key == longer:
                sender.metadata_client.manifests.pop(key)
            return original_get(key)
        sender.metadata_client.get = get
    assert sender.get_num_new_matched_tokens(request, 8) == (8, False)
    assert sender.metadata_client.reads == [longer, shorter]
    selected = sender._prefix_matches[request.request_id]
    assert selected.committed_tokens == 16 and selected.local_tokens == 8
    assert selected.page_hashes == ()


@pytest.mark.parametrize("field,value", [("namespace", "wrong-model"),
    ("prefix_view", "old-reset"), ("committed_tokens", 16), ("keys", [])])
def test_lookup_corrupt_present_manifest_is_not_a_miss(bridge_module, field, value):
    sender, request, payloads = lookup_case(bridge_module)
    key, _ = publish(sender, request, 24, payloads)
    sender.metadata_client.manifests[key][field] = value
    with pytest.raises(TransferError, match="incompatible identity|incomplete/mixed"):
        sender.get_num_new_matched_tokens(request, 0)


def test_lookup_exists_error_remains_explicit(bridge_module):
    sender, request, _ = lookup_case(bridge_module)
    def fail(keys):
        raise TransferError("SDK unavailable")
    sender.metadata_client.exists = fail
    with pytest.raises(TransferError, match="SDK unavailable"):
        sender.get_num_new_matched_tokens(request, 0)
    assert len(sender.metadata_client.reads) == 1  # Longest miss precedes the batch error.


def test_light_identity_and_default_plan_keep_existing_keys_and_alignment(bridge_module):
    sender, request, _ = lookup_case(bridge_module)
    full = prefix_plan(request, 3, 5, 24, (), sender.geometry, 2)
    light = prefix_plan(request, 3, 5, 24, (), iter(sender.geometry), 2,
                        include_page_hashes=False)
    assert light == replace(full, page_hashes=())
    assert prefix_identity(request, 24, sender.geometry, 2) == full.base_key
    assert tuple(len(hashes) for _, hashes in full.page_hashes) == (6, 3)
    with pytest.raises(ValueError, match="complete physical FULL page"):
        prefix_identity(request, 12, sender.geometry, 2)


def save_case(bridge_module, monkeypatch):
    sender, _, _, _, _, store = connector_pair(bridge_module, monkeypatch, "align")
    scheduler, request, pages = scheduler_for(sender, "align", tokens=12, prompt=12)
    sender.world_size = 2
    scheduler._oscar_prefix_checkpoint_policy = NS(alignment_tokens=4, terminal_position=lambda n: 8)
    request.num_computed_tokens = 4
    sender.metadata_client = Metadata()
    payloads = set()
    sender.transport = NS(exists=lambda keys: tuple(key in payloads for key in keys))
    return sender, scheduler, request, pages, payloads


def test_publication_skips_only_existing_complete_all_tp_checkpoint(bridge_module, monkeypatch):
    sender, _, request, _, payloads = save_case(bridge_module, monkeypatch)
    key, _ = publish(sender, request, 8, payloads)
    metadata = sender.build_connector_meta(scheduled(4, ()))
    assert metadata.prefixes == ()
    assert sender.metadata_client.reads == [key]


@pytest.mark.parametrize("failure", ["absent", "evicted", "missing_tp", "shorter_only"])
def test_publication_missing_or_partial_checkpoint_still_saves(bridge_module, monkeypatch, failure):
    sender, _, request, _, payloads = save_case(bridge_module, monkeypatch)
    if failure != "absent":
        key, keys = publish(sender, request, 4 if failure == "shorter_only" else 8, payloads)
        if failure == "missing_tp":
            payloads.remove(keys[-1])
        elif failure == "evicted":
            sender.metadata_client.get = lambda key: None
    metadata = sender.build_connector_meta(scheduled(4, ()))
    assert len(metadata.prefixes) == 1
    assert metadata.prefixes[0].committed_tokens == 8
    assert metadata.prefixes[0].page_hashes == ()


def test_publication_duplicate_request_prefix_probes_and_checks_once(bridge_module, monkeypatch):
    sender, scheduler, request, pages, payloads = save_case(bridge_module, monkeypatch)
    second = NS(**vars(request))
    second.request_id = "second"
    scheduler.requests[second.request_id] = second
    pages[second.request_id] = pages[request.request_id]
    key, _ = publish(sender, request, 8, payloads)
    output = scheduled(4, ())
    output.num_scheduled_tokens[second.request_id] = 4
    metadata = sender.build_connector_meta(output)
    assert metadata.prefixes == ()
    assert sender.metadata_client.probes == [(key,)]
    assert sender.metadata_client.reads == [key]


def test_publication_cross_namespace_is_error_not_skip(bridge_module, monkeypatch):
    sender, _, request, _, payloads = save_case(bridge_module, monkeypatch)
    key, _ = publish(sender, request, 8, payloads)
    sender.metadata_client.manifests[key]["namespace"] = "wrong-PT"
    with pytest.raises(TransferError, match="incompatible identity"):
        sender.build_connector_meta(scheduled(4, ()))
