"""Archive #70-73/#129/#148/#171: independent work and provenance checks.

Compare analytical reuse counts with enumeration of request-relative query
groups and explicit cluster ownership. No timing thresholds or speed claims.
"""
from math import ceil

import pytest

from tools.analyze_long_context_work import count_chunk, count_chunks, cast_issue_counts


@pytest.mark.parametrize("query,context", [(16384,16384),(13824,16128),(1536,29952),
                                           (16128,32256),(2304,62208),(1024,64512)])
def test_counts_match_enumerated_leaders_and_cluster_owners(query, context):
    row = count_chunk(query, context)
    leaders = [(start,min(start+21,query)) for start in range(0,query,21)]
    cluster_width = row["cluster_size"] * 21
    history_units = [(start,end,kv) for start,end in leaders for kv in range(64,context,256)]
    owners = set()
    clustered_leaders = 0
    for start,end in leaders:
        anchor=start//cluster_width*cluster_width
        grouped=(anchor+cluster_width<=query and end-start==21)
        owner=anchor if grouped else start
        owners.add((grouped,owner))
        clustered_leaders+=grouped
    assert row["history_qk_pv_tile_pairs"] == len(history_units)
    assert row["history_unpack_tiles_after_reuse"] == len(owners)*len(range(64,context,256))
    assert row["cluster_state_loop_gm_address_bytes"] == clustered_leaders*len(range(64,context,256))*264192
    current=sum(len(range(context,context+end,256)) for _,end in leaders)
    assert row["current_qk_pv_tile_pairs"] == current


def test_ideal_history_amplifies_sixfold_without_length_route_switch():
    small=count_chunks([16384]*2)["totals_per_request_per_full_layer_per_rank"]
    large=count_chunks([16384]*4)["totals_per_request_per_full_layer_per_rank"]
    for key in ("history_qk_pv_tile_pairs","history_unpack_tiles_after_reuse",
                "cluster_state_loop_gm_address_bytes"):
        assert large[key] == 6*small[key]
    assert large["current_qk_pv_tile_pairs"] == 3*small["current_qk_pv_tile_pairs"]
    assert large["new_tokens"] == 2*small["new_tokens"]


def test_prefix_hit_is_explicit_and_changes_work_without_claiming_a_hit_rate():
    cold=count_chunks([16128,13824,1536,1280])
    hit=count_chunks([1536,1280],cached_prefix_tokens=29952)
    assert cold["prompt_tokens"] == hit["prompt_tokens"] == 32768
    assert cold["assumed_cached_prefix_tokens"] == 0
    assert hit["assumed_cached_prefix_tokens"] == 29952
    assert hit["totals_per_request_per_full_layer_per_rank"]["new_tokens"] == 2816
    assert hit["chunks"][0]["route"] == "official_C4"


@pytest.mark.parametrize("elements", [0,1,63,64,65,4096,16320,16321,16384])
def test_cast_instruction_counts_cover_each_element_once(elements):
    old=[(start,min(start+64,elements)) for start in range(0,elements,64)]
    repeats=[]
    full=elements//64
    for start in range(0,full,255):repeats.append((start*64,min(start+255,full)*64))
    if elements%64:repeats.append((full*64,elements))
    assert [i for a,b in repeats for i in range(a,b)] == list(range(elements))
    result=cast_issue_counts(elements)
    assert result["old_single_repeat_instructions"] == len(old)
    assert result["new_batched_repeat_instructions"] == len(repeats)


def test_reject_unmodeled_decode_geometry_and_invalid_inputs():
    for query,context in ((4,32768),(0,0),(16385,0),(1024,-1)):
        with pytest.raises(ValueError):count_chunk(query,context)
