"""Mixed CV owner-tile proof for the isolated fast balanced kernel.

Archive #126/#129/#140/#143-151 and startup D.4: this checks only task
ownership and source isolation. CPU arithmetic cannot establish target NPU
precision, graph replay or speed; those remain separate gates.
"""

from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "csrc/kernels/attention_cv_fast.cpp"
BALANCED = ROOT / "csrc/kernels/attention_cv_fast_balanced.cpp"


def _owners(tokens: int, per_token: int, tile: int, cores: int):
    tiles = (tokens + tile - 1) // tile
    counts = bytearray(tokens * per_token)
    token_owners = {}
    for work_id in range(tiles * per_token):
        segment = work_id // tiles
        token_begin = (work_id % tiles) * tile
        for token in range(token_begin, min(tokens, token_begin + tile)):
            task_id = token * per_token + segment
            counts[task_id] += 1
            token_owners[token, segment] = work_id % cores
    assert all(count == 1 for count in counts)
    return token_owners, tiles * per_token


def test_mixed_q4_and_long_leaders_spread_without_duplicate_tasks():
    n, cores = 16384, 20
    old, old_items = _owners(n, 3, 21, cores)
    new, new_items = _owners(n, 3, 4, cores)
    short = tuple(range(0, 124, 4))
    long = tuple(range(124, n, 21))
    old_short = Counter(old[token, 0] for token in short)
    new_short = Counter(new[token, 0] for token in short)
    assert len(old_short) == 6 and max(old_short.values()) == 6
    assert len(new_short) == 20 and max(new_short.values()) == 2
    assert len({old[token, 0] for token in long}) == 20
    assert len({new[token, 0] for token in long}) == 20
    assert (old_items, new_items) == (2343, 12288)
    assert len(old) == len(new) == n * 3

    # The task generator's mathematical qcount is unchanged. One long leader
    # owns all its 21 query rows even when the four-token owner tile ends.
    covered = []
    for begin in short:
        covered.extend(range(begin, begin + 4))
    for begin in long:
        covered.extend(range(begin, min(begin + 21, n)))
    assert covered == list(range(n))


def test_frontier_padding_and_split_segments_remain_bijective():
    for tokens, kv_heads, splits in ((1, 1, 1), (17, 1, 3), (128, 1, 3),
                                      (129, 2, 3), (16384, 1, 1)):
        per_token = kv_heads * 3 * splits
        owners, _ = _owners(tokens, per_token, 4, 20)
        assert len(owners) == tokens * per_token
        assert all((token, segment) in owners
                   for token in range(tokens) for segment in range(per_token))


def test_padded_128_q4_only_reduces_critical_history_leaders_by_one():
    # Pure N128/S3 stays on the already measured fast base route. Its owner
    # improvement is small compared with the N16384/S1 mixed prefix above.
    old, _ = _owners(128, 9, 21, 20)
    new, _ = _owners(128, 9, 4, 20)
    leaders = range(0, 124, 4)  # 31 live q4 requests; last four are padding.
    old_counts = Counter(old[token, split] for token in leaders for split in range(3))
    new_counts = Counter(new[token, split] for token in leaders for split in range(3))
    assert (len(old_counts), max(old_counts.values())) == (18, 6)
    assert (len(new_counts), max(new_counts.values())) == (20, 5)


def test_c4_pass_two_anchors_keep_unique_members_after_owner_change():
    n, request_begin = 16384, 124
    owners, _ = _owners(n, 3, 4, 20)
    anchors = range(request_begin, n - 83, 84)
    seen = set()
    for anchor in anchors:
        members = tuple(anchor + 21 * member for member in range(4))
        assert all((token, 0) in owners for token in members)
        assert not any(token in seen for token in members)
        seen.update(members)
        # Pass two's global 84-token buckets partition every anchor once;
        # the request-relative C4 member predicate remains in its old code.
        assert sum(bucket * 84 <= anchor < min((bucket + 1) * 84, n)
                   for bucket in range((n + 83) // 84)) == 1
    assert len(seen) == 4 * len(tuple(anchors))


def test_balanced_clone_changes_only_owner_schedule_and_exported_names():
    base = BASE.read_text()
    balanced = BALANCED.read_text()
    marker = "// EXPERIMENT ONLY: exact INT2 unpack for the fe0 query-major schedule. Archive"
    assert marker in base and marker in balanced
    # The new D.4 header is intentionally additional. All inherited math and
    # flags must remain text-identical after normalizing the owner schedule.
    balanced = balanced.split(marker, 1)[1]
    base = base.split(marker, 1)[1]
    balanced = balanced.replace(
        "    // This is an owner tile, not the task's mathematical query group.\n"
        "    // Prepare still permits qcount<=kQueryRows/(hq/hk); a 21-token leader\n"
        "    // may cross several four-token owner tiles, whose followers only write\n"
        "    // their own status. Do not split the leader or its ordered KV traversal.\n"
        "    constexpr int64_t scheduleTile=4;",
        "    const int64_t queryTile=kQueryRows/(g.hq/g.hk);")
    balanced = balanced.replace("CvTaskSchedule schedule{g.tokens,scheduleTile,perToken}",
                                "CvTaskSchedule schedule{g.tokens,queryTile,perToken}")
    balanced = balanced.replace("tokenBegin+scheduleTile,g.tokens",
                                "tokenBegin+queryTile,g.tokens")
    balanced = balanced.replace("oscar_attention_cv_fast_balanced_kernel",
                                "oscar_attention_cv_fast_kernel")
    balanced = balanced.replace("attention_cv_fast_balanced_launch",
                                "attention_cv_fast_launch")
    assert balanced == base
