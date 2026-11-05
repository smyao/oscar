"""Archive #86/#97/#165/#166/#170/#174: exclude fitted recipe3 source records.

Offline-only input construction. Reconstruct and authenticate the original
whole-prompt seed input before selecting new records. Different calibration
buckets begin at different complete records; this never changes serving.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from tools.calibration_data import CALIBRATION_SPLITS
from tools.calibrate_model_rotations import load_messages, prompt_ids_sha256

SOURCE_PLAN = "recipe3_excluded_records_independent_buckets_v1"


def _digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("recipe3 sampling requires lowercase SHA256 identities")
    return value


def _seed_metadata(value, source_sha256):
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("recipe3 sampling requires authenticated baseline capture metadata")
    _hash(value.get("baseline_artifact_sha256"))
    _hash(value.get("prompt_sha256"))
    profile = value.get("calibration_profile")
    if (not isinstance(profile, dict) or profile.get("recipe_version") != 3 or
            profile.get("rotation_variant") != "perhead_center" or
            value.get("model_fingerprint") != profile.get("model_fingerprint") or
            type(profile.get("max_tokens")) is not int or profile["max_tokens"] <= 0 or
            type(value.get("token_count")) is not int or not 0 < value["token_count"] <= profile["max_tokens"]):
        raise ValueError("recipe3 sampling baseline profile/token identity differs")
    source = profile.get("prompt_source")
    if (not isinstance(source, dict) or source.get("sha256") != _hash(source_sha256) or
            value.get("prompt_source") != source or
            value.get("prompt_source_sha256", source_sha256) != source_sha256):
        raise ValueError("recipe3 sampling source differs from baseline calibration")
    lengths = value.get("prompt_lengths")
    if lengths is not None and (not isinstance(lengths, list) or not lengths or
            any(type(n) is not int or n <= 0 for n in lengths) or sum(lengths) != value["token_count"]):
        raise ValueError("recipe3 sampling baseline prompt lengths differ")
    return copy.deepcopy(value)


def _render(tokenizer, message):
    ids = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_dict=False)
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("recipe3 sampling requires one rendered prompt per source record")
        ids = ids[0]
    if not isinstance(ids, list) or not ids or any(type(n) is not int or n < 0 for n in ids):
        raise ValueError("recipe3 sampling requires nonempty integer prompt tokens")
    return ids


def _seed_prefix(records, prompts, seed, chunk):
    count, selected = 0, []
    for record, prompt in zip(records, prompts):
        if count + len(prompt) > seed["calibration_profile"]["max_tokens"]:
            break
        if len(prompt) > chunk:
            raise ValueError("recipe3 seed prompt exceeds the original whole-prefill budget")
        selected.append(prompt)
        count += len(prompt)
        if count == seed["calibration_profile"]["max_tokens"]:
            break
    if (not selected or count != seed["token_count"] or prompt_ids_sha256(selected) != seed["prompt_sha256"] or
            seed.get("prompt_lengths", list(map(len, selected))) != list(map(len, selected))):
        raise ValueError("recipe3 original whole-prompt reconstruction differs from authenticated baseline")
    excluded = sorted(set(records[:len(selected)]))
    return selected, excluded


def _bucket_spans(pool, lengths, offset, target):
    remaining, index, spans = target, offset, []
    while remaining:
        identity = pool[index % len(pool)]
        take = min(remaining, lengths[identity])
        spans.append({"record_sha256": identity, "start": 0, "end": take,
                      "cycle": (index - offset) // len(pool)})
        remaining -= take
        index += 1
    return spans


def _partitions(records, excluded):
    eligible = sorted(set(records) - set(excluded))
    if len(eligible) < len(CALIBRATION_SPLITS):
        raise ValueError("recipe3 baseline leaves fewer than three independent source records")
    return {split: eligible[index::len(CALIBRATION_SPLITS)]
            for index, split in enumerate(CALIBRATION_SPLITS)}


def build_recipe3_request_plan(tokenizer, messages, policy, seed_capture, *, source_sha256):
    """Authenticate fitted inputs, then pack only unfitted complete records."""
    if policy.get("source_plan") != SOURCE_PLAN:
        raise ValueError("recipe3 sampling policy identity differs")
    seed = _seed_metadata(seed_capture, source_sha256)
    records = [_digest(message) for message in messages]
    prompts = [_render(tokenizer, message) for message in messages]
    selected, excluded = _seed_prefix(records, prompts, seed, policy["sampling_chunk_tokens"])
    by_id = {}
    for identity, prompt in zip(records, prompts):
        if identity in by_id and by_id[identity] != prompt:
            raise ValueError("recipe3 duplicate source record rendered inconsistent tokens")
        by_id[identity] = prompt
    partitions = _partitions(records, excluded)
    lengths = {identity: len(ids) for identity, ids in by_id.items()}
    plan = []
    for split, pool in partitions.items():
        buckets = policy["capture_lengths"][split]
        if len(pool) < len(buckets):
            raise ValueError("recipe3 held-out split needs a distinct source start for each length bucket")
        for index, length in enumerate(buckets):
            offset = index * len(pool) // len(buckets)
            spans = _bucket_spans(pool, lengths, offset, length)
            ids = [token for span in spans for token in by_id[span["record_sha256"]][span["start"]:span["end"]]]
            used = list(dict.fromkeys(span["record_sha256"] for span in spans))
            stream_tokens = sum(lengths[identity] for identity in pool)
            plan.append({"request_id": f"{split}_{length}_{index}", "split": split,
                         "length_bucket": length, "prompt_token_ids": ids,
                         "source_record_sha256": used, "source_stream_tokens": stream_tokens,
                         "source_repeated": length > stream_tokens,
                         "source_record_offset": offset, "source_record_spans": spans,
                         "prompt_sha256": prompt_ids_sha256([ids])})
    provenance = {"schema_version": 1, "source_plan": SOURCE_PLAN,
                  "seed_capture": seed, "seed_capture_sha256": _digest(seed),
                  "prompt_source_sha256": source_sha256,
                  "source_records": [{"sha256": identity, "token_count": len(ids)}
                                     for identity, ids in zip(records, prompts)],
                  "seed_prompt_token_ids": selected,
                  "excluded_record_sha256": excluded,
                  "partition_record_sha256": partitions}
    validate_recipe3_sampling(provenance, plan, policy, seed, source_sha256, messages=messages)
    return plan, provenance


def validate_recipe3_sampling(provenance, plan, policy, seed_capture, source_sha256, *, messages):
    """Check retained seed identity, exact exclusion, and each bucket's spans.

    The captured seed token lists must reproduce the original authenticated
    aggregate hash. Source record identities are independently rebuilt from
    the authenticated original source bytes by the trace loader. No model or
    tokenizer is loaded while fitting a retained capture.
    """
    seed = _seed_metadata(seed_capture, source_sha256)
    if (not isinstance(provenance, dict) or provenance.get("schema_version") != 1 or
            provenance.get("source_plan") != SOURCE_PLAN or policy.get("source_plan") != SOURCE_PLAN or
            provenance.get("seed_capture") != seed or provenance.get("seed_capture_sha256") != _digest(seed) or
            provenance.get("prompt_source_sha256") != source_sha256):
        raise ValueError("recipe3 trace seed/source provenance differs")
    rows, prompts = provenance.get("source_records"), provenance.get("seed_prompt_token_ids")
    expected_records = [_digest(message) for message in messages]
    if (not isinstance(rows, list) or len(rows) != len(expected_records) or not rows or
            any(not isinstance(row, dict) or row.get("sha256") != identity or
                type(row.get("token_count")) is not int or row["token_count"] <= 0
                for row, identity in zip(rows, expected_records))):
        raise ValueError("recipe3 trace source record identities/token lengths differ")
    if (not isinstance(prompts, list) or not prompts or len(prompts) > len(rows) or
            any(not isinstance(ids, list) or not ids or any(type(n) is not int or n < 0 for n in ids)
                for ids in prompts)):
        raise ValueError("recipe3 trace lacks authenticated reconstructed seed prompt tokens")
    n = len(prompts)
    counts = [len(ids) for ids in prompts]
    if (counts != [row["token_count"] for row in rows[:n]] or sum(counts) != seed["token_count"] or
            seed.get("prompt_lengths", counts) != counts or prompt_ids_sha256(prompts) != seed["prompt_sha256"] or
            (n < len(rows) and sum(counts) + rows[n]["token_count"] <= seed["calibration_profile"]["max_tokens"])):
        raise ValueError("recipe3 trace reconstructed seed does not match whole-prompt baseline")
    excluded = sorted(set(expected_records[:n]))
    partitions = _partitions(expected_records, excluded)
    if (provenance.get("excluded_record_sha256") != excluded or
            provenance.get("partition_record_sha256") != partitions):
        raise ValueError("recipe3 trace seed-record exclusion or held-out partition differs")
    lengths = {}
    for row in rows:
        identity = row["sha256"]
        if identity in lengths and lengths[identity] != row["token_count"]:
            raise ValueError("recipe3 trace duplicate source record length differs")
        lengths[identity] = row["token_count"]
    expected = [(split, index, length) for split in CALIBRATION_SPLITS
                for index, length in enumerate(policy["capture_lengths"][split])]
    if not isinstance(plan, list) or len(plan) != len(expected):
        raise ValueError("recipe3 trace request coverage differs")
    for row, (split, index, length) in zip(plan, expected):
        pool, buckets = partitions[split], policy["capture_lengths"][split]
        if len(pool) < len(buckets):
            raise ValueError("recipe3 held-out split lacks distinct source starts")
        offset = index * len(pool) // len(buckets)
        spans = _bucket_spans(pool, lengths, offset, length)
        used = list(dict.fromkeys(span["record_sha256"] for span in spans))
        stream_tokens = sum(lengths[identity] for identity in pool)
        if (not isinstance(row, dict) or row.get("request_id") != f"{split}_{length}_{index}" or
                row.get("split") != split or row.get("length_bucket") != length or
                row.get("source_record_offset") != offset or row.get("source_record_spans") != spans or
                row.get("source_record_sha256") != used or row.get("source_stream_tokens") != stream_tokens or
                type(row.get("source_repeated")) is not bool or row["source_repeated"] != (length > stream_tokens)):
            raise ValueError("recipe3 trace bucket offset/actual source coverage differs")


def validate_recipe3_trace_manifest(top, config, policy, *, files=None):
    source = Path(top.get("prompt_source", ""))
    if files is not None:
        source = files.resolve(source)
    if not source.is_file():
        raise ValueError("recipe3 trace requires its authenticated source file")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha != top.get("prompt_source_sha256"):
        raise ValueError("recipe3 trace source file hash differs")
    validate_recipe3_sampling(top.get("recipe3_sampling"), top.get("request_plan"), policy,
        config.get("optr_seed_capture"), source_sha, messages=load_messages(source))
