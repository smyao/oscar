"""Archive #37-49/#126/#145/#164: test-only v4 rotated-source fixtures.

Reuse the existing physical/native metadata geometry and prequantization RNG
rows, then independently rotate and encode with the v4 reference. No old v3
output or decoded old cache defines a v4 golden. Production never imports this.
"""
from dataclasses import replace

from . import probe_history_reuse as reuse
from .official_reference import CACHE_FORMAT, pack_kv, rotate_bf16


def make_fixture(torch, shape, *, rotation_mode="nontrivial", k_clip=.96, v_clip=.92):
    histories = {}
    def capture(request, _begin, _query, key, value, _current_key, _current_value):
        histories[request] = (key, value)
    clean = replace(shape, corrupt_metadata=False, corrupt_qr=False, corrupt_dead_tail=False)
    fixture = reuse.make_fixture(torch, clean, rotation_mode=rotation_mode, on_request=capture)
    fixture["sample_indices"] = tuple(sorted(fixture["expected"]))
    # Keep sample selection compatibility without retaining any old numerical
    # golden. Lifecycle probes must obtain values from the v4 oracle.
    fixture["expected"] = {index: None for index in fixture["sample_indices"]}
    fixture["spec"] = shape
    dim, heads = shape.dim, shape.kv_heads
    if dim != 256:
        raise ValueError("official production fixture requires D256")
    rk = (torch.eye(dim) if rotation_mode == "identity" else reuse._hadamard(torch, dim)).bfloat16()
    rv = fixture["cpu"]["rv"].bfloat16()
    data = fixture["cpu"]
    original = {name: data[name].clone() for name in ("q", "ck", "cv")}
    q = rotate_bf16(original["q"], rk)
    data["q"] = q
    data["qr"] = q.clone()
    data["ck"] = rotate_bf16(original["ck"], rk)
    data["cv"] = rotate_bf16(original["cv"], rv)
    valid = data["tags"] >= 0
    data["wk"][valid] = rotate_bf16(data["wk"][valid], rk)
    data["wv"][valid] = rotate_bf16(data["wv"][valid], rv)
    data["rv"] = rv
    slot_bytes = dim // 2 + 16
    stride = reuse.BLOCK_TOKENS * heads * slot_bytes
    raw = torch.full((reuse.PREFIX + fixture["blocks"] * stride,), 0xA5, dtype=torch.uint8)
    slots = raw[reuse.PREFIX:].view(fixture["blocks"], reuse.BLOCK_TOKENS, heads, slot_bytes)
    for request, pages in enumerate(fixture["page_assignments"]):
        key, value = histories[request]
        if not key.numel():
            continue
        packed = pack_kv(rotate_bf16(key, rk), rotate_bf16(value, rv), k_clip, v_clip)
        for logical, physical in enumerate(pages):
            start = logical * reuse.BLOCK_TOKENS
            count = min(reuse.BLOCK_TOKENS, key.shape[0] - start)
            if count > 0:
                slots[physical, :count] = packed[start:start + count]
    if shape.corrupt_metadata or shape.corrupt_dead_tail:
        position = (shape.corrupt_position if shape.corrupt_metadata else
                    shape.contexts[0] + shape.qlens[0] + 1)
        physical = fixture["page_assignments"][0][position // reuse.BLOCK_TOKENS]
        slots[physical, position % reuse.BLOCK_TOKENS, heads - 1, dim // 2:dim // 2 + 4] = 0
    if shape.corrupt_qr:
        # Explicit fault injection into the history-query buffer only; a valid
        # fixture always supplies identical rotated Q to every source.
        data["qr"][shape.corrupt_qr_token, 0, 0] = float("nan")
    data["raw"] = raw
    fixture.update(stride=stride, cache_format=CACHE_FORMAT, slot_bytes=slot_bytes,
                   original_inputs=original, original_histories=histories,
                   rotation_k=rk, k_clip=k_clip, v_clip=v_clip,
                   sink_tokens=reuse.SINK, speculative_tokens=reuse.SPECULATIVE,
                   window_ring_tokens=shape.recent_tokens + reuse.SPECULATIVE,
                   cached_quant_ends=[min(c, max(reuse.SINK, c + 1 - shape.recent_tokens))
                                      for c in shape.contexts],
                   input_sha256=reuse._fingerprint(data))
    return fixture
