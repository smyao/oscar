"""Archive #126/#145/#164: test-only v3 inputs from original BF16 rows.

Reuse native task/page geometry and the original RNG inputs, never decode an
old cache to manufacture a new-codec golden. No serving module imports this.
"""
from dataclasses import replace
from . import probe_history_reuse as reuse
from .aligned_reference import pack_kv, rotate_bf16, CACHE_FORMAT


def make_fixture(torch, shape, *, rotation_mode="nontrivial", k_clip=.96, v_clip=.92):
    histories = {}
    def capture(request, _begin, _q, key, value, _ck, _cv):
        histories[request] = (key, value)
    clean = replace(shape, corrupt_metadata=False, corrupt_qr=False, corrupt_dead_tail=False)
    fixture = reuse.make_fixture(torch, clean, rotation_mode=rotation_mode, on_request=capture)
    fixture["spec"] = shape
    dim, hk = shape.dim, shape.kv_heads
    if dim != 256:
        raise ValueError("aligned production fixture requires D256")
    rk = torch.eye(dim) if rotation_mode == "identity" else reuse._hadamard(torch, dim)
    rv = fixture["cpu"]["rv"].to(torch.bfloat16)
    stride = reuse.BLOCK_TOKENS * hk * (dim // 2 + 16)
    raw = torch.full((reuse.PREFIX + fixture["blocks"] * stride,), 0xA5, dtype=torch.uint8)
    slots = raw[reuse.PREFIX:].view(fixture["blocks"], reuse.BLOCK_TOKENS, hk, dim // 2 + 16)
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
    qr = rotate_bf16(fixture["cpu"]["q"], rk)
    if shape.corrupt_metadata or shape.corrupt_dead_tail:
        position = shape.corrupt_position if shape.corrupt_metadata else shape.contexts[0] + shape.qlens[0] + 1
        physical = fixture["page_assignments"][0][position // reuse.BLOCK_TOKENS]
        slots[physical, position % reuse.BLOCK_TOKENS, hk - 1, dim // 2:dim // 2 + 4] = 0
    if shape.corrupt_qr:
        qr[shape.corrupt_qr_token, 0, 0] = float("nan")
    fixture["cpu"].update(raw=raw, qr=qr, rv=rv)
    fixture.update(stride=stride, cache_format=CACHE_FORMAT, slot_bytes=dim // 2 + 16,
                   rotation_k=rk.to(torch.bfloat16), k_clip=k_clip, v_clip=v_clip,
                   input_sha256=reuse._fingerprint(fixture["cpu"]))
    return fixture
