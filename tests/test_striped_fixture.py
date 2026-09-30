"""Archive #13-22/#154: independent scalar bit mapping and byte preservation."""
import torch
import pytest
from tools.striped_fixture import canonical_slots_to_striped, striped_slots_to_canonical


def test_all_words_and_all_dimensions_roundtrip():
    # Every uint16 bit pattern appears in each of 32 independent word slots.
    source = torch.arange(65536, dtype=torch.int32)
    for begin in range(0, 65536, 1024):
        words = source[begin:begin + 1024]
        striped = torch.empty((words.numel(), 136), dtype=torch.uint8)
        striped[:, :128:2] = (words & 255)[:, None]
        striped[:, 1:128:2] = (words >> 8)[:, None]
        striped[:, 128:] = torch.arange(8, dtype=torch.uint8)
        canonical = striped_slots_to_canonical(striped)
        assert torch.equal(canonical_slots_to_striped(canonical), striped)
        # Independent identity: natural dimension d comes from bit d//32.
        for d in range(256):
            got = (canonical[:, d // 4].int() >> (2 * (d % 4))) & 3
            expected = (words >> (2 * (d // 32))) & 3
            assert torch.equal(got, expected)


def test_random_slots_metadata_and_no_alias():
    torch.manual_seed(154)
    canonical = torch.randint(0, 256, (3, 5, 136), dtype=torch.uint8)
    before = canonical.clone()
    striped = canonical_slots_to_striped(canonical)
    assert torch.equal(striped_slots_to_canonical(striped), canonical)
    assert torch.equal(striped[..., 128:132], canonical[..., 64:68])
    assert torch.equal(striped[..., 132:136], canonical[..., 132:136])
    assert torch.equal(canonical, before)
    assert striped.data_ptr() != canonical.data_ptr()


def test_wrong_layout_rejected():
    with pytest.raises(ValueError, match="D256"):
        canonical_slots_to_striped(torch.zeros(3, 72, dtype=torch.uint8))
