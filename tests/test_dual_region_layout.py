"""Archive #17-20/#34/#36/#127-129: byte ownership and virtual128 contracts."""
from dataclasses import replace

import pytest

from oscar_ascend.layout import HybridPageLayout,SlotLayout
from oscar_ascend.dual_region_layout import DualRegionPageLayout


def native():
    return HybridPageLayout(SlotLayout(256,256,1),30720,393216,262144,128,
                            817152,'none')


def overlap(a,b):return max(a[0],b[0])<min(a[1],b[1])


def test_same_native_allocation_twice_the_full_token_capacity():
    old=native();new=DualRegionPageLayout(old)
    assert old.block_size==2816 and new.block_size==5632
    assert new.virtual_blocks_per_page==44
    assert new.page_size_bytes==old.page_size_bytes==817152
    ledger=new.byte_ledger(13)
    assert sum(ledger[k] for k in ('packed_payload_capacity_bytes','unused_conv_bytes',
                                  'unused_segment_alignment_bytes'))==ledger['native_pool_bytes']
    assert new.payload_bytes==2*old.payload_bytes==765952


def test_every_packed_row_stays_in_its_exact_native_owned_segments():
    old=native();new=DualRegionPageLayout(old);count=3;size=old.slot_layout.token_bytes
    regions=[]
    for page in range(count):
        segments=new.segment_intervals(count,page)
        for token in range(new.block_size):
            start=new.packed_offset(count,page,token);region=(start,start+size)
            chosen=segments[token//new.segment_tokens]
            assert chosen[0]<=region[0]<region[1]<=chosen[1]
            assert not overlap(region,old.conv_interval(count,page))
            for other in range(count):
                if other!=page:
                    assert not overlap(region,old.conv_interval(count,other))
                    assert not overlap(region,old.ssm_interval(count,other))
            regions.append(region)
    regions.sort()
    assert all(left[1]<=right[0] for left,right in zip(regions,regions[1:]))


def test_native_virtual128_formula_and_segment_crossing():
    layout=DualRegionPageLayout(native());count=4
    for page in range(count):
        for pos in (0,127,128,2815,2816,2943,2944,5631):
            block=page*44+pos//128
            slot=block*128+pos%128
            assert slot==layout.metadata_slot(count,page,pos)
            assert layout.slot_offset(count,slot)==layout.packed_offset(count,page,pos)
    first_end=layout.packed_offset(count,2,2815)+136
    second_start=layout.packed_offset(count,2,2816)
    assert second_start!=first_end  # never pretend the two regions are one AoS row
    assert second_start==count*(30720+393216)+2*393216


def test_prefix_geometry_and_invalid_addresses_fail_explicitly():
    old=native()
    for mode in ('align','all'):
        with pytest.raises(ValueError,match='mode none'):
            DualRegionPageLayout(replace(old,native_mamba_cache_mode=mode,
                                         native_mamba_block_size=768))
    with pytest.raises(ValueError,match='equally sized'):
        DualRegionPageLayout(replace(old,native_page_size_bytes=old.page_size_bytes+128))
    layout=DualRegionPageLayout(old)
    for page,token in ((-1,0),(4,0),(0,-1),(0,5632)):
        with pytest.raises(ValueError):layout.packed_offset(4,page,token)
