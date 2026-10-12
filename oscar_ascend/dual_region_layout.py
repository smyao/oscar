"""Two INT2 segments in the same native-owned FULL page.

Archive #17-20/#26/#34/#36/#37-49/#69/#127-129/#155: preserve the native
SoA allocation, disjoint global page ownership and metadata128 semantics.
This optional layout is not a serving route yet. It requires a separate
request-stable exact window; no per-page snapshot may coexist in segment2.
GDN conv/SSM layout and original native page size remain unchanged.
"""
from dataclasses import dataclass

from .layout import HybridPageLayout


@dataclass(frozen=True)
class DualRegionPageLayout:
    native: HybridPageLayout

    def __post_init__(self):
        if not isinstance(self.native,HybridPageLayout):
            raise ValueError('dual-region layout requires verified native geometry')
        if self.native.native_mamba_cache_mode!='none':
            raise ValueError('request windows require native Mamba mode none')
        if self.native.native_padding_bytes!=self.native.ssm_bytes:
            raise ValueError('dual-region ABI requires equally sized native SSM/padding segments')
        if self.native.kernel_block_size!=128:
            raise ValueError('dual-region metadata ABI requires virtual128')
        if (self.native.slot_layout.head_size != 256 or
                self.native.slot_layout.head_size_v != 256):
            raise ValueError('dual-region striped reader/writer ABI requires D256')

    @property
    def block_size(self):return 2*self.native.block_size

    @property
    def segment_tokens(self):return self.native.block_size

    @property
    def virtual_blocks_per_page(self):return self.block_size//self.native.kernel_block_size

    @property
    def page_size_bytes(self):return self.native.page_size_bytes

    @property
    def payload_bytes(self):return self.block_size*self.native.slot_layout.token_bytes

    def segment_intervals(self,num_blocks,physical_page):
        if (type(num_blocks) is not int or num_blocks <= 0 or
                type(physical_page) is not int or not 0 <= physical_page < num_blocks):
            raise ValueError('invalid physical pool page')
        first=self.native.ssm_interval(num_blocks,physical_page)
        start=num_blocks*(self.native.conv_bytes+self.native.ssm_bytes)
        start+=physical_page*self.native.native_padding_bytes
        return first,(start,start+self.native.native_padding_bytes)

    def packed_offset(self,num_blocks,physical_page,token_in_page):
        """Logical packed row location; stripe permutation stays within a segment."""
        if type(token_in_page) is not int or not 0<=token_in_page<self.block_size:
            raise ValueError('packed token outside logical physical-page capacity')
        segment,offset=divmod(token_in_page,self.segment_tokens)
        start,_=self.segment_intervals(num_blocks,physical_page)[segment]
        return start+offset*self.native.slot_layout.token_bytes

    def metadata_slot(self,num_blocks,physical_page,token_in_page):
        # Same number that native block_table expansion and slot mapping emit:
        # (physical*44 + token//128)*128 + token%128 for D256/Hkv1.
        self.packed_offset(num_blocks,physical_page,token_in_page)
        return physical_page*self.block_size+token_in_page

    def slot_offset(self,num_blocks,slot):
        if type(slot) is not int or slot<0:raise ValueError('invalid live slot')
        physical,token=divmod(slot,self.block_size)
        return self.packed_offset(num_blocks,physical,token)

    def byte_ledger(self,num_blocks):
        if type(num_blocks) is not int or num_blocks<=0:raise ValueError('invalid physical pool size')
        return {'native_pool_bytes':num_blocks*self.page_size_bytes,
                'packed_payload_capacity_bytes':num_blocks*self.payload_bytes,
                'unused_conv_bytes':num_blocks*self.native.conv_bytes,
                'unused_segment_alignment_bytes':num_blocks*2*self.native.unused_ssm_bytes,
                'precise_window_bytes_in_native_pool':0,
                'request_window_sidecar':'must be separately reserved before KV profiling'}
