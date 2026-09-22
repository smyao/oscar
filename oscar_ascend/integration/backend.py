"""Native backend and metadata construction seams.

Archive #27/#28/#34/#36/#37–49: no placeholder attention, host length
readback, or capture-only suppression masquerading as graph replay.
Only an installed, ready runtime can provide the real AttentionImpl.
"""

import torch
from dataclasses import replace
from vllm.v1.attention.backend import AttentionBackend, AttentionCGSupport, AttentionMetadataBuilder

from ..layout import SlotLayout
from .metadata import GraphMetadataBindings, from_common
from .runtime_api import require_runtime


class OscarAttentionBackend(AttentionBackend):
    accept_output_buffer = True
    forward_includes_kv_cache_update = True
    supported_dtypes = [torch.bfloat16]

    @staticmethod
    def get_supported_head_sizes():
        return [64, 128, 256]

    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return require_runtime().get_impl_cls()

    @staticmethod
    def get_builder_cls():
        return OscarMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str=""):
        # Native signature has no head_size_v. Actual hybrid allocation must
        # use OscarFullAttentionSpec.layout and packed_view, not this hint.
        return (num_blocks, block_size, num_kv_heads, SlotLayout(head_size).slot_bytes)

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [128]


class OscarMetadataBuilder(AttentionMetadataBuilder):
    # The intended graph contract is uniform decode (q_len = 1 + MTP).
    # This declaration is not evidence that graph capture/replay has passed.
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold = None

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.capacity = None
        self.bindings = GraphMetadataBindings()

    def reorder_batch(self, input_batch, scheduler_output):
        return False

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        return self.bindings.bind(from_common(common_attn_metadata, capacity=self.capacity))

    def build_for_graph_capture(self, common_attn_metadata, attn_state=None):
        # Native _dummy_run does not necessarily initialize slot values when
        # is_graph_capturing=True (model_runner_v1.py:3548). Mark its synthetic
        # inputs explicitly. The complete kernels are still captured; replay
        # reads the same native buffer after _prepare_inputs fills real slots.
        common_attn_metadata.slot_mapping.fill_(-1)
        return self.bindings.bind(from_common(common_attn_metadata, capacity=self.capacity, capture_origin=True))

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return self.build_for_graph_capture(common_attn_metadata)

    def build_for_drafting(self, common_attn_metadata, draft_index):
        # Native padded first pass retains rejected verification rows. Later
        # draft slots follow the accepted position, while seq_lens is merely
        # incremented (llm_base_proposer.py:1608/1654/1953). Recover the logical
        # query position from slot + virtual block table on device, not mRoPE.
        if type(draft_index) is not int or draft_index < 0:
            raise ValueError("draft_index must be a nonnegative integer")
        return replace(self.build(0, common_attn_metadata, fast_build=True), draft_index=draft_index)
