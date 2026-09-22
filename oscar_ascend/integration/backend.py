"""Native backend and metadata construction seams.

Archive #27/#28/#34/#36/#37–49: no placeholder attention, host length
readback, or capture-only suppression masquerading as graph replay.
Only an installed, ready runtime can provide the real AttentionImpl.
"""

from vllm.v1.attention.backend import AttentionBackend, AttentionCGSupport, AttentionMetadataBuilder

from ..layout import SlotLayout
from .metadata import from_common
from .runtime_api import require_runtime


class OscarAttentionBackend(AttentionBackend):
    accept_output_buffer = True

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

    def reorder_batch(self, input_batch, scheduler_output):
        return False

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        return from_common(common_attn_metadata, capacity=self.capacity)

    def build_for_graph_capture(self, common_attn_metadata, attn_state=None):
        return from_common(common_attn_metadata, capacity=self.capacity, capture_origin=True)

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return self.build_for_graph_capture(common_attn_metadata)
