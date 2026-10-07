"""Native backend and metadata construction seams.

Archive #27/#28/#34/#36/#37–49/#140/#142/#148/#155: no placeholder attention, host length
readback, or capture-only suppression masquerading as graph replay.
Only an installed, ready runtime can provide the real AttentionImpl.
"""

import torch
from vllm.v1.attention.backend import AttentionBackend, AttentionCGSupport, AttentionMetadataBuilder

from ..layout import SlotLayout
from .first_draft_current_only import supports_first_draft_current_only
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
        runtime_config = require_runtime().config
        self.current_only = runtime_config.get("experimental_current_only", False)
        self.first_draft_current_only = (
            self.current_only and supports_first_draft_current_only(vllm_config))
        first_current = runtime_config.get("experimental_first_mtp_current_fia", False)
        if type(first_current) is not bool:
            raise ValueError("experimental_first_mtp_current_fia must be an explicit boolean")
        self.first_draft_current_fia = first_current and supports_first_draft_current_only(vllm_config)
        mixed_split = runtime_config.get("experimental_mixed_decode_split", False)
        if type(mixed_split) is not bool:
            raise ValueError("experimental_mixed_decode_split must be an explicit boolean")
        cache = getattr(vllm_config, "cache_config", None)
        parallel = getattr(vllm_config, "parallel_config", None)
        self.mixed_decode_split = (
            mixed_split and getattr(cache, "enable_prefix_caching", None) is False
            and getattr(parallel, "prefill_context_parallel_size", None) == 1
            and getattr(parallel, "decode_context_parallel_size", None) == 1)

    def reorder_batch(self, input_batch, scheduler_output):
        return False

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        # Native llm_base_proposer.py:913 calls first draft with its model as
        # positional argument 3; later drafts call build_for_drafting. Mark
        # both, including draft_index=0. The separate experimental suffix
        # qualification never enables native-current for the MTP prefix.
        first_draft = type(fast_build) is not bool
        metadata = from_common(common_attn_metadata, capacity=self.capacity,
                               is_draft=first_draft, current_only=self.current_only,
                               first_draft_current_only=(
                                   first_draft and self.first_draft_current_only),
                               first_draft_current_fia=(first_draft and self.first_draft_current_fia),
                               mixed_decode_split=self.mixed_decode_split)
        return self.bindings.bind(metadata)

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
        metadata = from_common(common_attn_metadata, capacity=self.capacity,
                               is_draft=True, draft_index=draft_index)
        return self.bindings.bind(metadata)
