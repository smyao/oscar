"""Archive #140/#142/#148/#155: restrict the unenabled first-MTP experiment.

Pinned native llm_base_proposer.py:1351-1399 preserves first-pass positions,
hidden states and query intervals only on the ordinary no-extra-slots path.
speculative.py normalizes Qwen3.5 to method=mtp/model_type=qwen3_5_mtp;
eager, nonparallel MTP with PCP/DCP=1 is the audited configuration here.
This is host configuration inspection, never device metadata inference.
"""


def supports_first_draft_current_only(vllm_config) -> bool:
    spec = getattr(vllm_config, "speculative_config", None)
    parallel = getattr(vllm_config, "parallel_config", None)
    draft = getattr(spec, "draft_model_config", None)
    hf_config = getattr(draft, "hf_config", None)
    return (
        getattr(spec, "method", None) == "mtp"
        and getattr(spec, "enforce_eager", None) is True
        and getattr(spec, "parallel_drafting", None) is False
        and getattr(spec, "disable_padded_drafter_batch", None) is False
        and getattr(hf_config, "model_type", None) == "qwen3_5_mtp"
        and getattr(hf_config, "architectures", None) == ["Qwen3_5MTP"]
        and type(getattr(parallel, "prefill_context_parallel_size", None)) is int
        and parallel.prefill_context_parallel_size == 1
        and type(getattr(parallel, "decode_context_parallel_size", None)) is int
        and parallel.decode_context_parallel_size == 1
    )
