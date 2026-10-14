"""Fixed-shape greedy sampling for the external OSCAR integration.

Archive #157-160: an invalid accepted count can move the later-MTP anchor
outside its request. Native reject_sample.py:91-108 consumes unspecified
masked loads in an unguarded token loop. This adapter removes that source
risk; it does not establish the cause of either reported target failure.

D.4 comparison: this is sampling metadata before prepare, not dequant/FIA.
It never reads KV history and uses O(batch * (max_spec_len + 1)) tensors.
The historical 725 ms host prepare and 6.5 s full-history dequant motivate
fixed device shapes with no host tensor reads; no NPU timing is claimed.
"""


def rejection_greedy_sample_with_triton(
    output_token_ids,
    num_draft_tokens,
    cu_num_draft_tokens,
    draft_token_ids,
    target_argmax,
    bonus_token_ids,
    is_greedy,
    max_spec_len,
    grid,
    block_size,
):
    """Keep the native call signature and mutate only greedy output rows.

    All tensor work stays on the input device. The native sampler owns the
    probabilities, target argmax and random writer; this function implements
    only its greedy prefix/first-rejection/bonus rule. Tests may supply CPU
    tensors to this same implementation, without a production CPU route.
    """
    import torch

    batch_size, width = output_token_ids.shape
    if width != max_spec_len + 1 or len(num_draft_tokens) != batch_size:
        raise ValueError("OSCAR greedy sampling requires native batch/output shapes")
    if cu_num_draft_tokens.shape != (batch_size,):
        raise ValueError("OSCAR greedy sampling requires one cumulative draft count per request")
    if draft_token_ids.shape != target_argmax.shape or draft_token_ids.ndim != 1:
        raise ValueError("OSCAR greedy sampling requires aligned flat draft and target IDs")
    if batch_size == 0:
        return

    device = output_token_ids.device
    columns = torch.arange(width, dtype=torch.int64, device=device).unsqueeze(0)
    greedy_rows = (torch.ones((batch_size, 1), dtype=torch.bool, device=device)
                   if is_greedy is None else (is_greedy != 0).reshape(batch_size, 1))
    bonuses = bonus_token_ids.reshape(batch_size, 1).to(output_token_ids.dtype)
    empty = torch.full_like(output_token_ids, -1)

    # No gather is valid for an empty flat draft tensor. This is a shape-only
    # branch: every request has zero drafts and its greedy output is its bonus.
    if draft_token_ids.shape[0] == 0:
        selected = torch.where(columns == 0, bonuses, empty)
    else:
        ends = cu_num_draft_tokens.to(dtype=torch.int64)
        starts = torch.cat((ends.new_zeros((1,)), ends[:-1]))
        counts = (ends - starts).unsqueeze(1)
        indices = (starts.unsqueeze(1) + columns).clamp(0, draft_token_ids.shape[0] - 1)
        drafts = torch.gather(draft_token_ids, 0, indices.reshape(-1)).reshape(batch_size, width)
        targets = torch.gather(target_argmax, 0, indices.reshape(-1)).reshape(batch_size, width)
        in_draft = columns < counts
        mismatch = in_draft & (drafts != targets)
        first_mismatch = torch.where(mismatch, columns, counts).amin(dim=1, keepdim=True)
        # The first rejected position still emits the target's argmax.
        copy_prefix = in_draft & (columns <= first_mismatch)
        selected = torch.where(copy_prefix, targets.to(output_token_ids.dtype), empty)
        needs_bonus = (first_mismatch == counts) & (columns == counts)
        selected = torch.where(needs_bonus, bonuses, selected)

    # Fixed B x W copy: random rows retain their exact original contents;
    # greedy rows include an explicit -1 tail after rejection or the bonus.
    output_token_ids.copy_(torch.where(greedy_rows, selected, output_token_ids))


def rejection_sample_with_guarded_cu(
    original,
    draft_token_ids,
    num_draft_tokens,
    max_spec_len,
    cu_num_draft_tokens,
    draft_probs,
    target_logits_or_tuple,
    bonus_token_ids,
    sampling_metadata,
    synthetic_mode=False,
    synthetic_conditional_rates=None,
    ori_target_logits=None,
):
    """Give native first-row cu[-1] loads valid backing without changing values.

    Native ordinary greedy/random kernels evaluate that load before choosing
    a zero start index. A private contiguous view preserves the logical cu
    tensor and all sampling arguments, with a valid zero immediately before
    its data pointer. A 32-byte prefix preserves metadata pointer alignment.
    The native random kernels/probabilities are unchanged.
    """
    import torch

    if cu_num_draft_tokens.ndim != 1 or cu_num_draft_tokens.stride() != (1,):
        raise ValueError("OSCAR guarded sampling requires native contiguous cumulative counts")
    guard_elements = 32 // cu_num_draft_tokens.element_size()
    backing = torch.cat((cu_num_draft_tokens.new_zeros((guard_elements,)), cu_num_draft_tokens))
    guarded_cu = backing[guard_elements:]
    return original(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens,
        max_spec_len=max_spec_len,
        cu_num_draft_tokens=guarded_cu,
        draft_probs=draft_probs,
        target_logits_or_tuple=target_logits_or_tuple,
        bonus_token_ids=bonus_token_ids,
        sampling_metadata=sampling_metadata,
        synthetic_mode=synthetic_mode,
        synthetic_conditional_rates=synthetic_conditional_rates,
        ori_target_logits=ori_target_logits,
    )
