import torch

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton


def update_streamingllm_buffer(
    req_to_token: torch.Tensor,
    sink_tokens: int,
    window_tokens: int,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    bs: int,
    device: str,
):
    """
    Build ragged indices for StreamingLLM: union of first S sink tokens and last W tokens per sequence.
    Returns (kv_indptr[int32:(bs+1)], kv_indices[int32:sum_lens], kv_lens[int32:bs]).
    """
    # Compute per-sequence lengths
    S = max(int(sink_tokens), 0)
    W = max(int(window_tokens), 0)
    # sink lengths: min(S, L)
    sink_lens = torch.minimum(seq_lens, torch.tensor(S, dtype=seq_lens.dtype, device=seq_lens.device))
    # tail starts at max(0, L - W), but avoid overlapping with sink [0, S)
    tail_start = torch.maximum(
        torch.zeros_like(seq_lens),
        seq_lens - torch.tensor(W, dtype=seq_lens.dtype, device=seq_lens.device),
    )
    tail_start = torch.maximum(tail_start, sink_lens)
    tail_lens = torch.clamp(seq_lens - tail_start, min=0)

    # Total per-seq lens
    kv_lens = sink_lens + tail_lens
    kv_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
    kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens.to(torch.int32), dim=0)
    kv_indptr = kv_indptr[: bs + 1]

    # Build sink indices if any
    total_sink = int(sink_lens.sum().item())
    sink_indices = None
    if total_sink > 0:
        sink_indices = torch.empty((total_sink,), dtype=torch.int32, device=device)
        sink_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        sink_indptr[1 : bs + 1] = torch.cumsum(sink_lens.to(torch.int32), dim=0)
        sink_indptr = sink_indptr[: bs + 1]
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            sink_lens,
            sink_indptr,
            None,
            sink_indices,
            req_to_token.stride(0),
        )

    # Build tail indices if any
    total_tail = int(tail_lens.sum().item())
    tail_indices = None
    if total_tail > 0:
        tail_indices = torch.empty((total_tail,), dtype=torch.int32, device=device)
        tail_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        tail_indptr[1 : bs + 1] = torch.cumsum(tail_lens.to(torch.int32), dim=0)
        tail_indptr = tail_indptr[: bs + 1]
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            tail_lens,
            tail_indptr,
            tail_start,
            tail_indices,
            req_to_token.stride(0),
        )

    # Merge into final indices per sequence
    kv_indices = torch.empty((int(kv_lens.sum().item()),), dtype=torch.int32, device=device)
    if bs == 0:
        return kv_indptr, kv_indices, kv_lens

    sink_off = 0
    tail_off = 0
    out_off = 0
    for b in range(bs):
        s_len = int(sink_lens[b].item())
        t_len = int(tail_lens[b].item())
        if s_len > 0:
            kv_indices[out_off : out_off + s_len] = sink_indices[sink_off : sink_off + s_len]
            sink_off += s_len
            out_off += s_len
        if t_len > 0:
            kv_indices[out_off : out_off + t_len] = tail_indices[tail_off : tail_off + t_len]
            tail_off += t_len
            out_off += t_len

    return kv_indptr, kv_indices, kv_lens


