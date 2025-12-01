
import torch
import torch.nn.functional as F

def snapkv_compute_indices_torch(
    q,                  # [total_q_tokens, num_heads, head_dim]
    k_buffer,           # [pool_size, num_heads, head_dim]
    req_pool_indices,   # [bs]
    seq_lens,           # [bs] (Total length of prompt)
    extend_seq_lens,    # [bs] (Length of this chunk)
    req_to_token,       # [pool_size, max_len]
    window_size,
    max_capacity,
    kernel_size,
    pooling_mode='avgpool'
):
    """
    Compute SnapKV indices for a batch of requests using PyTorch.
    This is a prototype implementation (Path A).
    """
    # Handle flattened Q: [Total, Hidden] -> [Total, Heads, HeadDim]
    if q.dim() == 2:
        total_tokens, hidden_size = q.shape
        head_dim = k_buffer.shape[-1]
        num_q_heads = hidden_size // head_dim
        q = q.view(total_tokens, num_q_heads, head_dim)

    bs = len(req_pool_indices)
    device = q.device
    
    # Output: List of tensors, one per request. 
    # (Can't easily stack if they have different lengths, but max_capacity is fixed cap)
    # We return a tensor of shape [bs, max_capacity] filled with indices, padded with -1.
    # But actually we want "Compressed Indices" + "Current Window".
    # SnapKV says: Store "Compressed Past".
    # We will return:
    # 1. selected_indices_list: List of [num_selected] tensors.
    
    selected_indices_out = []
    
    # Pointer for q (since it's flattened)
    q_offset = 0
    
    for i in range(bs):
        cur_seq_len = seq_lens[i].item()
        cur_extend_len = extend_seq_lens[i].item()
        pool_idx = req_pool_indices[i].item()
        
        # Slice Q for this request
        q_req = q[q_offset : q_offset + cur_extend_len] # [ext_len, H, D]
        q_offset += cur_extend_len
        
        # If prompt is short, no compression needed
        if cur_seq_len <= max_capacity:
            selected_indices_out.append(None) # Marker for "No compression"
            continue
            
        # 1. Prepare Observation Window Query
        # We need the LAST window_size tokens of the prompt.
        # Q: does q_req contain them?
        # If cur_extend_len >= window_size, yes.
        # If not, we might be missing some Qs. 
        # Assumption: Chunk size >> window_size (64).
        if cur_extend_len < window_size:
            # Fallback: just use whatever Q we have. 
            # (In production we might need to cache Q or ensure chunk size)
            q_window = q_req
        else:
            q_window = q_req[-window_size:]
            
        # [W, H, D] -> [H, W, D]
        q_window = q_window.transpose(0, 1) 
        
        # 2. Gather K (History)
        # We need K for the *entire* prompt up to (L - window_size).
        # SnapKV: "Calculate attention of Window vs All History".
        # Actually usually "History" means "Before Window". 
        # Design doc: "Compressed Past" (TopK) + "Current Window" (Keep All).
        # So we compress [0, L-W).
        
        valid_past_len = cur_seq_len - window_size
        if valid_past_len <= 0:
             selected_indices_out.append(None)
             continue
             
        # Get physical indices for past tokens
        # req_to_token: [pool_size, max_len]
        past_indices = req_to_token[pool_idx, :valid_past_len] # [valid_past_len]
        
        # Gather K: [valid_past_len, H, D]
        # This is the expensive part in Python
        k_past = k_buffer[past_indices] 
        
        # [valid_past_len, H, D] -> [H, D, valid_past_len]
        k_past = k_past.permute(1, 2, 0)
        
        # 3. Compute Attention Score
        # Attn = Q @ K.T / sqrt(D)
        # [H, W, D] @ [H, D, Past] -> [H, W, Past]
        scale = 1.0 / (q_window.shape[-1] ** 0.5)
        attn_score = torch.matmul(q_window, k_past) * scale
        
        # 4. Aggregate Importance (Sum over Window)
        # [H, W, Past] -> [H, Past]
        importance = attn_score.sum(dim=1)
        
        # 5. Pooling
        # Pool 1D over the 'Past' dimension.
        # shape: [H, Past] -> [H, 1, Past] for pool1d
        importance = importance.unsqueeze(1)
        padding = kernel_size // 2
        if pooling_mode == 'maxpool':
            importance = F.max_pool1d(importance, kernel_size=kernel_size, padding=padding, stride=1)
        else:
            importance = F.avg_pool1d(importance, kernel_size=kernel_size, padding=padding, stride=1)
        importance = importance.squeeze(1) # [H, Past]
        
        # Handle shape mismatch due to padding (sometimes pool output size != input size)
        if importance.shape[-1] != valid_past_len:
            importance = importance[:, :valid_past_len]
            
        # 6. TopK Selection
        # We want to keep (max_capacity - window_size) tokens.
        k_select = max_capacity - window_size
        if k_select >= valid_past_len:
            # Keep all
            selected_indices_out.append(None)
            continue
            
        # Select indices per head?
        # Design doc: "Head Independence... Head 0 keeps 100-200, Head 1 keeps 300-400".
        # So we get [H, k_select] indices.
        topk_indices = torch.topk(importance, k=k_select, dim=-1).indices # [H, k_select]
        
        # We need to map these logical indices (0..valid_past_len) back to physical indices.
        # past_indices: [valid_past_len]
        # We gather from past_indices.
        # [H, k_select]
        
        # Store logical indices or physical?
        # If we store physical, we can just use them in decode.
        # But past_indices is 1D. We select different subset for each head.
        # So result is [H, k_select] physical indices.
        
        # past_indices is on device.
        # topk_indices is on device.
        # Gather:
        # physical_topk[h, i] = past_indices[topk_indices[h, i]]
        
        # Broadcast past_indices: [1, valid_past_len]
        # Gather dim 1.
        physical_topk = past_indices.unsqueeze(0).expand(k_buffer.shape[1], -1).gather(1, topk_indices)
        
        selected_indices_out.append(physical_topk) # [H, k_select]
        
    return selected_indices_out


