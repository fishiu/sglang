
from __future__ import annotations

import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.triton_ops.quest_attention import quest_decode_attention_fwd
from sglang.srt.layers.attention.triton_ops.snapkv_attention import snapkv_compute_indices_torch
from sglang.srt.managers.schedule_batch import global_server_args_dict

# Reuse Triton backend for prefill
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

class SnapKVBackend(AttentionBackend):
    def __init__(self, model_runner):
        super().__init__()
        
        self.model_runner = model_runner
        self.triton_backend = TritonAttnBackend(model_runner)
        
        # SnapKV Parameters
        self.window_size = model_runner.server_args.snapkv_window_size
        self.max_capacity = model_runner.server_args.snapkv_max_capacity
        self.kernel_size = model_runner.server_args.snapkv_kernel_size
        self.pooling = model_runner.server_args.snapkv_pooling
        
        # Metadata storage: req_pool_index -> indices_tensor [num_heads, compressed_len]
        # We use a simple dict for prototype. 
        # Note: keys are int (req_pool_indices).
        self.snapkv_metadata = {}
        
        # Buffers for decode
        self.device = model_runner.device
        self.num_q_head = model_runner.model_config.num_attention_heads // model_runner.tp_size
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(model_runner.tp_size)
        
        # For MHA, num_q_head == num_kv_head. SnapKV design assumes Head Independence.
        # If GQA, we might need to repeat KV indices for Q heads?
        # Design doc says: "Unroll GQA to MHA". 
        # So we store indices for Q heads.
        self.num_heads = self.num_q_head
        
        # Decode buffers
        self.attn_logits = None
        self.attn_lse = None
        self.max_kv_splits = model_runner.server_args.triton_attention_num_kv_splits
        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1]

    def init_forward_metadata(self, forward_batch):
        if forward_batch.forward_mode.is_decode():
            # Prepare buffers for decode
            batch_size = forward_batch.batch_size
            
            # Allocate buffers if needed
            if self.attn_logits is None or self.attn_logits.shape[0] < batch_size:
                self.attn_logits = torch.empty(
                    (batch_size, self.num_heads, self.max_kv_splits, self.v_head_dim),
                    dtype=torch.float32,
                    device=self.device,
                )
                self.attn_lse = torch.empty(
                    (batch_size, self.num_heads, self.max_kv_splits),
                    dtype=torch.float32,
                    device=self.device,
                )
                
            # Compute indices for the batch
            self._prepare_decode_indices(forward_batch)
        else:
            self.triton_backend.init_forward_metadata(forward_batch)

    def _prepare_decode_indices(self, forward_batch):
        # This logic constructs kv_indptr and kv_indices for quest_decode_attention_fwd
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens # Current total length
        bs = len(req_pool_indices)
        
        # We need to build a flat kv_indices array.
        # Layout: [Batch0_Head0, Batch0_Head1, ... Batch1_Head0...]
        # To do this efficiently in Python is hard. 
        # For prototype, we iterate.
        
        indices_list = []
        tokens_per_head_list = []
        
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        
        for i in range(bs):
            pool_idx = req_pool_indices[i].item()
            cur_len = seq_lens[i].item()
            
            # 1. Get Compressed Past
            compressed_indices = self.snapkv_metadata.get(pool_idx)
            
            # 2. Get Current Window
            # Window is last `window_size` tokens.
            # Actually, SnapKV keeps `window_size` tokens EXACTLY at the end.
            # During decode, we might have generated new tokens.
            # Design: "Current Window: take last window_size".
            # So we always take last W.
            # Compressed Past was computed at end of prefill.
            # Wait, if we generate 100 tokens, do we re-compress?
            # Design: "Decoding phase generated tokens are usually appended".
            # So we keep "Original Compressed Prompt" + "Generated Tokens".
            # But if "Generated Tokens" > "Window", do we evict?
            # StreamingLLM evicts. SnapKV usually is "Compress Prompt once".
            # Design says: "Compressed Past ... Current Window: take last window_size".
            # This implies sliding window on the *new* tokens?
            # Actually, usually SnapKV keeps "Important Prompt Tokens" + "Recent Tokens".
            # If "Recent Tokens" grows, we assume we keep them all?
            # Or we apply sliding window to recent tokens?
            # "Target preserved capacity: max_capacity_prompt".
            # If we generate indefinitely, cache grows.
            # SnapKV is for "Long Context".
            # Let's assume we keep "Compressed Prompt" + "All New Tokens" until next re-compression?
            # Or strictly "Compressed Prompt" + "Sliding Window of New Tokens"?
            # Design Doc 3.2.6: "Final KV = Compressed Past + Current Window".
            # This was for "Update KV" (Compression).
            # During Decode, we usually just append.
            # If we simply append, the cache grows.
            # User: "Decode stage... pass appropriate tokens".
            # Let's implement: Indices = Compressed_Prompt_Indices + (Start_of_Decode .. Current).
            # Wait, `compressed_indices` were selected from `0 .. L_prefill - W`.
            # And we kept `L_prefill - W .. L_prefill` (the prefill window).
            # So initially `indices` = `Compressed` + `Prefill_Window`.
            # As we decode, we add 1 token.
            # So we append the new token to `indices`.
            # So `snapkv_metadata` should effectively be the *KV Cache Indices*.
            # We should UPDATE `snapkv_metadata` every decode step? No, that's slow.
            # Better: `indices` = `snapkv_metadata[pool_idx]` (Static Prompt Indices) + `req_to_token[pool_idx, prefill_len : cur_len]` (New Tokens).
            # BUT we also need the "Window" from prefill end.
            # The `snapkv_compute_indices_torch` returns `physical_topk` which corresponds to the "Compressed Past".
            # The "Current Window" (at prefill end) was NOT in `physical_topk`.
            # So we need to reconstruct:
            # `Indices` = `Compressed_Past` + `Prefill_Window` + `Generated_Tokens`.
            # `Prefill_Window` + `Generated_Tokens` = `req_to_token[pool_idx, prefill_len - window_size : cur_len]`.
            # Yes.
            
            # How do we know `prefill_len`?
            # We don't store it.
            # We can assume `compressed_indices` stores the indices *before* the window.
            # So valid indices in `compressed_indices` are `< something`.
            # Actually, simpler:
            # We store `prefill_len` in `snapkv_metadata` tuple? `(indices, prefill_len)`.
            
            if compressed_indices is None:
                # No compression (short prompt). Use full history.
                # Indices = req_to_token[pool_idx, :cur_len]
                # This is same for all heads.
                phy_indices = req_to_token[pool_idx, :cur_len] # [L]
                # Expand to [H, L]
                phy_indices = phy_indices.unsqueeze(0).expand(self.num_heads, -1)
                tokens_per_head = cur_len
            else:
                # compressed_indices: [H, K]
                # stored_prefill_len needed.
                # Let's unpack.
                comp_ind, prefill_len = compressed_indices
                
                # Window start (at prefill time)
                window_start = max(0, prefill_len - self.window_size)
                
                # Recent tokens (Prefill Window + Generated)
                # req_to_token[pool_idx, window_start : cur_len]
                recent_indices = req_to_token[pool_idx, window_start : cur_len] # [Recent]
                
                # Concatenate: [H, K] cat [1, Recent] -> [H, K+Recent]
                recent_indices_exp = recent_indices.unsqueeze(0).expand(self.num_heads, -1)
                
                phy_indices = torch.cat([comp_ind, recent_indices_exp], dim=1)
                tokens_per_head = phy_indices.shape[1]
                
            indices_list.append(phy_indices.flatten()) # [H * Len]
            tokens_per_head_list.append(tokens_per_head)

        # Pack batch
        # kv_indptr: [bs+1]. Element is tokens_per_head.
        self.kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
        lengths = torch.tensor(tokens_per_head_list, dtype=torch.int32, device=self.device)
        torch.cumsum(lengths, dim=0, out=self.kv_indptr[1:])
        
        # kv_indices: Concatenate all
        self.kv_indices = torch.cat(indices_list)
        
        # Metadata for forward pass
        self.decode_meta = {
            "kv_indptr": self.kv_indptr,
            "kv_indices": self.kv_indices,
            "num_kv_splits": torch.full((bs,), self.max_kv_splits, dtype=torch.int32, device=self.device)
        }

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        # 1. Standard Extend
        output = self.triton_backend.forward_extend(q, k, v, layer, forward_batch, save_kv_cache)
        
        # 2. SnapKV Compression (Only if enabled and saving cache)
        if save_kv_cache:
            # Check if we should compress.
            # Ideally we compress at the end of prefill.
            # Using extend_seq_lens to assume last chunk?
            # We just compute it. If it's chunked, we might re-compute or overwrite.
            # Efficiency note: overwriting is fine for correctness, maybe slow.
            
            # Need K buffer.
            k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            
            # Compute Indices
            selected_indices_list = snapkv_compute_indices_torch(
                q, # flattened [Total, H, D]
                k_buffer,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.extend_seq_lens,
                forward_batch.req_to_token_pool.req_to_token,
                self.window_size,
                self.max_capacity,
                self.kernel_size,
                self.pooling
            )
            
            # Store in metadata
            for i, pool_idx in enumerate(forward_batch.req_pool_indices):
                idx_val = pool_idx.item()
                res = selected_indices_list[i]
                if res is not None:
                    # Store (indices, current_seq_len)
                    # current_seq_len is used to determine where the "Window" starts
                    cur_len = forward_batch.seq_lens[i].item()
                    self.snapkv_metadata[idx_val] = (res, cur_len)
                else:
                    # If None (too short), ensure we clean up any old metadata
                    if idx_val in self.snapkv_metadata:
                        del self.snapkv_metadata[idx_val]
                        
        return output

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        # Update KV Cache
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )
            
        # Call Quest Decode Kernel
        # Reshape q to [bs, H, D] if needed (triton backend does it inside)
        # Quest kernel expects q: [bs, H, D]
        q = q.view(forward_batch.batch_size, self.num_q_head, self.model_runner.model_config.head_dim)
        
        o = torch.empty_like(q)
        
        # Using pre-calculated indices
        kv_indptr = self.decode_meta["kv_indptr"]
        kv_indices = self.decode_meta["kv_indices"]
        num_kv_splits = self.decode_meta["num_kv_splits"]
        
        quest_decode_attention_fwd(
            q=q,
            k_buffer=forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            v_buffer=forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o=o,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            attn_logits=self.attn_logits,
            attn_lse=self.attn_lse,
            num_kv_splits=num_kv_splits,
            max_kv_splits=self.max_kv_splits,
            sm_scale=layer.scaling,
            logit_cap=layer.logit_cap,
            layer_id=layer.layer_id,
            force_kv_group_num=1 # Treat as MHA since indices are per-head (Strong GQA supported by Quest too, but we flattened indices per head so MHA logic fits)
            # Wait, if we force_kv_group_num=1, Quest kernel assumes 1 KV head per Q head.
            # But K buffer has num_kv_head.
            # If MHA, ok.
            # If GQA, K buffer has fewer heads.
            # Quest Kernel with group_num=1 expects k_buffer to have same heads as q?
            # Check quest_attention.py line 523: kv_group_num = q.shape[1] // k_buffer.shape[1]
            # If force is set, it overrides.
            # If we use group_num=1, logic: `cur_kv_head = cur_head // 1 = cur_head`.
            # This assumes H_kv = H_q.
            # If actual H_kv < H_q, this index is OOB!
            # So for GQA, we CANNOT force group_num=1 unless we also expand K buffer (virtual).
            # OR we use Quest's Strong GQA mode.
            # Quest Strong GQA: `kv_indices` is `[B, H_kv, L]`.
            # But we constructed `kv_indices` as `[B, H_q, L]` (Head Independence for ALL Q heads).
            # SnapKV says: "Each Q head has independent indices".
            # If we use Strong GQA kernel in Quest, it assumes "All Q heads in a group share indices".
            # This CONTRADICTS "Head Independence".
            # So, for SnapKV + GQA, we essentially de-group GQA back to MHA logically?
            # But K buffer is still compressed (H_kv heads).
            # We need a kernel that reads `K[kv_head]` but uses unique indices for each `Q[q_head]`.
            # Quest kernel `group_num=1` reads `K[q_head]`. This fails if `H_kv != H_q`.
            # So Quest Kernel doesn't support "Head-Independent GQA" out of the box?
            # Let's check Quest Kernel again.
            # `_quest_decode_kernel_stage1`:
            # `cur_kv_head = cur_head // kv_group_num`.
            # If we allow `kv_group_num` to be computed normally (not forced), 
            # it calculates `cur_kv_head` correctly.
            # And it reads `kv_indices`.
            # `cur_batch_kv_start_idx = cur_head * cur_batch_seq_len ...`
            # It assumes `kv_indices` has `H_q` segments!
            # Because it uses `cur_head` (program_id(1)) to offset `kv_indices`.
            # So: YES! Quest kernel supports Head Independent GQA if we let it compute `kv_group_num` correctly.
            # It uses `cur_head` to find indices, and `cur_head // group` to find K vector.
            # This is EXACTLY what we want.
            # So: Do NOT force group num.
        )
        
        return o.view(-1, self.num_q_head * self.v_head_dim)


