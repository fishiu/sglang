"""
Quest Attention Triton Kernels

实现 Query-Aware Sparsity 的三阶段稀疏注意力：
1. Estimate: 用 page-level 元数据估算注意力得分上界
2. TopK Selection: 选择最重要的 K 个 pages（Python 实现）
3. Sparse Attention: 对选中的 pages 执行完整注意力

核心设计：
- Page 粒度的元数据（min/max）用于快速估算
- Last page 强制保留（最近生成的 tokens）
- 懒初始化：decode 第一步从 KV cache 回读计算元数据
"""

import os
import torch
import triton
import triton.language as tl
import torch.cuda.nvtx as nvtx


@triton.jit
def tanh(x):
    """Tanh activation using sigmoid"""
    return 2 * tl.sigmoid(2 * x) - 1


# Quest 专用的 decode kernel（支持 per-head kv_indptr）


@triton.jit
def _quest_decode_kernel_stage1(
    Q,                  # [batch, num_heads, head_dim]
    K_Buffer,           # [size, num_heads, head_dim]
    V_Buffer,           # [size, num_heads, v_head_dim]
    sm_scale,           # float
    kv_indptr,          # [batch + 1] - 每个 batch 的 token 数量（所有 head 相同）
    kv_indices,         # [total_selected_tokens] - flat token indices
    Att_Out,            # [batch, num_heads, max_kv_splits, v_head_dim]
    Att_Lse,            # [batch, num_heads, max_kv_splits]
    num_kv_splits,      # [batch]
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,  # 1
    BLOCK_DMODEL: tl.constexpr,  # 128
    BLOCK_DV: tl.constexpr,  # 128
    BLOCK_N: tl.constexpr,  # 64
    MIN_BLOCK_KV: tl.constexpr,  # 16
    logit_cap: tl.constexpr,  # 0.0
    Lk: tl.constexpr,  # 128
    Lv: tl.constexpr,  # 128
    head_num: tl.constexpr,  # num_heads, for debug
    MAX_KV_SPLITS: tl.constexpr,  # max_kv_splits, for debug
):
    """
    Quest Decode Stage1: 简化版（所有 head 选中相同数量的 tokens）
    
    与标准 decode kernel 的主要区别：
    - kv_indptr 无 head 维度：[batch+1]，所有 head 共享相同的 token 数量
    - 每个 head 从 kv_indices[cur_head * tokens_per_head] 开始读取
    """
    cur_batch = tl.program_id(0)  # 0
    cur_head = tl.program_id(1)  # 0
    split_kv_id = tl.program_id(2)  # 0
    
    cur_kv_head = cur_head // kv_group_num  # 0 // 1 = 0
    
    offs_d = tl.arange(0, BLOCK_DMODEL)  # 0 ... 127
    offs_dv = tl.arange(0, BLOCK_DV)  # 0 ... 127
    mask_d = offs_d < Lk  # true, true, ..., true
    mask_dv = offs_dv < Lv # true, true, ..., true
    
    # 从 kv_indptr 获取当前 batch 的 token 数量（所有 head 相同）
    # kv_indptr: [batch+1]
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)
    
    # 当前 batch 在 kv_indices 中的起始偏移（按每个 batch 的 token 数乘以 head 数）
    # 注意：kv_indptr 单位是“每 head 的 token 数”，因此需要乘以 head_num 才能得到跨 head 的总偏移
    cur_batch_base_tokens = tl.load(kv_indptr + cur_batch)
    cur_batch_kv_start_idx = cur_head * cur_batch_seq_len + cur_batch_base_tokens * head_num
    
    kv_splits = tl.load(num_kv_splits + cur_batch)
    
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    
    # 计算当前 split 的范围
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )  # div(div(257, 8), 16)*16=48 很浪费就差一点点就32了
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    
    # 初始化 online softmax 变量
    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
    
    if split_kv_end > split_kv_start:
        q = tl.load(Q + off_q, mask=mask_d, other=0.0)
        
        # 遍历当前 split 的 tokens
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            
            # 从 kv_indices 加载实际 token 位置
            # 注意：kv_indices 的索引需要加上 cur_batch_kv_start_idx（该 head 的起始偏移）
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            
            # 加载 K 和计算 QK
            offs_buf_k = (
                kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0.0,
            )
            
            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale
            
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)
            
            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))
            
            # 加载 V 和累积
            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )
            
            # Online softmax
            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)
            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max
        
        # 存储中间结果
        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )
        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=mask_dv,
        )
        
        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv
        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


@triton.jit
def _quest_decode_grouped_kernel_stage1(
    Q,                  # [batch, num_q_heads, head_dim]
    K_Buffer,           # [size, num_kv_heads, head_dim]
    V_Buffer,           # [size, num_kv_heads, v_head_dim]
    sm_scale,
    kv_indptr,          # [batch + 1]
    kv_indices,         # [total_selected_tokens]
    Att_Out,            # [batch, num_q_heads, max_kv_splits, v_head_dim]
    Att_Lse,            # [batch, num_q_heads, max_kv_splits]
    num_kv_splits,      # [batch]
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,        # = num_q_heads // num_kv_heads
    num_q_heads: tl.constexpr,         # total Q heads (Hq)
    num_kv_heads: tl.constexpr,        # KV heads (Hkv)
    BLOCK_DMODEL: tl.constexpr,   # padded head_dim (K dimension for QK)
    BLOCK_DV: tl.constexpr,       # padded v_head_dim (N dimension for PV)
    BLOCK_N: tl.constexpr,        # tokens per iteration in sequence dim
    BLOCK_H: tl.constexpr,        # max Q heads per CTA tile
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    """Grouped Quest decode stage1.

    Each program (CTA) handles a *tile* of query heads that belong to some KV
    head, for a specific (batch, split) pair.

    Head tiling strategy (mirrors decode_attention._fwd_grouped_kernel_stage1):
    - Global Q heads:        0 .. num_q_heads-1
    - GQA group size:        kv_group_num (H_q / H_kv)
    - Per KV group CTA tile: BLOCK_H (<= kv_group_num or equal to it)
    - Grid dim1 (head blocks): cdiv(num_q_heads, min(BLOCK_H, kv_group_num))

    Inside the kernel:
    - We infer KV head index from head-block id and kv_group_num/BLOCK_H.
    - For that KV head, we compute attention for up to BLOCK_H Q heads in its
      group, using masks to skip out-of-range heads.

    Tensor shapes (logical view):
    - Q:        [B, Hq, D]          (query vectors)
    - K_Buffer: [T, Hkv, D]         (key cache, shared across groups)
    - V_Buffer: [T, Hkv, Dv]        (value cache)
    - kv_indptr: [B + 1]            (per-batch token counts per KV head)
    - kv_indices: flattened indices over [B, Hkv, L_b] for the selected tokens.
    - Att_Out: [B, Hq, S, Dv]       (stage1 partial outputs)
    - Att_Lse: [B, Hq, S]           (per-split log-sum-exp)

    Compared to the initial Quest grouped kernel, this version mirrors
    decode_attention._fwd_grouped_kernel_stage1 more closely:
    - K and V are loaded in [D, N] / [N, Dv] layouts suitable for tl.dot
    - QK and P*V use tl.dot so that Triton can map them to matrix instructions
    - head dimension is blocked with BLOCK_H and masked at the tail
    """
    # Program ids:
    # - pid(0): batch index b
    # - pid(1): "head block id" (not directly the KV head id)
    # - pid(2): split index s over the selected tokens
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    # Derive KV head id from the head-block id.
    # When kv_group_num <= BLOCK_H: cdiv(kv_group_num, BLOCK_H) == 1,
    # so cur_kv_head == cur_head_id (one CTA per KV head).
    # When kv_group_num  > BLOCK_H: a KV group spans multiple head blocks.
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    # Effective number of Q heads per block (may be smaller than BLOCK_H
    # when kv_group_num < BLOCK_H).
    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num

    # Compute the global Q head indices covered by this head block.
    # cur_head: [BLOCK_H], but only the first VALID_BLOCK_H entries belong
    # to this KV group; the rest are masked out.
    offs_h = tl.arange(0, BLOCK_H)
    cur_head = cur_head_id * VALID_BLOCK_H + offs_h
    # Mask heads that are outside this block's valid range or beyond Hq.
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < num_q_heads)

    # Head / value dimensions
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    # Per-batch token range (shared by all heads within this batch)
    cur_batch_token_start = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_token_start
    kv_splits = tl.load(num_kv_splits + cur_batch)

    # Offsets into Q:
    # offs_q has shape [BLOCK_H, BLOCK_DMODEL] and indexes Q[b, cur_head, :]
    offs_q = (
        cur_batch * stride_qbs
        + cur_head[:, None] * stride_qh
        + offs_d[None, :]
    )

    # Split range on the selected tokens of this batch
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    # Online softmax state per Q head in this group
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        # Load Q: [BLOCK_H, BLOCK_DMODEL]
        q = tl.load(
            Q + offs_q,
            mask=(mask_h[:, None]) & (mask_d[None, :]),
            other=0.0,
        )

        # Per-KV-head CSR layout:
        # kv_indices is flattened over [B, Hkv, L_b], where L_b =
        # kv_indptr[b+1] - kv_indptr[b] is the number of selected tokens per
        # KV head in batch b. Tokens for (b, kv_head) occupy:
        #   [base_tokens_before * Hkv + kv_head * L_b ... + L_b)
        # where base_tokens_before = kv_indptr[b].
        base_tokens_before = cur_batch_token_start  # starting offset (per-batch)
        base = base_tokens_before * num_kv_heads + cur_kv_head * cur_batch_seq_len

        # Iterate over tokens in this split with BLOCK_N granularity
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)

            # Load logical token locations (global token ids in KV cache)
            kv_loc = tl.load(
                kv_indices + base + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )

            # Load K as [BLOCK_DMODEL, BLOCK_N]:
            # - first dim: head_dim (K)
            # - second dim: token index within this split (N)
            # This layout matches tl.dot([H, K], [K, N]) -> [H, N].
            offs_buf_k = (
                kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )

            # QK: [BLOCK_H, BLOCK_N]
            qk = tl.dot(q, k.to(q.dtype))
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            # Mask out:
            # - inactive heads (mask_h)
            # - tokens beyond split_kv_end
            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end),
                qk,
                float("-inf"),
            )

            # Load V as [BLOCK_N, BLOCK_DV]
            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            # Online softmax update across the N dimension
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)  # [BLOCK_H]
            re_scale = tl.exp(e_max - n_e_max)          # [BLOCK_H]
            p = tl.exp(qk - n_e_max[:, None])           # [BLOCK_H, BLOCK_N]

            acc *= re_scale[:, None]
            # p: [BLOCK_H, BLOCK_N], v: [BLOCK_N, BLOCK_DV]
            acc += tl.dot(p.to(v.dtype), v)            # [BLOCK_H, BLOCK_DV]

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        # Store per-split partial outputs
        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        # Store per-split LSE for later Stage2 merge
        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


@triton.jit
def _quest_decode_kernel_stage2(
    Mid_O,          # [batch, num_heads, max_kv_splits, v_head_dim]
    Mid_O_1,        # [batch, num_heads, max_kv_splits]
    O,              # [batch, num_heads, v_head_dim]
    kv_indptr,      # [batch + 1]
    num_kv_splits,  # [batch]
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    """
    Quest Decode Stage2: 聚合 stage1 的结果
    
    简化版：kv_indptr 无 head 维度，所有 head 共享相同的 token 数量
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    
    # 从 kv_indptr 获取序列长度（所有 head 相同）
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)
    kv_splits = tl.load(num_kv_splits + cur_batch)
    
    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv
    
    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
    
    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv
    
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    
    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
        
        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os,
                mask=mask_d,
                other=0.0,
            )
            tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)
            
            n_e_max = tl.maximum(tlogic, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv
            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max
    
    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def quest_decode_attention_fwd(
    q,              # [batch, num_q_heads, head_dim]
    k_buffer,       # [size, num_kv_heads, head_dim]
    v_buffer,       # [size, num_kv_heads, v_head_dim]
    o,              # [batch, num_q_heads, v_head_dim]
    kv_indptr,      # [batch + 1] - 所有 head 共享相同的 token 数量
    kv_indices,     # [total_selected_tokens]
    attn_logits,    # [batch, num_q_heads, max_kv_splits, v_head_dim]
    attn_lse,       # [batch, num_q_heads, max_kv_splits]
    num_kv_splits,  # [batch]
    max_kv_splits,  # int
    sm_scale,       # float
    logit_cap=0.0,
    layer_id=None,
    force_kv_group_num: int | None = None,
):
    """
    Quest Decode Attention Forward (两阶段)
    
    简化版：
    - kv_indptr 无 head 维度：[batch+1]
    - 所有 head 选中相同数量的 tokens（Quest TopK）
    """
    BLOCK = 64
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]
    
    batch, head_num = q.shape[0], q.shape[1]
    MAX_KV_SPLITS = max_kv_splits

    # Stage1: 计算部分结果
    kv_group_num = q.shape[1] // k_buffer.shape[1]
    if force_kv_group_num is not None:
        kv_group_num = force_kv_group_num

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    if kv_group_num == 1:
        # MHA / weak GQA: 每个 Q head 独立计算（与原始 Quest 行为一致）
        grid = (batch, head_num, MAX_KV_SPLITS)
        num_warps = 4

        if layer_id == 0:
            nvtx.range_push("quest_attnl0")

        _quest_decode_kernel_stage1[grid](
            q,
            k_buffer,
            v_buffer,
            sm_scale,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            q.stride(0),
            q.stride(1),
            k_buffer.stride(0),
            k_buffer.stride(1),
            v_buffer.stride(0),
            v_buffer.stride(1),
            attn_logits.stride(0),
            attn_logits.stride(1),
            attn_logits.stride(2),
            kv_group_num=kv_group_num,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DV=BLOCK_DV,
            BLOCK_N=BLOCK,
            MIN_BLOCK_KV=32,
            logit_cap=logit_cap,
            num_warps=num_warps,
            num_stages=2,
            Lk=Lk,
            Lv=Lv,
            head_num=head_num,
            MAX_KV_SPLITS=MAX_KV_SPLITS,
        )

        if layer_id == 0:
            nvtx.range_pop()
    else:
        # Strong GQA: grouped kernel，一次处理一个 KV head 对应的 kv_group_num 个 Q heads。
        # 我们按 decode_attention._decode_grouped_att_m_fwd 的方式在 head 维做 tile：
        # - head_num = Hq
        # - BLOCK_H = 16 Q heads per CTA
        # - grid_y = ceil(Hq / min(BLOCK_H, kv_group_num))
        # CSR layout 在强 GQA 下是 per-KV-head 的：kv_indices 展开为 [B, Hkv, L_b]。
        num_kv_heads = k_buffer.shape[1]
        BLOCK_H = 16
        grid = (
            batch,
            triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
            MAX_KV_SPLITS,
        )

        _quest_decode_grouped_kernel_stage1[grid](
            q,
            k_buffer,
            v_buffer,
            sm_scale,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            q.stride(0),
            q.stride(1),
            k_buffer.stride(0),
            k_buffer.stride(1),
            v_buffer.stride(0),
            v_buffer.stride(1),
            attn_logits.stride(0),
            attn_logits.stride(1),
            attn_logits.stride(2),
            kv_group_num=kv_group_num,
            num_q_heads=head_num,
            num_kv_heads=num_kv_heads,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DV=BLOCK_DV,
            BLOCK_N=BLOCK,
            BLOCK_H=BLOCK_H,
            MIN_BLOCK_KV=32,
            logit_cap=logit_cap,
            Lk=Lk,
            Lv=Lv,
        )
    
    # Stage2: 聚合结果
    grid = (batch, head_num)
    _quest_decode_kernel_stage2[grid](
        attn_logits,
        attn_lse,
        o,
        kv_indptr,
        num_kv_splits,
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_KV_SPLITS=MAX_KV_SPLITS,
        MIN_BLOCK_KV=32,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
    )


@triton.jit
def quest_estimate_kernel(
    Q,                      # [batch, num_heads, head_dim]
    K_metadata,             # [num_pages, num_heads, head_dim, 2]
    Seq_lens,               # [batch]
    Estimated_scores,       # 输出: [batch, num_heads, max_pages]
    Req_to_token,           # [batch, max_context_len]
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_mp,
    stride_mh,
    stride_md,
    stride_sb,
    stride_sh,
    stride_rb,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Estimate 阶段：用元数据快速估算每个 page 的注意力得分上界
    
    Grid: (batch, num_heads, max_pages-1)  # 排除 last page
    
    算法：
        对于每个维度 d，选择能产生最大乘积的 K 边界：
        - 若 Q[d] > 0，选 K_max[d]（正数乘大数）
        - 若 Q[d] < 0，选 K_min[d]（负数乘小数得大值）
        score = Σ_d max(Q[d]*K_min[d], Q[d]*K_max[d])
    """
    # 获取当前 thread block 的索引
    bid = tl.program_id(0)  # batch index
    hid = tl.program_id(1)  # head index
    pid = tl.program_id(2)  # page index in batch
    
    # 获取当前 batch 的序列长度
    seq_len = tl.load(Seq_lens + bid)
    num_pages = (seq_len + page_size - 1) // page_size
    
    # 过滤：只处理 [0, num_pages-2]，排除 last page
    if pid >= num_pages - 1:
        return
    
    # 计算全局 page 索引：通过 req_to_token 映射
    token_offset_in_seq = pid * page_size
    global_token_idx = tl.load(Req_to_token + bid * stride_rb + token_offset_in_seq)
    global_page_idx = global_token_idx // page_size
    
    # 加载 Q 向量: [head_dim]
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < head_dim
    q = tl.load(
        Q + bid * stride_qb + hid * stride_qh + offs_d * stride_qd,
        mask=mask_d,
        other=0.0,
    )
    
    # 加载元数据: K_min 和 K_max
    # K_metadata[page, head, d, 0=min/1=max]
    # stride_md 是 head_dim 维度的 stride (=2)，已包含最后一维 [min,max] 的信息
    # 关键：使用 global_page_idx 而非 pid
    k_min = tl.load(
        K_metadata + global_page_idx * stride_mp + hid * stride_mh + offs_d * stride_md + 0,
        mask=mask_d,
        other=0.0,
    )
    k_max = tl.load(
        K_metadata + global_page_idx * stride_mp + hid * stride_mh + offs_d * stride_md + 1,
        mask=mask_d,
        other=0.0,
    )
    
    # 计算得分：Σ_d max(q*k_min, q*k_max)
    qk_min = q * k_min
    qk_max = q * k_max
    score_per_dim = tl.maximum(qk_min, qk_max)
    score = tl.sum(score_per_dim, axis=0)
    
    # 存储得分
    tl.store(
        Estimated_scores + bid * stride_sb + hid * stride_sh + pid,
        score,
    )


@triton.jit
def quest_estimate_split_kernel(
    Q,                      # [batch, num_q_heads, head_dim]
    K_metadata,             # [num_pages, num_kv_heads, head_dim, 2]
    Seq_lens,               # [batch]
    Estimated_scores,       # Output: [batch, num_q_heads, max_pages]
    Req_to_token,           # [batch, max_context_len]
    Req_pool_indices,       # [batch] mapping to rows in Req_to_token pool
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_mp,
    stride_mh,
    stride_md,
    stride_sb,
    stride_sh,
    stride_rb,
    stride_rp,
    BLOCK_DMODEL: tl.constexpr,
    NUM_SPLITS: tl.constexpr,     # 固定 splits 数量（如 8/16/32）
    kv_group_num: tl.constexpr,   # = num_q_heads // num_kv_heads
):
    """
    Estimate（split 版本）：将原先按 page 维展开的网格，改为固定的 splits 维度。

    网格：grid = (batch, num_q_heads, NUM_SPLITS)
    - 每个 split 负责一段 page 区间：[split_start, split_end)
    - 仅计算非 last page（保持与旧逻辑一致）

    张量形状：
    - Q:                [B, Hq, D]
    - K_metadata:       [num_pages_global, Hkv, D, 2]
    - Seq_lens:         [B]
    - Estimated_scores: [B, H, max_pages]
    - Req_to_token:     [B, max_context_len]
    """
    bid = tl.program_id(0)  # batch index
    hid = tl.program_id(1)  # q_head index
    sid = tl.program_id(2)  # split index

    # 当前 batch 的页数（向上取整）
    seq_len = tl.load(Seq_lens + bid)
    num_pages = (seq_len + page_size - 1) // page_size
    valid_pages = tl.maximum(num_pages - 1, 0)  # 排除 last page

    # 计算该 split 的 page 范围
    pages_per_split = tl.cdiv(valid_pages, NUM_SPLITS)
    split_start = pages_per_split * sid
    split_end = tl.minimum(split_start + pages_per_split, valid_pages)

    # 载入 Q 向量
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < head_dim
    q = tl.load(
        Q + bid * stride_qb + hid * stride_qh + offs_d * stride_qd,
        mask=mask_d,
        other=0.0,
    )

    # 遍历该 split 覆盖的所有 page
    for pid in range(split_start, split_end):
        # 通过 req_to_token 获得全局 page id
        token_offset_in_seq = pid * page_size
        pool_row = tl.load(Req_pool_indices + bid)
        global_token_idx = tl.load(
            Req_to_token + pool_row * stride_rb + token_offset_in_seq
        )
        global_page_idx = global_token_idx // page_size

        # 加载该 page 的 metadata（注意 GQA：Q 头映射到 KV 头）
        kv_hid = hid // kv_group_num
        k_min = tl.load(
            K_metadata + global_page_idx * stride_mp + kv_hid * stride_mh + offs_d * stride_md + 0,
            mask=mask_d,
            other=0.0,
        )
        k_max = tl.load(
            K_metadata + global_page_idx * stride_mp + kv_hid * stride_mh + offs_d * stride_md + 1,
            mask=mask_d,
            other=0.0,
        )

        # 计算 Σ_d max(q*d_min, q*d_max)
        qk_min = q * k_min
        qk_max = q * k_max
        score_per_dim = tl.maximum(qk_min, qk_max)
        score = tl.sum(score_per_dim, axis=0)

        # 写回（last page 保持为 0，因此我们只写 [0, valid_pages-1]）
        tl.store(
            Estimated_scores + bid * stride_sb + hid * stride_sh + pid,
            score,
        )


@triton.jit
def quest_estimate_split_kernel_grouped(
    Q,                      # [batch, num_q_heads, head_dim]
    K_metadata,             # [num_pages, num_kv_heads, head_dim, 2]
    Seq_lens,               # [batch]
    Estimated_scores,       # Output: [batch, num_kv_heads, max_pages]
    Req_to_token,           # [max_pool, max_context_len]
    Req_pool_indices,       # [batch] mapping to rows in Req_to_token pool
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_mp,
    stride_mh,
    stride_md,
    stride_sb,
    stride_sh,
    stride_rb,
    stride_rp,
    BLOCK_DMODEL: tl.constexpr,
    NUM_SPLITS: tl.constexpr,     # 固定 splits 数量（如 8/16/32）
    kv_group_num: tl.constexpr,   # = num_q_heads // num_kv_heads
):
    """
    Strong-GQA Estimate（split 版本，按 KV 组聚合）。

    网格：grid = (batch, num_kv_heads, NUM_SPLITS)
    - 每个 program 处理一个 (batch=b, kv_head=h_kv, split=s) 组合。
    - 内部遍历该 KV 组下的所有 Q heads（大小 = kv_group_num），对 score 做
      group-reduction：
          score_group(page) = Σ_{h_q in group} score(q_{h_q}, K_page)
    - 然后将 group score 写回到 Estimated_scores 中组内所有 Q heads 的槽位：
          Estimated_scores[b, h_q, page] = score_group(page)

    张量形状：
    - Q:                [B, Hq, D]
    - K_metadata:       [num_pages_global, Hkv, D, 2]
    - Seq_lens:         [B]
    - Estimated_scores: [B, Hkv, max_pages]  （按 KV head 存储，每个 KV 组一条曲线）
    - Req_to_token:     [B, max_context_len]

    注意：
    - 这里假设 GQA 配置满足 Hq = Hkv * kv_group_num。
    - 为了避免在 Python 侧再做一次 group-reduction，这里直接在 kernel
      内对 group 维度求和，并同时广播回所有组内 Q heads。
    """
    bid = tl.program_id(0)  # batch index
    kv_hid = tl.program_id(1)  # kv_head index (group id)
    sid = tl.program_id(2)  # split index

    # 当前 batch 的页数（向上取整）
    seq_len = tl.load(Seq_lens + bid)
    num_pages = (seq_len + page_size - 1) // page_size
    valid_pages = tl.maximum(num_pages - 1, 0)  # 排除 last page

    # 计算该 split 的 page 范围
    pages_per_split = tl.cdiv(valid_pages, NUM_SPLITS)
    split_start = pages_per_split * sid
    split_end = tl.minimum(split_start + pages_per_split, valid_pages)

    # 预备 head_dim 维度的 offsets/mask
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < head_dim

    # 遍历该 split 覆盖的所有 page（仅 [0, valid_pages-1]）
    for pid in range(split_start, split_end):
        # 通过 req_to_token 获得全局 page id
        token_offset_in_seq = pid * page_size
        pool_row = tl.load(Req_pool_indices + bid)
        global_token_idx = tl.load(
            Req_to_token + pool_row * stride_rb + token_offset_in_seq
        )
        global_page_idx = global_token_idx // page_size

        # 加载该 page 的 metadata（按 KV head 维度）
        k_min = tl.load(
            K_metadata
            + global_page_idx * stride_mp
            + kv_hid * stride_mh
            + offs_d * stride_md
            + 0,
            mask=mask_d,
            other=0.0,
        )
        k_max = tl.load(
            K_metadata
            + global_page_idx * stride_mp
            + kv_hid * stride_mh
            + offs_d * stride_md
            + 1,
            mask=mask_d,
            other=0.0,
        )

        # 对该 KV 组内的所有 Q heads 累加 score
        group_score = tl.zeros((), dtype=tl.float32)

        # kv_group_num 是 constexpr，循环会被 Triton 展开：
        # 对于每个组内 Q head：
        for g in range(kv_group_num):
            # 该组内 Q head 的全局索引 h_q
            hq = kv_hid * kv_group_num + g

            # 加载该 Q head 的向量 q[b, h_q, :]
            q = tl.load(
                Q + bid * stride_qb + hq * stride_qh + offs_d * stride_qd,
                mask=mask_d,
                other=0.0,
            )

            # Σ_d max(q_d * k_min_d, q_d * k_max_d)
            qk_min = q * k_min
            qk_max = q * k_max
            score_per_dim = tl.maximum(qk_min, qk_max)
            score_i = tl.sum(score_per_dim, axis=0)

            group_score += score_i

        # 将 group_score 写回到该 KV 组的 Estimated_scores 槽位（不再按 Q head 扩展）
        tl.store(
            Estimated_scores + bid * stride_sb + kv_hid * stride_sh + pid,
            group_score,
        )


@torch.compile
def _quest_estimate_torch_impl(
    q,
    k_metadata,
    req_to_token,
    req_pool_indices,
    seq_lens,
    page_size: int,
    max_pages_process: int,
    kv_group_num: int,
    grouped: bool,
):
    B, Hq, D = q.shape
    Hkv = k_metadata.shape[1]

    # 1. Prepare indices
    # [1, P]
    page_offsets = (
        torch.arange(max_pages_process, device=q.device).unsqueeze(0) * page_size
    )
    # [B, 1]
    pool_indices = req_pool_indices.unsqueeze(1)

    # Gather global token IDs: [B, P]
    # req_to_token is [PoolSize, MaxLen]
    global_token_ids = req_to_token[pool_indices, page_offsets]
    global_page_ids = global_token_ids // page_size

    # 2. Gather K metadata
    # [B, P, Hkv, D, 2]
    k_data = k_metadata[global_page_ids]

    # 3. Compute Score
    # q: [B, Hkv, G, D]
    q_reshaped = q.view(B, Hkv, kv_group_num, D)

    # Broadcast shapes:
    # q: [B, 1, Hkv, G, D]
    # k: [B, P, Hkv, 1, D]
    q_b = q_reshaped.unsqueeze(1)
    k_min = k_data[..., 0].unsqueeze(3)
    k_max = k_data[..., 1].unsqueeze(3)

    # Compute max(q * k_min, q * k_max) for each dimension
    # This is the correct formula from the original Triton kernel
    qk_min = q_b * k_min
    qk_max = q_b * k_max
    score_per_dim = torch.maximum(qk_min, qk_max)

    # [B, P, Hkv, G, D] -> sum over D -> [B, P, Hkv, G]
    scores_g = torch.sum(score_per_dim, dim=-1)

    # 4. Handle grouping
    if grouped:
        # Sum over groups: [B, P, Hkv]
        scores = torch.sum(scores_g, dim=-1)
        # [B, Hkv, P]
        scores = scores.permute(0, 2, 1)
    else:
        # Flatten groups to heads: [B, P, Hq]
        scores = scores_g.flatten(2, 3)
        # [B, Hq, P]
        scores = scores.permute(0, 2, 1)

    # 5. Mask invalid pages
    # valid range: 0 .. num_pages - 2 (inclusive)
    num_pages = (seq_lens + page_size - 1) // page_size
    # [B, 1]
    valid_limit = (num_pages - 1).unsqueeze(1)
    # [1, P]
    page_indices = torch.arange(max_pages_process, device=q.device).unsqueeze(0)

    # [B, P] -> [B, 1, P]
    mask = (page_indices < valid_limit).unsqueeze(1)

    # Apply mask (fill 0.0)
    scores = torch.where(mask, scores, torch.zeros_like(scores))

    return scores


def quest_estimate_scores_triton(
    q: torch.Tensor,                    # [B, Hq, D]
    k_metadata: torch.Tensor,           # [num_pages, Hkv, D, 2]
    seq_lens: torch.Tensor,             # [B]
    estimated_scores: torch.Tensor,     # weak: [B, Hq, max_pages], strong: [B, Hkv, max_pages]
    page_size: int,
    req_to_token: torch.Tensor,         # [max_pool, max_context_len]
    req_pool_indices: torch.Tensor,     # [B]
    num_page_splits: int,
    grouped: bool = False,              # 是否在 KV 组上做 score 的 group-reduction
):
    """
    Python 封装：估算 page-level 注意力得分
    
    输出 estimated_scores 中，最后一个 page 的得分保持为 0（不参与 TopK）
    """
    batch_size, num_q_heads, head_dim = q.shape
    max_pages = estimated_scores.shape[-1]

    # 计算 grid 和 block 配置（head_dim 向上取 2 的幂，方便 Triton 向量化）
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)

    # Splits 配置：固定 NUM_SPLITS，kernel 内部循环 pages。
    # num_page_splits 是一个较小的常数（如 8/16/32），用于在 page 维做粗
    # 粒度切分，避免单个 program 处理过长的 page 区间。
    if num_page_splits is None or num_page_splits <= 0:
        num_page_splits = 1

    # GQA: map Q heads to KV heads via kv_group_num
    num_kv_heads = k_metadata.shape[1]
    assert num_q_heads % num_kv_heads == 0, "Invalid GQA config: Hq must be multiple of Hkv"
    kv_group_num = num_q_heads // num_kv_heads
    nvtx.range_push("quest_estimate_kernel_splits")

    if not grouped or kv_group_num == 1:
        # 弱 GQA / MHA：直接按 Q head 维度并行，每个 head 独立估计 page scores。
        grid = (batch_size, num_q_heads, num_page_splits)
        quest_estimate_split_kernel[grid](
            q,
            k_metadata,
            seq_lens,
            estimated_scores,
            req_to_token,
            req_pool_indices,
            page_size,
            head_dim,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_metadata.stride(0),
            k_metadata.stride(1),
            k_metadata.stride(2),
            estimated_scores.stride(0),
            estimated_scores.stride(1),
            req_to_token.stride(0),
            req_pool_indices.stride(0),
            BLOCK_DMODEL=BLOCK_DMODEL,
            NUM_SPLITS=num_page_splits,
            kv_group_num=kv_group_num,
            num_warps=4,
            num_stages=2,
        )
    else:
        # 强 GQA：estimated_scores 视为 [B, Hkv, max_pages]，按 KV head 维度并行，在
        # kernel 内对组内所有 Q heads 做 score 的 group-reduction，并将 group score
        # 存入对应的 KV 组槽位（不再扩展到 Q heads）。
        grid = (batch_size, num_kv_heads, num_page_splits)
        quest_estimate_split_kernel_grouped[grid](
            q,
            k_metadata,
            seq_lens,
            estimated_scores,
            req_to_token,
            req_pool_indices,
            page_size,
            head_dim,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_metadata.stride(0),
            k_metadata.stride(1),
            k_metadata.stride(2),
            estimated_scores.stride(0),
            estimated_scores.stride(1),
            req_to_token.stride(0),
            req_pool_indices.stride(0),
            BLOCK_DMODEL=BLOCK_DMODEL,
            NUM_SPLITS=num_page_splits,
            kv_group_num=kv_group_num,
            num_warps=4,
            num_stages=2,
        )

    nvtx.range_pop()


def quest_estimate_scores_torch(
    q: torch.Tensor,                    # [B, Hq, D]
    k_metadata: torch.Tensor,           # [num_pages, Hkv, D, 2]
    seq_lens: torch.Tensor,             # [B]
    estimated_scores: torch.Tensor,     # weak: [B, Hq, max_pages], strong: [B, Hkv, max_pages]
    page_size: int,
    req_to_token: torch.Tensor,         # [max_pool, max_context_len]
    req_pool_indices: torch.Tensor,     # [B]
    num_page_splits: int,
    grouped: bool = False,              # 是否在 KV 组上做 score 的 group-reduction
    debug: bool = False,                # CPU 验证
    max_pages_to_process: int | None = None,
):
    """
    Python 封装：估算 page-level 注意力得分
    
    输出 estimated_scores 中，最后一个 page 的得分保持为 0（不参与 TopK）
    """
    batch_size, num_q_heads, head_dim = q.shape
    
    # GQA: map Q heads to KV heads via kv_group_num
    num_kv_heads = k_metadata.shape[1]
    assert num_q_heads % num_kv_heads == 0, "Invalid GQA config: Hq must be multiple of Hkv"
    kv_group_num = num_q_heads // num_kv_heads

    if max_pages_to_process is None:
        # Calculate effective max pages to avoid wasted compute
        # Syncing max seq len is cheap compared to running computation on padded pages
        if batch_size > 0:
            max_seq_len = seq_lens.max().item()
            max_effective_pages = (max_seq_len + page_size - 1) // page_size
        else:
            max_effective_pages = 0
    else:
        max_effective_pages = max_pages_to_process
    
    # Clamp to output capacity
    max_pages_out = estimated_scores.shape[-1]
    process_pages = min(max_effective_pages, max_pages_out)
    
    if process_pages <= 0:
        return

    nvtx.range_push("quest_estimate_torch")

    scores = _quest_estimate_torch_impl(
        q,
        k_metadata,
        req_to_token,
        req_pool_indices,
        seq_lens,
        page_size,
        process_pages,
        kv_group_num,
        grouped
    )
    
    # Write back results
    estimated_scores[:, :, :process_pages].copy_(scores)

    nvtx.range_pop()


@triton.jit
def quest_select_topk_kernel_max(
    estimated_scores_ptr,
    seq_lens_ptr,
    selected_pages_ptr,
    stride_sb,
    stride_sh,
    stride_sel_b,
    stride_sel_h,
    page_size: tl.constexpr,
    quest_topk: tl.constexpr,       # total selected pages including last page
    max_pages: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,      # next power-of-2 of max_pages
    BLOCK_SLOTS: tl.constexpr,      # next power-of-2 of quest_topk
    num_heads: tl.constexpr,
):
    program_id = tl.program_id(0)
    batch_idx = program_id // num_heads
    head_idx = program_id % num_heads

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    num_pages = (seq_len + page_size - 1) // page_size
    valid_pages = tl.maximum(num_pages - 1, 0)

    scores_base = estimated_scores_ptr + batch_idx * stride_sb + head_idx * stride_sh
    offs = tl.arange(0, BLOCK_PAGES)
    scores = tl.load(scores_base + offs, mask=offs < max_pages, other=-float("inf"))
    scores = tl.where(offs < valid_pages, scores, -float("inf"))

    selected_base = selected_pages_ptr + batch_idx * stride_sel_b + head_idx * stride_sel_h
    idx_slots = tl.arange(0, BLOCK_SLOTS)
    # initialize to -1 for all quest_topk slots
    tl.store(
        selected_base + idx_slots, tl.full([BLOCK_SLOTS], -1, dtype=tl.int32), mask=idx_slots < quest_topk
    )

    # pick top-(quest_topk-1) pages excluding last page; the last slot is reserved for last page
    if valid_pages <= (quest_topk - 1):
        rng = tl.arange(0, BLOCK_SLOTS)
        sequential = tl.where(rng < valid_pages, rng, -1)
        tl.store(selected_base + rng, sequential, mask=rng < (quest_topk - 1))
    else:
        for i in range(quest_topk - 1):
            max_idx = tl.argmax(scores, axis=0)
            tl.store(selected_base + i, max_idx.to(tl.int32))
            scores = tl.where(offs == max_idx, -float("inf"), scores)

    # append last page to the last slot
    last_page = tl.where(num_pages > 0, num_pages - 1, 0)
    tl.store(selected_base + (quest_topk - 1), last_page.to(tl.int32))


@triton.jit
def quest_select_topk_kernel_sort(
    estimated_scores_ptr,
    seq_lens_ptr,
    selected_pages_ptr,
    stride_sb,
    stride_sh,
    stride_sel_b,
    stride_sel_h,
    page_size: tl.constexpr,
    quest_topk: tl.constexpr,       # total selected pages including last page
    max_pages: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,      # next power-of-2 of max_pages
    BLOCK_SLOTS: tl.constexpr,      # next power-of-2 of quest_topk
    num_heads: tl.constexpr,
    K_P2: tl.constexpr,
):
    program_id = tl.program_id(0)
    batch_idx = program_id // num_heads
    head_idx = program_id % num_heads

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    num_pages = (seq_len + page_size - 1) // page_size
    valid_pages = tl.maximum(num_pages - 1, 0)

    scores_base = estimated_scores_ptr + batch_idx * stride_sb + head_idx * stride_sh
    offs = tl.arange(0, BLOCK_PAGES)
    # Load scores, mask invalid/padding with -inf
    scores = tl.load(scores_base + offs, mask=offs < max_pages, other=-float("inf"))
    scores = tl.where(offs < valid_pages, scores, -float("inf"))

    selected_base = selected_pages_ptr + batch_idx * stride_sel_b + head_idx * stride_sel_h
    idx_slots = tl.arange(0, BLOCK_SLOTS)
    # initialize to -1
    tl.store(
        selected_base + idx_slots, tl.full([BLOCK_SLOTS], -1, dtype=tl.int32), mask=idx_slots < quest_topk
    )

    K_SELECT: tl.constexpr = quest_topk - 1

    # Case 1: Not enough valid pages to fill K-1 slots
    # We just select all valid pages (0 to valid_pages-1)
    if valid_pages <= K_SELECT:
        rng = tl.arange(0, BLOCK_SLOTS)
        sequential = tl.where(rng < valid_pages, rng, -1)
        # store only up to K_SELECT slots (though valid_pages is smaller, mask handles it)
        tl.store(selected_base + rng, sequential, mask=rng < K_SELECT)
    else:
        # Case 2: Use tl.topk to select top K-1 pages
        if K_SELECT > 0:
             # --- Packing Logic ---
             # Float32 -> Int32 bitcast
             i_scores = scores.to(tl.int32, bitcast=True)
             
             # Map to Monotonic Int32 (Descending Sort friendly)
             # If x < 0: mask = 0x7FFFFFFF. key = x ^ mask. (-inf->MinInt, -0->-1)
             # If x >= 0: mask = 0. key = x. (+0->0, +inf->MaxInt)
             # Total order: -inf < -0 < +0 < +inf.
             mask = tl.where(i_scores < 0, 0x7FFFFFFF, 0).to(tl.int32)
             mapped_scores = i_scores ^ mask
             
             # Pack into Int64: [Score(32) | Index(32)]
             # Ensure bitwise operations use 64-bit container
             # mapped_scores is int32. 
             # We need to cast to int64. 
             # Note: mapped_scores is SIGNED int32. 
             # -inf maps to 0x80000000 (-2^31).
             # Casting to int64 preserves value (sign extension).
             # -2^31 -> -2^31 in int64 (FFFF...8000...).
             # Shift << 32 -> FFFF..8000..0000..
             # This preserves order for Signed Int64 comparison.
             packed = (mapped_scores.to(tl.int64) << 32) | offs.to(tl.int64)
             
             # tl.topk
             # Ensure K for topk is power of 2 to match register shape requirements
             # (Triton primitives often return power-of-2 sized tensors)
             # We ask for slightly more if K_SELECT is not power of 2, then mask store.
             topk_packed = tl.topk(packed, K_P2)
             
             # Unpack indices (Low 32 bits)
             topk_indices = topk_packed.to(tl.int32)
             
             # Store
             offs_store = tl.arange(0, K_P2)
             tl.store(selected_base + offs_store, topk_indices, mask=offs_store < K_SELECT)
             
    # append last page to the last slot (index quest_topk - 1)
    last_page = tl.where(num_pages > 0, num_pages - 1, 0)
    tl.store(selected_base + (quest_topk - 1), last_page.to(tl.int32))


@triton.jit
def quest_expand_pages_to_csr_kernel(
    selected_pages_ptr,         # [bsz, num_heads, quest_topk]
    seq_lens_ptr,               # [bsz]
    req_to_token_ptr,           # [max_pool, max_seq_len]
    req_pool_indices_ptr,       # [bsz]
    kv_indices_ptr,             # output [bsz * num_heads, tokens_cap]
    tokens_per_head_ptr,        # output [bsz * num_heads]
    stride_sel_b,               # 340
    stride_sel_h,               # 17
    stride_rt_b: tl.constexpr,  # 32772
    stride_rp: tl.constexpr,    # 1
    stride_rt_pos: tl.constexpr,# 1
    tokens_cap: tl.constexpr,   # 272
    BLOCK_TOKENS: tl.constexpr, # 512
    stride_idx_h: tl.constexpr, # 1
    page_size: tl.constexpr,    # 16
    quest_topk: tl.constexpr,   # total selected pages including last page
    num_heads: tl.constexpr,    # 20
):
    program_id = tl.program_id(0)  # 0
    batch_idx = program_id // num_heads  # 0
    head_idx = program_id % num_heads  # 0

    # Ensure scalar math stays in int32
    seq_len = tl.load(seq_lens_ptr + batch_idx).to(tl.int32)  # 501
    ps = tl.full((), page_size, dtype=tl.int32)  # 16
    one = tl.full((), 1, dtype=tl.int32)  # 1
    num_pages = (seq_len + ps - one) // ps  # (501+16-1)//16=32
    pool_row = tl.load(req_pool_indices_ptr + batch_idx)

    selected_base = selected_pages_ptr + batch_idx * stride_sel_b + head_idx * stride_sel_h  # ptr + 0 * 340 + 0 * 17 = ptr + 0
    indices_base = kv_indices_ptr + (batch_idx * num_heads + head_idx) * stride_idx_h  # ptr + (0*20+0)*1 = ptr + 0

    offsets_cap = tl.arange(0, BLOCK_TOKENS)  # [0, 1, 2, ..., 511]
    tl.store(
        indices_base + offsets_cap,
        tl.full([BLOCK_TOKENS], 0, dtype=tl.int32),
        mask=offsets_cap < tokens_cap,
    )  # kv_indices initialize to 0

    write_offset = tl.zeros((), dtype=tl.int32)  # 0
    token_offsets = tl.arange(0, page_size)  # [0, 1, 2, ..., 15]

    # iterate over quest_topk slots (topk-1 + last)
    for i in range(quest_topk):
        page = tl.load(selected_base + i).to(tl.int32)  # 12 (, 13, 11, 19)
        is_valid_page = (page >= 0) & (page < num_pages)
        start = tl.where(is_valid_page, page * ps, tl.zeros((), dtype=tl.int32))  # 12 * 16 = 192
        end = tl.minimum(start + ps, seq_len)  # min(192+16, 501)=208
        length = tl.where(is_valid_page, end - start, tl.zeros((), dtype=tl.int32))  # 16

        # compute per-lane offsets and robust mask to ensure in-bounds
        lane_offsets = write_offset + token_offsets  # [0, 1, 2, ..., 15]
        mask_token = token_offsets < length  # [T,T,T,...,T] length=16
        mask_ptr = mask_token & (lane_offsets < tokens_cap)
        # map logical position to physical token id via req_to_token
        logical_pos = start + token_offsets  # [192, 193, 194, ..., 207]
        phys_ids = tl.load(
            req_to_token_ptr + pool_row * stride_rt_b + logical_pos * stride_rt_pos,
            mask=mask_token,
            other=0,
        )
        tl.store(indices_base + lane_offsets, phys_ids, mask=mask_ptr)

        write_offset += length

    tl.store(tokens_per_head_ptr + program_id, write_offset)


@triton.jit
def quest_pack_kv_indices_kernel(
    kv_indices_src_ptr,        # [bs*H, tokens_cap]
    kv_indices_dst_ptr,        # [sum_b L_b * H]
    kv_indptr_ptr,             # [bs+1]
    tokens_per_batch_ptr,      # [bs]
    stride_src_h: tl.constexpr,
    tokens_cap: tl.constexpr,
    num_heads: tl.constexpr,
    BLOCK_COPY: tl.constexpr,
    ITERS: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)

    Lb = tl.load(tokens_per_batch_ptr + bid).to(tl.int32)
    if Lb == 0:
        return

    # src row = (bid * H + hid)
    src_row = bid * num_heads + hid
    src_base = kv_indices_src_ptr + src_row * stride_src_h

    # dst base offset = (kv_indptr[bid] * H) + hid * Lb
    base_tokens_before = tl.load(kv_indptr_ptr + bid).to(tl.int32)
    dst_offset_base = base_tokens_before * num_heads + hid * Lb
    dst_base = kv_indices_dst_ptr + dst_offset_base

    offs = tl.arange(0, BLOCK_COPY)
    # copy in fixed number of chunks; guard with masks
    for it in range(ITERS):
        start = tl.full((), it * BLOCK_COPY, dtype=tl.int32)
        # source bounds: start + offs < tokens_cap
        src_mask = (start + offs) < tokens_cap
        # dest bounds: start + offs < Lb
        dst_mask = (start + offs) < Lb
        mask = src_mask & dst_mask
        vals = tl.load(src_base + start + offs, mask=mask, other=0)
        tl.store(dst_base + start + offs, vals, mask=mask)


def quest_select_topk_pages(
    estimated_scores: torch.Tensor,     # [batch, num_heads, max_pages]
    seq_lens: torch.Tensor,             # [batch]
    req_to_token: torch.Tensor,         # [max_pool, max_context_len]
    req_pool_indices: torch.Tensor,     # [batch]
    quest_topk: int,
    page_size: int,
    kernel_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """选择 topk-1 个非末页并追加 last page，共 quest_topk 个页；展开为 CSR 索引。"""
    batch_size, num_heads, max_pages = estimated_scores.shape
    device = estimated_scores.device

    assert quest_topk > 0, "quest_topk must be greater than 0"

    # Allocate slots for (topk-1) + last = quest_topk pages
    selected_pages = torch.empty(
        (batch_size, num_heads, quest_topk), dtype=torch.int32, device=device
    )

    # Round sizes up to power-of-two for tl.arange
    BLOCK_PAGES = triton.next_power_of_2(max_pages)
    BLOCK_SLOTS = triton.next_power_of_2(quest_topk)

    grid = (batch_size * num_heads,)
    if kernel_type == "max":
        quest_select_topk_kernel_max[grid](
            estimated_scores,
            seq_lens,
            selected_pages,
            estimated_scores.stride(0),
            estimated_scores.stride(1),
            selected_pages.stride(0),
            selected_pages.stride(1),
            page_size=page_size,
            quest_topk=quest_topk,
            max_pages=max_pages,
            BLOCK_PAGES=BLOCK_PAGES,
            BLOCK_SLOTS=BLOCK_SLOTS,
            num_heads=num_heads,
        )
    elif kernel_type == "sort":
        quest_select_topk_kernel_sort[grid](
            estimated_scores,
            seq_lens,
            selected_pages,
            estimated_scores.stride(0),
            estimated_scores.stride(1),
            selected_pages.stride(0),
            selected_pages.stride(1),
            page_size=page_size,
            quest_topk=quest_topk,
            max_pages=max_pages,
            BLOCK_PAGES=BLOCK_PAGES,
            BLOCK_SLOTS=BLOCK_SLOTS,
            num_heads=num_heads,
            K_P2=triton.next_power_of_2(max(1, quest_topk - 1)),
        )

    tokens_cap = quest_topk * page_size
    BLOCK_TOKENS = triton.next_power_of_2(tokens_cap)
    kv_indices_buf = torch.empty(
        (batch_size * num_heads, tokens_cap), dtype=torch.int32, device=device
    )
    tokens_per_head = torch.zeros(
        (batch_size * num_heads,), dtype=torch.int32, device=device
    )

    quest_expand_pages_to_csr_kernel[grid](
        selected_pages,
        seq_lens,
        req_to_token,
        req_pool_indices,
        kv_indices_buf,
        tokens_per_head,
        selected_pages.stride(0),
        selected_pages.stride(1),
        stride_rt_b=req_to_token.stride(0),
        stride_rp=req_pool_indices.stride(0),
        stride_rt_pos=req_to_token.stride(1),
        tokens_cap=tokens_cap,
        BLOCK_TOKENS=BLOCK_TOKENS,
        stride_idx_h=kv_indices_buf.stride(0),
        page_size=page_size,
        quest_topk=quest_topk,
        num_heads=num_heads,
    )

    # Reshape to per-batch, per-head counts: [bs, Hq]
    tokens_per_head = tokens_per_head.view(batch_size, num_heads)

    # All heads are guaranteed to select the same number of tokens for a given batch
    # (only the specific token ids may differ). Therefore we can simply take the
    # first head's length per batch as tokens_per_batch to avoid a redundant reduction.
    # Shapes:
    # - tokens_per_head: [bs, Hq]
    # - tokens_per_batch: [bs]
    tokens_per_batch = tokens_per_head[:, 0].contiguous()

    # TODO(xiaoyuan): remove env var
    # Optional strict check (debug): verify all heads agree on the count per batch.
    if os.environ.get("QUEST_ASSERT_EQUAL_TOKENS", "0") == "1" and batch_size > 0:
        # Check each row equals its first element across heads
        ref = tokens_per_batch[:, None].expand_as(tokens_per_head)
        if not torch.equal(tokens_per_head, ref):
            raise RuntimeError(
                "[Quest] tokens_per_head are not equal across heads within a batch."
            )

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    torch.cumsum(tokens_per_batch, dim=0, out=kv_indptr[1:])

    kv_indices_buf = kv_indices_buf.view(batch_size, num_heads, tokens_cap)
    total_tokens_per_head = kv_indptr[-1].item()
    total_tokens_all_heads = total_tokens_per_head * num_heads
    kv_indices = torch.empty((total_tokens_all_heads,), dtype=torch.int32, device=device)

    # pack into tightly-packed layout expected by decode kernel
    BLOCK_COPY = triton.next_power_of_2(tokens_cap)
    ITERS = (tokens_cap + BLOCK_COPY - 1) // BLOCK_COPY
    src_2d = kv_indices_buf.reshape(-1, tokens_cap)
    quest_pack_kv_indices_kernel[(batch_size, num_heads)](
        src_2d,
        kv_indices,
        kv_indptr,
        tokens_per_batch,
        src_2d.stride(0),
        tokens_cap,
        num_heads,
        BLOCK_COPY,
        ITERS,
    )

    # no debug printing

    return kv_indptr, kv_indices


@triton.jit
def quest_update_metadata_kernel(
    K_new,                  # [batch, num_heads, head_dim] - 当前 step 的新 K
    K_metadata,             # [num_pages, num_heads, head_dim, 2]
    Out_cache_loc,          # [batch] - 写入位置（token index）
    page_size: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    stride_kb_new,
    stride_kh_new,
    stride_kd_new,
    stride_mp,
    stride_mh,
    stride_md,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    增量更新元数据：写入新 K 时同步更新 page 的 min/max
    
    Grid: (batch, num_heads)
    
    逻辑：
    加载当前 page 的 old min/max（首次为 inf），与新 K 比较后更新
    """
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    
    # 获取写入位置
    loc = tl.load(Out_cache_loc + bid)
    
    # 计算 page index
    page_idx = loc // page_size
    
    # 加载新 K（显式转换为 float32 以保持类型一致性）
    offs_d = tl.arange(0, BLOCK_DMODEL)
    k_new = tl.load(
        K_new + bid * stride_kb_new + hid * stride_kh_new + offs_d * stride_kd_new,
        mask=offs_d < head_dim,
        other=0.0,
    ).to(tl.float32)  # 显式转换为 float32
    
    # 加载旧的 min/max（首次会加载 inf 值）
    # K_metadata[page, head, d, 0=min/1=max]
    # stride_md 是 head_dim 维度的 stride (=2)，已包含最后一维 [min,max] 的信息
    old_min = tl.load(
        K_metadata + page_idx * stride_mp + hid * stride_mh + offs_d * stride_md + 0,
        mask=offs_d < head_dim,
        other=65504.0,
    ).to(tl.float32)  # 显式转换为 float32
    old_max = tl.load(
        K_metadata + page_idx * stride_mp + hid * stride_mh + offs_d * stride_md + 1,
        mask=offs_d < head_dim,
        other=-65504.0,
    ).to(tl.float32)  # 显式转换为 float32
    
    # 增量更新
    new_min = tl.minimum(old_min, k_new)
    new_max = tl.maximum(old_max, k_new)
    
    # 存储更新后的 min/max
    tl.store(
        K_metadata + page_idx * stride_mp + hid * stride_mh + offs_d * stride_md + 0,
        new_min,
        mask=offs_d < head_dim,
    )
    tl.store(
        K_metadata + page_idx * stride_mp + hid * stride_mh + offs_d * stride_md + 1,
        new_max,
        mask=offs_d < head_dim,
    )


@triton.jit
def quest_min_tokens_per_batch_kernel(
    tokens_per_head_ptr,   # [bs * num_heads]
    tokens_per_batch_ptr,  # [bs]
    num_heads: tl.constexpr,
):
    bid = tl.program_id(0)
    # sequentially reduce across heads (num_heads is small and constexpr)
    min_val = tl.full((), 0x7FFFFFFF, dtype=tl.int32)
    for h in range(num_heads):
        v = tl.load(tokens_per_head_ptr + bid * num_heads + h)
        min_val = tl.minimum(min_val, v)
    tl.store(tokens_per_batch_ptr + bid, min_val)


def quest_select_topk_pages_into(
    estimated_scores: torch.Tensor,     # [bs, num_heads, max_pages_cap]
    seq_lens: torch.Tensor,             # [bs]
    req_to_token: torch.Tensor,         # [max_pool, max_context_len]
    req_pool_indices: torch.Tensor,     # [bs]
    quest_topk: int,
    page_size: int,
    selected_pages: torch.Tensor,       # prealloc [bs, num_heads, quest_topk]
    kv_indices_buf: torch.Tensor,       # prealloc [(bs*num_heads), tokens_cap]
    tokens_per_head: torch.Tensor,      # prealloc [(bs*num_heads)]
    tokens_per_batch: torch.Tensor,     # prealloc [bs]
    kv_indptr: torch.Tensor,            # prealloc [bs+1]
    kv_indices: torch.Tensor,           # prealloc big 1D buffer
    num_heads: int,
    kernel_type: str,
):
    device = estimated_scores.device
    bs = estimated_scores.shape[0]
    max_pages = estimated_scores.shape[-1]
    tokens_cap = quest_topk * page_size

    # 1) select top-(k-1) + append last, into selected_pages
    BLOCK_PAGES = triton.next_power_of_2(max_pages)
    BLOCK_SLOTS = triton.next_power_of_2(quest_topk)
    grid_sel = (bs * num_heads,)
    if kernel_type == "max":
        quest_select_topk_kernel_max[grid_sel](
            estimated_scores,
            seq_lens,
            selected_pages,
            estimated_scores.stride(0),
            estimated_scores.stride(1),
            selected_pages.stride(0),
            selected_pages.stride(1),
            page_size=page_size,
            quest_topk=quest_topk,
            max_pages=max_pages,
            BLOCK_PAGES=BLOCK_PAGES,
            BLOCK_SLOTS=BLOCK_SLOTS,
            num_heads=num_heads,
        )
    elif kernel_type == "sort":
        quest_select_topk_kernel_sort[grid_sel](
            estimated_scores,
            seq_lens,
            selected_pages,
            estimated_scores.stride(0),
            estimated_scores.stride(1),
            selected_pages.stride(0),
            selected_pages.stride(1),
            page_size=page_size,
            quest_topk=quest_topk,
            max_pages=max_pages,
            BLOCK_PAGES=BLOCK_PAGES,
            BLOCK_SLOTS=BLOCK_SLOTS,
            num_heads=num_heads,
            K_P2=triton.next_power_of_2(max(1, quest_topk - 1)),
        )

    # 2) expand pages to tokens (per head) + count tokens per head
    BLOCK_TOKENS = triton.next_power_of_2(tokens_cap)
    quest_expand_pages_to_csr_kernel[grid_sel](
        selected_pages,
        seq_lens,
        req_to_token,
        req_pool_indices,
        kv_indices_buf,
        tokens_per_head,
        selected_pages.stride(0),
        selected_pages.stride(1),
        stride_rt_b=req_to_token.stride(0),
        stride_rp=req_pool_indices.stride(0),
        stride_rt_pos=req_to_token.stride(1),
        tokens_cap=tokens_cap,
        BLOCK_TOKENS=BLOCK_TOKENS,
        stride_idx_h=kv_indices_buf.stride(0),
        page_size=page_size,
        quest_topk=quest_topk,
        num_heads=num_heads,
    )

    # 3) per-batch token count: take the first head's length (all heads are equal)
    if bs > 0:
        # tokens_per_head: 1D [(bs * Hq)] -> 2D [bs, Hq]
        tph_2d = tokens_per_head.view(bs, num_heads)

        # Optional strict check (debug): ensure equality across heads
        if os.environ.get("QUEST_ASSERT_EQUAL_TOKENS", "0") == "1":
            ref = tph_2d[:, :1].expand_as(tph_2d)
            if not torch.equal(tph_2d, ref):
                raise RuntimeError(
                    "[Quest] tokens_per_head are not equal across heads within a batch."
                )

        # Fill tokens_per_batch[:bs] from the first head
        tokens_per_batch.copy_(tph_2d[:, 0].contiguous())
    # kv_indptr[0] expected to be set by caller; compute cumsum into kv_indptr[1:]
    torch.cumsum(tokens_per_batch, dim=0, out=kv_indptr[1:])

    # 4) pack into tightly-packed 1D indices layout expected by decode kernel
    src_2d = kv_indices_buf.view(bs, num_heads, tokens_cap).reshape(-1, tokens_cap)
    BLOCK_COPY = triton.next_power_of_2(tokens_cap)
    ITERS = (tokens_cap + BLOCK_COPY - 1) // BLOCK_COPY
    quest_pack_kv_indices_kernel[(bs, num_heads)](
        src_2d,
        kv_indices,
        kv_indptr,
        tokens_per_batch,
        src_2d.stride(0),
        tokens_cap,
        num_heads,
        BLOCK_COPY,
        ITERS,
    )


def quest_update_kv_and_metadata(
    k_new: torch.Tensor,                # [batch, num_heads, head_dim]
    v_new: torch.Tensor,                # [batch, num_heads, head_dim]
    k_buffer: torch.Tensor,             # [size, num_heads, head_dim]
    v_buffer: torch.Tensor,             # [size, num_heads, head_dim]
    k_metadata: torch.Tensor,           # [num_pages, num_heads, head_dim, 2]
    out_cache_loc: torch.Tensor,        # [batch]
    page_size: int,
):
    """
    Python 封装：同步更新 KV cache 和元数据
    
    步骤：
    1. 写入 KV cache
    2. 增量更新元数据（首次会从 inf 更新）
    """
    batch_size, num_heads, head_dim = k_new.shape
    
    # Step 1: 写入 KV cache
    k_buffer[out_cache_loc] = k_new
    v_buffer[out_cache_loc] = v_new
    
    # Step 2: 更新元数据
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)
    grid = (batch_size, num_heads)
    
    quest_update_metadata_kernel[grid](
        k_new,
        k_metadata,
        out_cache_loc,
        page_size,
        num_heads,
        head_dim,
        k_new.stride(0),
        k_new.stride(1),
        k_new.stride(2),
        k_metadata.stride(0),
        k_metadata.stride(1),
        k_metadata.stride(2),
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=2,
    )


@triton.jit
def quest_compute_extend_metadata_kernel(
    K_new,              # [extend_num_tokens, num_heads, head_dim]
    K_metadata,         # [num_pages, num_heads, head_dim, 2]
    Out_cache_loc,      # [extend_num_tokens]
    Extend_start_loc,   # [batch+1]
    Seq_lens,           # [batch]
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    stride_tok,
    stride_head,
    stride_d,
    stride_mp,  # 5120
    stride_mh,  # 256
    stride_md,  # 2
    BLOCK_DMODEL: tl.constexpr,  # 128
):
    """
    计算 extend 阶段的 page 元数据（min/max）
    
    Grid: (max_pages_per_batch, num_heads, batch_size)
    
    直接从输入的 K_new 计算，无需回读 KV cache
    利用"序列内部按 page 对齐"的假设，O(1) 索引计算
    """
    page_idx = tl.program_id(0)  # page index within the sequence
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)
    
    # === 边界处理 1：检查该 batch 是否有这个 page ===
    seq_len = tl.load(Seq_lens + batch_idx)  # 500
    num_pages = (seq_len + page_size - 1) // page_size  # 32
    if page_idx >= num_pages:
        return  # 该 batch 没有这么多 pages
    
    # === 边界处理 2：计算该 page 的实际 token 数量 ===
    page_start_in_seq = page_idx * page_size  # 0, 16, 32, ..., 496
    page_end_in_seq = tl.minimum((page_idx + 1) * page_size, seq_len)  # min(16 32 ... 512, 500)
    actual_tokens = page_end_in_seq - page_start_in_seq
    
    # 该 batch 在 K_new 中的起始位置
    batch_start = tl.load(Extend_start_loc + batch_idx)  # 0, 500
    
    # 初始化 min/max TODO: confirm if 65504 is enough
    offs_d = tl.arange(0, BLOCK_DMODEL)
    k_min = tl.full([BLOCK_DMODEL], 65504.0, dtype=tl.float32)
    k_max = tl.full([BLOCK_DMODEL], -65504.0, dtype=tl.float32)
    
    # === 关键优化：固定循环 page_size 次 ===
    for t in range(page_size):  # 编译时常量，Triton 可以优化
        if t < actual_tokens:  # 动态 mask 处理最后一个 page
            # 在 K_new 中的全局索引
            token_idx_global = batch_start + page_start_in_seq + t
            
            # 加载 K 向量
            k = tl.load(
                K_new + token_idx_global * stride_tok + head_idx * stride_head + offs_d * stride_d,
                mask=offs_d < head_dim,
                other=0.0,
            ).to(tl.float32)
            
            # 更新 min/max
            k_min = tl.minimum(k_min, k)
            k_max = tl.maximum(k_max, k)
    
    # === 确定全局 page index（只读一次 Out_cache_loc）===
    first_token_global = batch_start + page_start_in_seq  # 500(, 516, 532, ..., 996)
    loc = tl.load(Out_cache_loc + first_token_global)  # out_cache_loc[500]=528 physical loc of first token in this page
    global_page_idx = loc // page_size  # 33
    
    # 加载该 page 的旧 metadata（可能有之前的数据）
    # K_metadata[page, head, d, 0=min/1=max]
    # stride_md 是 head_dim 维度的 stride (=2)，已包含最后一维 [min,max] 的信息
    old_min = tl.load(
        K_metadata + global_page_idx * stride_mp + head_idx * stride_mh + offs_d * stride_md + 0,
        mask=offs_d < head_dim,
        other=65504.0,
    ).to(tl.float32)
    old_max = tl.load(
        K_metadata + global_page_idx * stride_mp + head_idx * stride_mh + offs_d * stride_md + 1,
        mask=offs_d < head_dim,
        other=-65504.0,
    ).to(tl.float32)
    
    # 合并新旧值
    new_min = tl.minimum(old_min, k_min)
    new_max = tl.maximum(old_max, k_max)
    
    # 存储更新后的 metadata
    tl.store(
        K_metadata + global_page_idx * stride_mp + head_idx * stride_mh + offs_d * stride_md + 0,
        new_min,
        mask=offs_d < head_dim,
    )
    tl.store(
        K_metadata + global_page_idx * stride_mp + head_idx * stride_mh + offs_d * stride_md + 1,
        new_max,
        mask=offs_d < head_dim,
    )


def quest_compute_extend_metadata(
    k_new: torch.Tensor,               # [extend_num_tokens, num_heads, head_dim]
    k_metadata: torch.Tensor,          # [num_pages, num_heads, head_dim, 2]
    out_cache_loc: torch.Tensor,       # [extend_num_tokens]
    extend_start_loc: torch.Tensor,    # [batch+1]
    seq_lens: torch.Tensor,            # [batch]
    page_size: int,
    max_pages_cap: int | None = None,
):
    """
    计算 extend 阶段的 page 元数据（min/max）
    
    使用三维并行: (max_pages, num_heads, batch_size)
    利用序列内部 page 对齐假设，O(1) 索引计算，最小化 load 开销
    直接从输入的 K 计算，无需回读 KV cache，数据局部性好
    """
    batch_size = seq_lens.shape[0]
    num_heads = k_new.shape[1]
    head_dim = k_new.shape[2]
    
    if batch_size == 0:
        return
    
    # 计算最大的 page 数量（避免 .item() 引起的 host 同步，允许外部传入上限）
    if max_pages_cap is None:
        nvtx.range_push("max pages (host-sync)")
        max_pages = ((seq_lens.max().item() + page_size - 1) // page_size)
        nvtx.range_pop()
    else:
        max_pages = int(max_pages_cap)
    
    nvtx.range_push("block dmodel")
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)
    grid = (max_pages, num_heads, batch_size)
    nvtx.range_pop()
    
    nvtx.range_push("quest_compute_extend_metadata_kernel")
    quest_compute_extend_metadata_kernel[grid](
        k_new,
        k_metadata,
        out_cache_loc,
        extend_start_loc,
        seq_lens,
        page_size,
        head_dim,
        k_new.stride(0),
        k_new.stride(1),
        k_new.stride(2),
        k_metadata.stride(0),
        k_metadata.stride(1),
        k_metadata.stride(2),
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=2,
    )
    nvtx.range_pop()
