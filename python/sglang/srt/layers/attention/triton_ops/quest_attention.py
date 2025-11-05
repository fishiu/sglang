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
    Debug_Info,         # [batch, num_heads, max_kv_splits, 5] - debug info
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
    
    # 当前 head 在 kv_indices 中的起始偏移
    cur_batch_kv_start_idx = cur_head * cur_batch_seq_len
    
    kv_splits = tl.load(num_kv_splits + cur_batch)
    
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    
    # 计算当前 split 的范围
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )  # div(div(257, 8), 16)*16=48 很浪费就差一点点就32了
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
    
    # # Calculate loop iterations for debugging
    # loop_iters = tl.cdiv(tl.maximum(split_kv_end - split_kv_start, 0), BLOCK_N)
    
    # # Store debug info: [cur_batch_seq_len, kv_len_per_split, split_kv_start, split_kv_end, loop_iters]
    # debug_offset = cur_batch * (head_num * MAX_KV_SPLITS * 5) + cur_head * (MAX_KV_SPLITS * 5) + split_kv_id * 5
    # tl.store(Debug_Info + debug_offset + 0, cur_batch_seq_len)
    # tl.store(Debug_Info + debug_offset + 1, kv_len_per_split)
    # tl.store(Debug_Info + debug_offset + 2, split_kv_start)
    # tl.store(Debug_Info + debug_offset + 3, split_kv_end)
    # tl.store(Debug_Info + debug_offset + 4, loop_iters)
    
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
    q,              # [batch, num_heads, head_dim]
    k_buffer,       # [size, num_heads, head_dim]
    v_buffer,       # [size, num_heads, v_head_dim]
    o,              # [batch, num_heads, v_head_dim]
    kv_indptr,      # [batch + 1] - 所有 head 共享相同的 token 数量
    kv_indices,     # [total_selected_tokens]
    attn_logits,    # [batch, num_heads, max_kv_splits, v_head_dim]
    attn_lse,       # [batch, num_heads, max_kv_splits]
    num_kv_splits,  # [batch]
    max_kv_splits,  # int
    sm_scale,       # float
    logit_cap=0.0,
    layer_id=None,
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
    grid = (batch, head_num, MAX_KV_SPLITS)
    kv_group_num = q.shape[1] // k_buffer.shape[1]
    
    num_warps = 4 if kv_group_num == 1 else 2
    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    # Create debug info tensor: [batch, num_heads, max_kv_splits, 5]
    debug_info = torch.zeros((batch, head_num, MAX_KV_SPLITS, 5), dtype=torch.int32, device=q.device)
    
    if layer_id == 0:
        print(f"PUSH quest_decode_attention_fwd_stage1")
        nvtx.range_push(f"quest_attnl0")
    
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
        debug_info,
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
        MIN_BLOCK_KV=16,
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
        
    # # Print debug info in a readable format
    # print("\n" + "="*80)
    # print(f"DEBUG INFO - Grid: {grid}")
    # print("="*80)
    # debug_cpu = debug_info.cpu()
    # for b in range(batch):
    #     for h in range(head_num):
    #         print(f"\n[Batch {b}, Head {h}]")
    #         print(f"  {'Split':<8} {'SeqLen':<10} {'LenPerSplit':<15} {'Start':<10} {'End':<10} {'LoopIters':<10}")
    #         print(f"  {'-'*8} {'-'*10} {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    #         for s in range(MAX_KV_SPLITS):
    #             info = debug_cpu[b, h, s]
    #             seq_len, len_per_split, start, end, loop_iters = info[0].item(), info[1].item(), info[2].item(), info[3].item(), info[4].item()
    #             if seq_len > 0:  # Only print if valid
    #                 print(f"  {s:<8} {seq_len:<10} {len_per_split:<15} {start:<10} {end:<10} {loop_iters:<10}")
    # print("="*80 + "\n")
    
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


def quest_estimate_scores(
    q: torch.Tensor,                    # [batch, num_heads, head_dim]
    k_metadata: torch.Tensor,           # [num_pages, num_heads, head_dim, 2]
    seq_lens: torch.Tensor,             # [batch]
    estimated_scores: torch.Tensor,     # 输出: [batch, num_heads, max_pages]
    page_size: int,
    req_to_token: torch.Tensor,         # [batch, max_context_len]
    debug: bool = False,                # 是否启用 CPU 验证
):
    """
    Python 封装：估算 page-level 注意力得分
    
    输出 estimated_scores 中，最后一个 page 的得分保持为 0（不参与 TopK）
    """
    batch_size, num_heads, head_dim = q.shape
    max_pages = estimated_scores.shape[-1]
    
    # 计算 grid 和 block 配置
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)
    grid = (batch_size, num_heads, max_pages - 1)  # 排除 last page
    
    # 启动 kernel
    nvtx.range_push("quest_estimate_kernel")
    quest_estimate_kernel[grid](
        q,
        k_metadata,
        seq_lens,
        estimated_scores,
        req_to_token,
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
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=2,
    )
    nvtx.range_pop()
    
    # Debug: CPU 验证逻辑
    # 可以通过环境变量 QUEST_DEBUG_ESTIMATE=1 启用
    import os
    if debug or os.environ.get("QUEST_DEBUG_ESTIMATE", "0") == "1":
        print("\n" + "="*80)
        print("Quest Estimate Scores - CPU 验证")
        print("="*80)
        
        # 移到 CPU 进行计算
        q_cpu = q.cpu()
        k_metadata_cpu = k_metadata.cpu()
        seq_lens_cpu = seq_lens.cpu()
        req_to_token_cpu = req_to_token.cpu()
        estimated_scores_cpu = torch.zeros_like(estimated_scores).cpu()
        
        # 遍历所有 (batch, head, page)
        for bid in range(batch_size):
            seq_len = seq_lens_cpu[bid].item()
            num_pages = (seq_len + page_size - 1) // page_size
            
            for hid in range(num_heads):
                for pid in range(max_pages - 1):  # 排除 last page
                    # 过滤：只处理 [0, num_pages-2]
                    if pid >= num_pages - 1:
                        continue
                    
                    # 计算全局 page 索引
                    token_offset_in_seq = pid * page_size
                    global_token_idx = req_to_token_cpu[bid, token_offset_in_seq].item()
                    global_page_idx = global_token_idx // page_size
                    
                    # 加载 Q 向量
                    q_vec = q_cpu[bid, hid, :]  # [head_dim]
                    
                    # 加载元数据
                    k_min = k_metadata_cpu[global_page_idx, hid, :, 0]  # [head_dim]
                    k_max = k_metadata_cpu[global_page_idx, hid, :, 1]  # [head_dim]
                    
                    # 计算得分: sum(max(q*k_min, q*k_max))
                    qk_min = q_vec * k_min
                    qk_max = q_vec * k_max
                    score = torch.maximum(qk_min, qk_max).sum().item()
                    
                    # 存储
                    estimated_scores_cpu[bid, hid, pid] = score
        
        # 比较结果
        estimated_scores_gpu = estimated_scores.cpu()
        diff = torch.abs(estimated_scores_gpu - estimated_scores_cpu)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        # 统计有效元素（非零的 GPU 结果）
        valid_mask = estimated_scores_gpu != 0
        num_valid = valid_mask.sum().item()
        
        if num_valid > 0:
            max_diff_valid = diff[valid_mask].max().item()
            mean_diff_valid = diff[valid_mask].mean().item()
            max_val = estimated_scores_gpu[valid_mask].abs().max().item()
            relative_error = max_diff_valid / max_val if max_val > 0 else 0
        else:
            max_diff_valid = 0
            mean_diff_valid = 0
            relative_error = 0
        
        print(f"Batch Size: {batch_size}, Num Heads: {num_heads}, Head Dim: {head_dim}")
        print(f"Max Pages: {max_pages}, Page Size: {page_size}")
        print(f"有效元素数量: {num_valid} / {batch_size * num_heads * (max_pages - 1)}")
        print(f"\n全局差异统计:")
        print(f"  最大绝对误差: {max_diff:.6e}")
        print(f"  平均绝对误差: {mean_diff:.6e}")
        print(f"\n有效元素差异统计:")
        print(f"  最大绝对误差: {max_diff_valid:.6e}")
        print(f"  平均绝对误差: {mean_diff_valid:.6e}")
        print(f"  最大相对误差: {relative_error:.6e}")
        
        # 显示几个样本对比
        print(f"\n样本对比（前 5 个有效元素）:")
        count = 0
        for bid in range(batch_size):
            if count >= 5:
                break
            for hid in range(num_heads):
                if count >= 5:
                    break
                for pid in range(max_pages - 1):
                    if count >= 5:
                        break
                    if valid_mask[bid, hid, pid]:
                        gpu_val = estimated_scores_gpu[bid, hid, pid].item()
                        cpu_val = estimated_scores_cpu[bid, hid, pid].item()
                        diff_val = diff[bid, hid, pid].item()
                        print(f"  [{bid},{hid},{pid}] GPU: {gpu_val:.6f}, CPU: {cpu_val:.6f}, Diff: {diff_val:.6e}")
                        count += 1
        
        # 判断是否通过
        TOLERANCE = 1e-2  # 相对误差容忍度
        if relative_error < TOLERANCE:
            print(f"\n✅ 验证通过！相对误差 {relative_error:.6e} < {TOLERANCE}")
        else:
            print(f"\n❌ 验证失败！相对误差 {relative_error:.6e} >= {TOLERANCE}")
            print("\n显示前 10 个最大误差的位置:")
            flat_diff = diff.flatten()
            flat_indices = torch.argsort(flat_diff, descending=True)[:10]
            for i, flat_idx in enumerate(flat_indices):
                idx = flat_idx.item()
                bid = idx // (num_heads * max_pages)
                hid = (idx // max_pages) % num_heads
                pid = idx % max_pages
                if pid < max_pages - 1:  # 只显示有效的 page
                    gpu_val = estimated_scores_gpu[bid, hid, pid].item()
                    cpu_val = estimated_scores_cpu[bid, hid, pid].item()
                    diff_val = diff[bid, hid, pid].item()
                    print(f"  #{i+1} [{bid},{hid},{pid}] GPU: {gpu_val:.6f}, CPU: {cpu_val:.6f}, Diff: {diff_val:.6e}")
        
        print("="*80 + "\n")


def quest_select_topk_pages(
    estimated_scores: torch.Tensor,     # [batch, num_heads, max_pages]
    seq_lens: torch.Tensor,             # [batch]
    quest_topk: int,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    TopK Selection: 选择得分最高的 K 个 pages + last page
    
    返回 CSR 格式的索引（简化版，所有 head 选中数量相同）：
    - kv_indptr: [batch + 1]  # 每个 batch 的 token 数量（每个 head 相同）
    - kv_indices: [total_selected_tokens]  # 所有 batch 和 head 的 token indices 拼接
    
    核心逻辑：
    1. 对每个 (batch, head)，在前 num_pages-1 个 pages 中选 TopK
    2. 强制追加 last_page_idx
    3. 展开 pages 为 token indices
    4. 转换为 CSR 格式（所有 head 共享相同的 token 数量）
    5. 支持 batch size > 1，通过 padding 最小值来处理不同的 seqlen
    """
    batch_size, num_heads, max_pages = estimated_scores.shape
    device = estimated_scores.device
    
    assert quest_topk > 0, "quest_topk must be greater than 0"
    
    # 计算每个 batch 的 num_pages 和 last_page
    # seq_lens: [batch]
    num_pages_per_batch = (seq_lens + page_size - 1) // page_size  # [batch]
    last_pages = num_pages_per_batch - 1  # [batch]
    
    # 为 topk 创建 mask，将无效 pages（超过实际 num_pages-1 的部分）设置为 -inf
    # estimated_scores: [batch, num_heads, max_pages]
    # 对于每个 batch，只有前 num_pages-1 个是有效的（排除 last page）
    nvtx.range_push("create mask for padding")
    page_indices = torch.arange(max_pages, device=device).unsqueeze(0)  # [1, max_pages]
    # valid_mask[b, p] = (p < num_pages_per_batch[b] - 1)
    valid_mask = page_indices < (num_pages_per_batch - 1).unsqueeze(-1)  # [batch, max_pages]
    valid_mask = valid_mask.unsqueeze(1)  # [batch, 1, max_pages] 扩展到 head 维度
    
    # 将无效位置的分数设置为 -inf，这样 topk 就不会选中它们
    scores_for_topk = estimated_scores.clone()  # [batch, num_heads, max_pages]
    scores_for_topk = torch.where(valid_mask, scores_for_topk, torch.tensor(float('-inf'), device=device))
    nvtx.range_pop()
    
    # 对于每个 batch，判断是否需要稀疏化
    # 如果 num_pages <= quest_topk + 1，则退化为稠密
    need_sparse = num_pages_per_batch > (quest_topk + 1)  # [batch]
    
    # 计算每个 batch 实际选择的 k（对于稠密情况，k = num_pages - 1）
    # k_per_batch[b] = min(quest_topk, num_pages_per_batch[b] - 1) if need_sparse[b] else num_pages_per_batch[b] - 1
    k_per_batch = torch.where(
        need_sparse,
        torch.minimum(torch.tensor(quest_topk, device=device), num_pages_per_batch - 1),
        num_pages_per_batch - 1
    )  # [batch]
    
    # 统一使用最大的 k 值来进行 topk 操作（方便批处理）
    max_k = k_per_batch.max().item()
    max_k = max(max_k, 1)  # 至少为 1
    
    # 对所有 batch 执行 topk（使用统一的 k）
    # scores_for_topk: [batch, num_heads, max_pages]
    nvtx.range_push("topk")
    topk_result = torch.topk(scores_for_topk, k=max_k, dim=-1)
    topk_pages = topk_result.indices  # [batch, num_heads, max_k]
    nvtx.range_pop()
    
    # 对于每个 batch，只保留前 k_per_batch[b] 个结果
    # 为了向量化，我们创建一个 mask
    nvtx.range_push("mask topk results")
    k_indices = torch.arange(max_k, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, max_k]
    topk_mask = k_indices < k_per_batch.unsqueeze(-1).unsqueeze(-1)  # [batch, 1, max_k]
    # 将超出 k_per_batch 的位置设置为 -1（稍后过滤）
    topk_pages = torch.where(topk_mask, topk_pages, torch.tensor(-1, dtype=torch.int32, device=device))
    nvtx.range_pop()
    
    # 追加 last page
    nvtx.range_push("append last page")
    last_page_tensor = last_pages.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_heads, 1)  # [batch, num_heads, 1]
    selected_pages = torch.cat([topk_pages, last_page_tensor], dim=-1)  # [batch, num_heads, max_k+1]
    nvtx.range_pop()
    
    # 展开 pages 为 token indices，构建 CSR 格式
    # selected_pages: [batch, num_heads, max_k+1]
    
    # 1. 计算每个 page 的起始 token 和有效 token 数量（全向量化）
    nvtx.range_push("calculate page starts and ends")
    page_starts = selected_pages * page_size  # [batch, num_heads, max_k+1]
    # 需要对每个 batch 使用其对应的 seq_len
    seq_lens_expanded = seq_lens.unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
    page_ends = torch.clamp(page_starts + page_size, max=seq_lens_expanded)  # [batch, num_heads, max_k+1]
    page_lens = page_ends - page_starts  # [batch, num_heads, max_k+1]
    
    # 对于 selected_pages == -1 的位置（被 mask 掉的），page_lens 应该为 0
    page_lens = torch.where(selected_pages >= 0, page_lens, torch.tensor(0, dtype=torch.int32, device=device))
    nvtx.range_pop()

    nvtx.range_push("calculate tokens count")
    # 2. 计算 kv_indptr（每个 batch 的 token 数量）
    # 对于每个 batch，所有 head 选中的 token 数量应该相同
    tokens_per_head = page_lens.sum(dim=-1)  # [batch, num_heads]
    # 验证每个 batch 内所有 head 的 token 数量相同
    # assert (tokens_per_head.max(dim=1)[0] == tokens_per_head.min(dim=1)[0]).all(), "All heads should have the same number of tokens within each batch"
    tokens_per_batch = tokens_per_head[:, 0]  # [batch]
    
    # 构建 kv_indptr: [batch+1]
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(tokens_per_batch, dim=0)
    nvtx.range_pop()

    # 3. 批量生成所有 token indices（关键优化）
    nvtx.range_push("generate token indices")
    max_page_len = page_lens.max().item()

    if max_page_len > 0:
        # 生成基础 offsets: [0, 1, 2, ..., max_page_len-1]
        offsets = torch.arange(max_page_len, dtype=torch.int32, device=device)  # [max_page_len]
        
        # 广播生成所有 (batch, head, page) 的 tokens: [batch, num_heads, max_k+1, max_page_len]
        page_starts_expanded = page_starts.unsqueeze(-1)  # [batch, num_heads, max_k+1, 1]
        page_lens_expanded = page_lens.unsqueeze(-1)  # [batch, num_heads, max_k+1, 1]
        offsets_expanded = offsets.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, max_page_len]
        
        # tokens[b, h, p, t] = page_starts[b, h, p] + t
        all_tokens = page_starts_expanded + offsets_expanded  # [batch, num_heads, max_k+1, max_page_len]
        
        # mask: 只保留有效 tokens (t < page_lens[b, h, p])
        mask = offsets_expanded < page_lens_expanded  # [batch, num_heads, max_k+1, max_page_len]
        
        # 展平并过滤（保持 batch, head 顺序）
        kv_indices = all_tokens[mask]  # [total_tokens]
    else:
        # 空序列
        kv_indices = torch.empty(0, dtype=torch.int32, device=device)
    nvtx.range_pop()

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
    
    nvtx.range_push("max pages")
    # 计算最大的 page 数量
    max_pages = ((seq_lens.max().item() + page_size - 1) // page_size)
    nvtx.range_pop()
    
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
