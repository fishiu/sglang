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

from sglang.srt.managers.schedule_batch import global_server_args_dict


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
    kv_indptr,          # [batch + 1, num_heads] - per-head CSR pointers
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
    stride_indptr_b,    # kv_indptr 的 batch stride 20
    stride_indptr_h,    # kv_indptr 的 head stride 1
    kv_group_num: tl.constexpr,  # 1
    BLOCK_DMODEL: tl.constexpr,  # 128
    BLOCK_DV: tl.constexpr,  # 128
    BLOCK_N: tl.constexpr,  # 64
    MIN_BLOCK_KV: tl.constexpr,  # 16
    logit_cap: tl.constexpr,  # 0.0
    Lk: tl.constexpr,  # 128
    Lv: tl.constexpr,  # 128
):
    """
    Quest Decode Stage1: 支持 per-head 的 kv_indptr
    
    与标准 decode kernel 的主要区别：
    - kv_indptr 有 head 维度：[batch+1, num_heads]
    - 每个 head 从 kv_indptr[cur_batch, cur_head] 获取起始位置
    """
    cur_batch = tl.program_id(0)  # 0
    cur_head = tl.program_id(1)  # 0
    split_kv_id = tl.program_id(2)  # 0
    
    cur_kv_head = cur_head // kv_group_num  # 0 // 1 = 0
    
    offs_d = tl.arange(0, BLOCK_DMODEL)  # 0 ... 127
    offs_dv = tl.arange(0, BLOCK_DV)  # 0 ... 127
    mask_d = offs_d < Lk  # true, true, ..., true
    mask_dv = offs_dv < Lv # true, true, ..., true
    
    # 从 per-head kv_indptr 获取当前 (batch, head) 的 KV 范围
    # kv_indptr: [batch+1, num_heads]
    cur_batch_kv_start_idx = tl.load(
        kv_indptr + cur_batch * stride_indptr_b + cur_head * stride_indptr_h
    )
    cur_batch_kv_end_idx = tl.load(
        kv_indptr + (cur_batch + 1) * stride_indptr_b + cur_head * stride_indptr_h
    )  # 257
    cur_batch_seq_len = cur_batch_kv_end_idx - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)
    
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    
    # 计算当前 split 的范围
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
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
def _quest_decode_kernel_stage2(
    Mid_O,          # [batch, num_heads, max_kv_splits, v_head_dim]
    Mid_O_1,        # [batch, num_heads, max_kv_splits]
    O,              # [batch, num_heads, v_head_dim]
    kv_indptr,      # [batch + 1, num_heads]
    num_kv_splits,  # [batch]
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    stride_indptr_b,
    stride_indptr_h,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    """
    Quest Decode Stage2: 聚合 stage1 的结果
    
    与标准版本一致，只是 kv_indptr 有 head 维度
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    
    # 从 per-head kv_indptr 获取序列长度
    cur_batch_seq_len = tl.load(
        kv_indptr + (cur_batch + 1) * stride_indptr_b + cur_head * stride_indptr_h
    ) - tl.load(
        kv_indptr + cur_batch * stride_indptr_b + cur_head * stride_indptr_h
    )
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
    kv_indptr,      # [batch + 1, num_heads] - per-head CSR pointers
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
    
    与标准 decode_attention_fwd 的区别：
    - kv_indptr 有 head 维度：[batch+1, num_heads]
    - 每个 head 可以有不同的选中 tokens（Quest TopK）
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
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        kv_indptr.stride(0),
        kv_indptr.stride(1),
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
    )

    if layer_id == 0:
        nvtx.range_pop()
    
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
        kv_indptr.stride(0),
        kv_indptr.stride(1),
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
    page_size: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_mp,
    stride_mh,
    stride_md,
    stride_sb,
    stride_sh,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Estimate 阶段：用元数据快速估算每个 page 的注意力得分上界
    
    Grid: (batch, num_heads, max_pages-1)  # 排除 last page
    
    算法：
        score[b,h,p] = Σ_d max(Q[b,h,d] * K_min[p,h,d], Q[b,h,d] * K_max[p,h,d])
    """
    # 获取当前 thread block 的索引
    bid = tl.program_id(0)  # batch index
    hid = tl.program_id(1)  # head index
    pid = tl.program_id(2)  # page index
    
    # 获取当前 batch 的序列长度
    seq_len = tl.load(Seq_lens + bid)
    num_pages = (seq_len + page_size - 1) // page_size
    
    # 过滤：只处理 [0, num_pages-2]，排除 last page
    if pid >= num_pages - 1:
        return
    
    # 加载 Q 向量: [head_dim]
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < BLOCK_DMODEL  # 假设 head_dim <= BLOCK_DMODEL
    q = tl.load(
        Q + bid * stride_qb + hid * stride_qh + offs_d * stride_qd,
        mask=mask_d,
        other=0.0,
    )
    
    # 加载元数据: K_min 和 K_max
    # K_metadata shape: [num_pages, num_heads, head_dim, 2]
    k_min = tl.load(
        K_metadata + pid * stride_mp + hid * stride_mh + offs_d * stride_md * 2 + 0 * stride_md,
        mask=mask_d,
        other=0.0,
    )
    k_max = tl.load(
        K_metadata + pid * stride_mp + hid * stride_mh + offs_d * stride_md * 2 + 1 * stride_md,
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
    quest_estimate_kernel[grid](
        q,
        k_metadata,
        seq_lens,
        estimated_scores,
        page_size,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_metadata.stride(0),
        k_metadata.stride(1),
        k_metadata.stride(2),
        estimated_scores.stride(0),
        estimated_scores.stride(1),
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=2,
    )


def quest_select_topk_pages(
    estimated_scores: torch.Tensor,     # [batch, num_heads, max_pages]
    seq_lens: torch.Tensor,             # [batch]
    quest_topk: int,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    TopK Selection: 选择得分最高的 K 个 pages + last page
    
    返回 CSR 格式的索引（per-head）：
    - kv_indptr: [batch + 1, num_heads]  # 注意：第一维是 batch+1，第二维是 num_heads
    - kv_indices: [total_selected_tokens]
    
    核心逻辑：
    1. 对每个 (batch, head)，在前 num_pages-1 个 pages 中选 TopK
    2. 强制追加 last_page_idx
    3. 展开 pages 为 token indices
    4. 转换为 CSR 格式（per-head）
    """
    batch_size, num_heads, max_pages = estimated_scores.shape
    device = estimated_scores.device
    
    # 为简化实现，先处理 batch_size=1 的情况
    assert batch_size == 1, "Quest 简化版仅支持 batch_size=1"
    assert quest_topk > 0, "quest_topk must be greater than 0"
    
    batch_idx = 0
    seq_len = seq_lens[batch_idx].item()
    num_pages = (seq_len + page_size - 1) // page_size
    last_page = num_pages - 1
    
    # 判断是否需要稀疏化
    if num_pages <= quest_topk + 1:
        # 退化为稠密：保留所有 pages
        # kv_indptr: [batch+1, num_heads] = [2, num_heads]
        # 第一个 batch 行全为 0，第二个 batch 行全为 seq_len
        kv_indptr = torch.zeros((batch_size + 1, num_heads), dtype=torch.int32, device=device)
        kv_indptr[1, :] = seq_len
        
        # kv_indices: 每个 head 都是完整的 [0, 1, ..., seq_len-1]
        kv_indices = torch.arange(seq_len, dtype=torch.int32, device=device)
        kv_indices = kv_indices.unsqueeze(0).expand(num_heads, -1).reshape(-1)
        
        return kv_indptr, kv_indices
    
    # 稀疏路径：TopK + last page
    k = min(quest_topk, num_pages - 1)
    
    # [num_heads, max_pages] -> TopK
    scores_for_topk = estimated_scores[batch_idx, :, :num_pages - 1]  # [num_heads, num_pages-1]
    topk_pages = torch.topk(scores_for_topk, k=k, dim=-1).indices  # [num_heads, k]
    
    # 追加 last page
    last_page_tensor = torch.full(
        (num_heads, 1), last_page, dtype=torch.int32, device=device
    )
    selected_pages = torch.cat([topk_pages, last_page_tensor], dim=-1)  # [num_heads, k+1]
    
    # 展开 pages 为 token indices，构建 CSR 格式
    # ========== 向量化优化：消除 .item() 和循环中的 kernel launch ==========
    # selected_pages: [num_heads, k+1]
    
    # 1. 计算每个 page 的起始 token 和有效 token 数量（全向量化）
    page_starts = selected_pages * page_size  # [num_heads, k+1]
    page_ends = torch.clamp(page_starts + page_size, max=seq_len)  # [num_heads, k+1]
    page_lens = page_ends - page_starts  # [num_heads, k+1] - 每个 page 的有效 token 数
    
    # 2. 计算 kv_indptr（每个 head 的累积 token 数）
    tokens_per_head = page_lens.sum(dim=1)  # [num_heads]
    kv_indptr = torch.zeros((batch_size + 1, num_heads), dtype=torch.int32, device=device)
    kv_indptr[1, :] = torch.cumsum(tokens_per_head, dim=0)  # 累积和
    
    # 3. 批量生成所有 token indices（关键优化）
    # 使用 repeat_interleave 避免循环
    total_tokens = kv_indptr[1, -1].item()
    
    # 方法：为每个 (head, page) 生成其对应的 token indices
    # 使用 repeat + mask 批量生成
    max_page_len = page_lens.max().item()
    
    if max_page_len > 0:
        # 生成基础 offsets: [0, 1, 2, ..., max_page_len-1]
        offsets = torch.arange(max_page_len, dtype=torch.int32, device=device)  # [max_page_len]
        
        # 广播生成所有 (head, page) 的 tokens: [num_heads, k+1, max_page_len]
        page_starts_expanded = page_starts.unsqueeze(-1)  # [num_heads, k+1, 1]
        page_lens_expanded = page_lens.unsqueeze(-1)  # [num_heads, k+1, 1]
        offsets_expanded = offsets.unsqueeze(0).unsqueeze(0)  # [1, 1, max_page_len]
        
        # tokens[h, p, t] = page_starts[h, p] + t
        all_tokens = page_starts_expanded + offsets_expanded  # [num_heads, k+1, max_page_len]
        
        # mask: 只保留有效 tokens (t < page_lens[h, p])
        mask = offsets_expanded < page_lens_expanded  # [num_heads, k+1, max_page_len]
        
        # 展平并过滤（保持 heads 顺序）
        kv_indices = all_tokens[mask]  # [total_tokens]
    else:
        # 空序列
        kv_indices = torch.empty(0, dtype=torch.int32, device=device)
    
    return kv_indptr, kv_indices


@triton.jit
def quest_lazy_init_metadata_kernel(
    K_buffer,               # [size, num_heads, head_dim] - 完整 KV cache
    K_metadata,             # [num_pages, num_heads, head_dim, 2] - 元数据 buffer
    Metadata_init,          # [num_pages, num_heads] - 初始化标记
    page_idx,               # scalar: 要初始化的 page index
    seq_len,                # scalar: 当前序列长度
    page_size: tl.constexpr,
    num_heads: tl.constexpr,
    stride_kb,
    stride_kh,
    stride_kd,
    stride_mp,
    stride_mh,
    stride_md,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_PAGE: tl.constexpr,
):
    """
    懒初始化：从 KV cache 回读一个 page 的所有 K 向量，计算 min/max
    
    Grid: (num_heads,)
    
    用于 decode 第一步，处理 prefill 阶段遗留的 pages
    """
    hid = tl.program_id(0)  # head index
    
    # 检查是否已初始化
    is_init = tl.load(Metadata_init + page_idx * num_heads + hid)
    if is_init:
        return
    
    # 计算该 page 的 token 范围
    page_start = page_idx * page_size
    page_end = tl.minimum((page_idx + 1) * page_size, seq_len)
    page_len = page_end - page_start
    
    if page_len <= 0:
        return
    
    # 初始化 min/max
    offs_d = tl.arange(0, BLOCK_DMODEL)
    k_min = tl.full([BLOCK_DMODEL], 65504.0, dtype=tl.float32)  # FP16 max
    k_max = tl.full([BLOCK_DMODEL], -65504.0, dtype=tl.float32)
    
    # 遍历该 page 的所有 tokens
    for t_offset in range(0, page_len):
        token_idx = page_start + t_offset
        
        # 加载 K 向量（显式转换为 float32）
        k = tl.load(
            K_buffer + token_idx * stride_kb + hid * stride_kh + offs_d * stride_kd,
            mask=offs_d < BLOCK_DMODEL,
            other=0.0,
        ).to(tl.float32)
        
        # 更新 min/max
        k_min = tl.minimum(k_min, k)
        k_max = tl.maximum(k_max, k)
    
    # 存储元数据
    tl.store(
        K_metadata + page_idx * stride_mp + hid * stride_mh + offs_d * stride_md * 2 + 0 * stride_md,
        k_min,
        mask=offs_d < BLOCK_DMODEL,
    )
    tl.store(
        K_metadata + page_idx * stride_mp + hid * stride_mh + offs_d * stride_md * 2 + 1 * stride_md,
        k_max,
        mask=offs_d < BLOCK_DMODEL,
    )
    
    # 标记已初始化
    tl.store(Metadata_init + page_idx * num_heads + hid, True)


@triton.jit
def quest_update_metadata_kernel(
    K_new,                  # [batch, num_heads, head_dim] - 当前 step 的新 K
    K_buffer,               # [size, num_heads, head_dim] - 完整 KV cache
    K_metadata,             # [num_pages, num_heads, head_dim, 2]
    Metadata_init,          # [num_pages, num_heads]
    Out_cache_loc,          # [batch] - 写入位置（token index）
    Seq_lens,               # [batch] - 当前序列长度
    page_size: tl.constexpr,
    num_heads: tl.constexpr,
    stride_kb_new,
    stride_kh_new,
    stride_kd_new,
    stride_kb,
    stride_kh,
    stride_kd,
    stride_mp,
    stride_mh,
    stride_md,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    增量更新元数据：写入新 K 时同步更新 page 的 min/max
    
    Grid: (batch, num_heads)
    
    逻辑：
    1. 如果 page 未初始化 -> 调用懒初始化（回读整个 page）
    2. 否则 -> 增量更新 min/max
    """
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    
    # 获取写入位置
    loc = tl.load(Out_cache_loc + bid)
    seq_len = tl.load(Seq_lens + bid)
    
    # 计算 page index 和 offset
    page_idx = loc // page_size
    page_offset = loc % page_size
    
    # 加载新 K（显式转换为 float32 以保持类型一致性）
    offs_d = tl.arange(0, BLOCK_DMODEL)
    k_new = tl.load(
        K_new + bid * stride_kb_new + hid * stride_kh_new + offs_d * stride_kd_new,
        mask=offs_d < BLOCK_DMODEL,
        other=0.0,
    ).to(tl.float32)  # 显式转换为 float32
    
    # 检查是否需要懒初始化
    is_init = tl.load(Metadata_init + page_idx * num_heads + hid)
    
    if not is_init:
        # 懒初始化：回读该 page 的所有已有 tokens
        # 注意：这里假设 page_offset > 0，即 page 内已有其他 tokens（来自 prefill）
        # 如果 page_offset == 0，说明这是新 page，直接初始化为 k_new
        
        # 初始化为 float32（类型一致性）
        k_min = k_new
        k_max = k_new
        
        # 回读该 page 的其他 tokens
        for t in range(page_offset):
            token_idx = page_idx * page_size + t
            k_t = tl.load(
                K_buffer + token_idx * stride_kb + hid * stride_kh + offs_d * stride_kd,
                mask=offs_d < BLOCK_DMODEL,
                other=0.0,
            ).to(tl.float32)  # 显式转换为 float32
            k_min = tl.minimum(k_min, k_t)
            k_max = tl.maximum(k_max, k_t)
        
        # 存储初始化的 min/max
        tl.store(
            K_metadata + page_idx * stride_mp + hid * stride_mh + offs_d * stride_md * 2 + 0 * stride_md,
            k_min,
            mask=offs_d < BLOCK_DMODEL,
        )
        tl.store(
            K_metadata + page_idx * stride_mp + hid * stride_mh + offs_d * stride_md * 2 + 1 * stride_md,
            k_max,
            mask=offs_d < BLOCK_DMODEL,
        )
        
        # 标记已初始化
        tl.store(Metadata_init + page_idx * num_heads + hid, True)
    
    else:
        # 增量更新
        old_min = tl.load(
            K_metadata + page_idx * stride_mp + hid * stride_mh + offs_d * stride_md * 2 + 0 * stride_md,
            mask=offs_d < BLOCK_DMODEL,
            other=65504.0,
        ).to(tl.float32)  # 显式转换为 float32
        old_max = tl.load(
            K_metadata + page_idx * stride_mp + hid * stride_mh + offs_d * stride_md * 2 + 1 * stride_md,
            mask=offs_d < BLOCK_DMODEL,
            other=-65504.0,
        ).to(tl.float32)  # 显式转换为 float32
        
        new_min = tl.minimum(old_min, k_new)
        new_max = tl.maximum(old_max, k_new)
        
        tl.store(
            K_metadata + page_idx * stride_mp + hid * stride_mh + offs_d * stride_md * 2 + 0 * stride_md,
            new_min,
            mask=offs_d < BLOCK_DMODEL,
        )
        tl.store(
            K_metadata + page_idx * stride_mp + hid * stride_mh + offs_d * stride_md * 2 + 1 * stride_md,
            new_max,
            mask=offs_d < BLOCK_DMODEL,
        )


def quest_update_kv_and_metadata(
    k_new: torch.Tensor,                # [batch, num_heads, head_dim]
    v_new: torch.Tensor,                # [batch, num_heads, head_dim]
    k_buffer: torch.Tensor,             # [size, num_heads, head_dim]
    v_buffer: torch.Tensor,             # [size, num_heads, head_dim]
    k_metadata: torch.Tensor,           # [num_pages, num_heads, head_dim, 2]
    metadata_init: torch.Tensor,        # [num_pages, num_heads]
    out_cache_loc: torch.Tensor,        # [batch]
    seq_lens: torch.Tensor,             # [batch]
    page_size: int,
):
    """
    Python 封装：同步更新 KV cache 和元数据
    
    步骤：
    1. 写入 KV cache
    2. 更新元数据（懒初始化或增量更新）
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
        k_buffer,
        k_metadata,
        metadata_init,
        out_cache_loc,
        seq_lens,
        page_size,
        num_heads,
        k_new.stride(0),
        k_new.stride(1),
        k_new.stride(2),
        k_buffer.stride(0),
        k_buffer.stride(1),
        k_buffer.stride(2),
        k_metadata.stride(0),
        k_metadata.stride(1),
        k_metadata.stride(2),
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=2,
    )


@triton.jit
def quest_init_all_pages_kernel(
    K_buffer,               # [size, num_heads, head_dim]
    K_metadata,             # [num_pages, num_heads, head_dim, 2]
    Metadata_init,          # [num_pages, num_heads]
    seq_len,                # scalar: 当前序列长度
    page_size: tl.constexpr,
    stride_kb,
    stride_kh,
    stride_kd,
    stride_mp,
    stride_mh,
    stride_md,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    预初始化：并行处理所有 prefill pages 的元数据
    
    Grid: (num_pages, num_heads)
    
    用于第一次 decode 前，一次性初始化所有 prefill pages
    """
    page_idx = tl.program_id(0)
    hid = tl.program_id(1)
    
    # 检查是否已初始化
    is_init = tl.load(Metadata_init + page_idx * tl.num_programs(1) + hid)
    if is_init:
        return
    
    # 计算该 page 的 token 范围
    page_start = page_idx * page_size
    page_end = tl.minimum((page_idx + 1) * page_size, seq_len)
    page_len = page_end - page_start
    
    if page_len <= 0:
        return
    
    # 初始化 min/max
    offs_d = tl.arange(0, BLOCK_DMODEL)
    k_min = tl.full([BLOCK_DMODEL], 65504.0, dtype=tl.float32)  # FP16 max
    k_max = tl.full([BLOCK_DMODEL], -65504.0, dtype=tl.float32)
    
    # 遍历该 page 的所有 tokens
    for t_offset in range(0, page_len):
        token_idx = page_start + t_offset
        
        # 加载 K 向量（显式转换为 float32）
        k = tl.load(
            K_buffer + token_idx * stride_kb + hid * stride_kh + offs_d * stride_kd,
            mask=offs_d < BLOCK_DMODEL,
            other=0.0,
        ).to(tl.float32)
        
        # 更新 min/max
        k_min = tl.minimum(k_min, k)
        k_max = tl.maximum(k_max, k)
    
    # 存储元数据
    tl.store(
        K_metadata + page_idx * stride_mp + hid * stride_mh + offs_d * stride_md * 2 + 0 * stride_md,
        k_min,
        mask=offs_d < BLOCK_DMODEL,
    )
    tl.store(
        K_metadata + page_idx * stride_mp + hid * stride_mh + offs_d * stride_md * 2 + 1 * stride_md,
        k_max,
        mask=offs_d < BLOCK_DMODEL,
    )
    
    # 标记已初始化
    tl.store(Metadata_init + page_idx * tl.num_programs(1) + hid, True)


def quest_init_all_prefill_pages(
    k_buffer: torch.Tensor,             # [size, num_heads, head_dim]
    k_metadata: torch.Tensor,           # [num_pages, num_heads, head_dim, 2]
    metadata_init: torch.Tensor,        # [num_pages, num_heads]
    seq_len: int,
    page_size: int,
):
    """
    预初始化所有 prefill pages 的元数据
    
    在第一次 decode 前调用，一次性初始化所有 pages（方案 A）
    """
    num_pages = (seq_len + page_size - 1) // page_size
    num_heads = k_buffer.shape[1]
    head_dim = k_buffer.shape[2]
    
    if num_pages == 0:
        return
    
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)
    grid = (num_pages, num_heads)
    
    quest_init_all_pages_kernel[grid](
        k_buffer,
        k_metadata,
        metadata_init,
        seq_len,
        page_size,
        k_buffer.stride(0),
        k_buffer.stride(1),
        k_buffer.stride(2),
        k_metadata.stride(0),
        k_metadata.stride(1),
        k_metadata.stride(2),
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=2,
    )

