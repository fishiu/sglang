"""
Quest Attention Backend - Query-Aware Sparsity for Efficient Long-Context Inference

实现基于 page-level 元数据的三阶段稀疏注意力：
1. Estimate: 用 min/max 元数据快速估算 page 重要性
2. TopK Selection: 选择最重要的 K 个 pages + last page
3. Sparse Attention: 只对选中 pages 做完整注意力

简化版特性：
- 仅支持 MHA（num_q_heads == num_kv_heads）
- Decode 阶段使用 Quest，Extend 阶段使用稠密 attention
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union
import os

import torch
import torch.cuda.nvtx as nvtx

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.layers.dp_attention import get_attention_tp_size

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class QuestAttnBackend(AttentionBackend):
    """Quest 注意力后端 - 基于 page-level 元数据的稀疏注意力"""
    
    def __init__(self, model_runner: ModelRunner):
        """
        初始化 Quest backend
        
        参数:
        - model_runner: 模型运行器，提供配置参数
        
        配置项：
        - quest_topk: 保留的最大 page 数（不含 last page）
        - page_size: 每个 page 的 token 数
        
        设计：
        - Decode 阶段：使用 Quest 稀疏化
        - Extend 阶段：Delegate 给 TritonAttnBackend（稠密 attention）
        """
        super().__init__()
        
        # 延迟导入避免 CUDA context 初始化
        from sglang.srt.layers.attention.triton_ops.quest_attention import (
            quest_estimate_scores_triton,
            quest_estimate_scores_torch,
            quest_select_topk_pages,
            quest_select_topk_pages_into,
            quest_update_kv_and_metadata,
            quest_compute_extend_metadata,
            quest_decode_attention_fwd,
        )
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
        
        self.quest_estimate_scores_triton = quest_estimate_scores_triton
        self.quest_estimate_scores_torch = quest_estimate_scores_torch
        self.quest_select_topk_pages = quest_select_topk_pages
        self.quest_select_topk_pages_into = quest_select_topk_pages_into
        self.quest_update_kv_and_metadata = quest_update_kv_and_metadata
        self.quest_compute_extend_metadata = quest_compute_extend_metadata
        self.quest_decode_attention_fwd = quest_decode_attention_fwd
        
        # 配置参数
        self.quest_topk = model_runner.server_args.quest_topk
        self.page_size = model_runner.server_args.page_size
        # Per-TP head counts (Q and KV)
        tp_size = get_attention_tp_size()
        self.num_q_head = model_runner.model_config.num_attention_heads // tp_size
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(tp_size)
        # Use Q-head count where a single head count is required
        self.num_head = self.num_q_head
        # Per-head dimension
        self.head_dim = model_runner.model_config.head_dim
        # Quest GQA mode: when True, keep per-Q-head selection (weak GQA).
        # When False, group Q heads that share a KV head for estimate/select (strong GQA).
        self.use_weak_gqa = getattr(model_runner.server_args, "quest_use_weak_gqa", False)
        self.quest_estimate_kernel = getattr(model_runner.server_args, "quest_estimate_kernel", "triton")
        self.quest_topk_kernel = getattr(model_runner.server_args, "quest_topk_kernel", "max")
        self.quest_estimate_split = getattr(model_runner.server_args, "quest_estimate_split", True)
        
        # Extend 阶段 delegation：创建 TritonAttnBackend 实例
        # 用于处理非 Quest 的 extend 路径
        self.triton_backend = TritonAttnBackend(model_runner)
        
        # 临时缓冲（在 init_forward_metadata 中分配）
        self.estimated_scores: torch.Tensor = None
        self.attn_logits: torch.Tensor = None
        self.attn_lse: torch.Tensor = None
        
        # 元数据（由 forward_metadata 提供）
        self.forward_metadata = None
        
        # Decode 配置
        self.max_kv_splits = model_runner.server_args.triton_attention_num_kv_splits
        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1]
        self.device = model_runner.device
        self.max_context_len = model_runner.model_config.context_len
        # TODO(xiaoyuan): compute splits num dynamiclly
        self.estimate_splits: int = model_runner.server_args.quest_estimate_splits

        # CUDA Graph related buffers (lazily initialized)
        self.cuda_graph_attn_logits: Optional[torch.Tensor] = None
        self.cuda_graph_attn_lse: Optional[torch.Tensor] = None
        self.cuda_graph_num_kv_splits: Optional[torch.Tensor] = None
        self.cuda_graph_kv_indices: Optional[torch.Tensor] = None
        self.kv_indptr: Optional[torch.Tensor] = None
        # Quest-specific graph scratch buffers (for in-graph update/estimate/select)
        self.cuda_graph_estimated_scores: Optional[torch.Tensor] = None
        self.cuda_graph_selected_pages: Optional[torch.Tensor] = None
        self.cuda_graph_kv_indices_buf: Optional[torch.Tensor] = None
        self.cuda_graph_tokens_per_head: Optional[torch.Tensor] = None
        self.cuda_graph_tokens_per_batch: Optional[torch.Tensor] = None
        # Flag indicating graph kv buffers are populated for current capture/replay context
        self._graph_meta_ready: bool = False
    
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """初始化前向传播的元数据和临时缓冲"""
        
        if forward_batch.forward_mode.is_decode():
            batch_size = forward_batch.batch_size
            max_seq_len = torch.max(forward_batch.seq_lens).item()
            max_pages = (max_seq_len + self.page_size - 1) // self.page_size
            
            # 分配 Estimate 阶段的输出缓冲
            # [batch, num_heads, max_pages]
            # 注意：last page 的得分保持为 0（初始化时默认为 0）
            self.estimated_scores = torch.zeros(
                (batch_size, self.num_head, max_pages),
                dtype=torch.float32,
                device="cuda",
            )
            
            # 分配稠密 decode 路径的临时缓冲（可能用于 fallback）
            self.attn_logits = torch.empty(
                (batch_size, self.num_head, self.max_kv_splits, self.v_head_dim),
                dtype=torch.float32,
                device="cuda",
            )
            self.attn_lse = torch.empty(
                (batch_size, self.num_head, self.max_kv_splits),
                dtype=torch.float32,
                device="cuda",
            )
            
            # 保存批次信息
            self.forward_metadata = {
                "batch_size": batch_size,
                "max_seq_len": max_seq_len,
                "max_pages": max_pages,
                "seq_lens": forward_batch.seq_lens,
                "req_pool_indices": forward_batch.req_pool_indices,
            }
        else:
            # Extend 阶段：Delegate 给 TritonAttnBackend
            self.triton_backend.init_forward_metadata(forward_batch)
            self.forward_metadata = None
    
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        """
        Extend 阶段：先计算元数据，再执行 attention
        
        流程：
        1. 计算当前层的 page 元数据（利用输入 k 的数据局部性）
        2. 使用标准 Triton extend attention（会写入 KV cache）
        """
        # Optional debug-time check: ensure per-request tokens are page-aligned in the KV pool.
        if os.environ.get("SGLANG_QUEST_CHECK_ALIGN", "0") == "1" and not get_is_capture_mode():
            self._check_page_alignment(forward_batch)
        # Step 1: 计算 page 元数据（趁 k 还在 cache 中）
        if save_kv_cache:
            nvtx.range_push("quest_compute_extend_metadata")
            self.quest_compute_extend_metadata(
                # For GQA, metadata is per KV head
                k_new=k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                k_metadata=forward_batch.token_to_kv_pool.get_metadata_buffer(layer.layer_id),
                out_cache_loc=forward_batch.out_cache_loc,
                extend_start_loc=forward_batch.extend_start_loc,
                seq_lens=forward_batch.extend_seq_lens,
                page_size=self.page_size,
                max_pages_cap=(self.max_context_len + self.page_size - 1) // self.page_size,
            )
            nvtx.range_pop()
        
        # TODO: consider merge set_kv_buffer into quest_compute_extend_metadata
        # Step 2: 标准 extend attention（内部会调用 set_kv_buffer 写入 KV cache）
        nvtx.range_push("triton_attn_forward_extend")
        output = self.triton_backend.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache
        )
        nvtx.range_pop()
        
        return output
    
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        """
        Decode 阶段：Quest 三阶段稀疏注意力
        
        流程：
        1. 更新 KV cache & 元数据（增量更新）
        2. Estimate: 用元数据估算 page scores
        3. TopK Selection: 选择重要 pages + last page
        4. Sparse Attention: 只对选中 pages 做完整 attention
        """
        # Optional debug-time check: ensure per-request tokens are page-aligned in the KV pool.
        if os.environ.get("SGLANG_QUEST_CHECK_ALIGN", "0") == "1" and not get_is_capture_mode():
            self._check_page_alignment(forward_batch)
        # Reshape q（处理 torch.compile 的 3D 输出问题）
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        
        # 分配输出 buffer
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)
        
        # If CUDA Graph metadata is pre-populated, skip append/estimate/select
        use_graph_meta = (
            isinstance(self.forward_metadata, dict)
            and "kv_indptr" in self.forward_metadata
            and "kv_indices" in self.forward_metadata
        )

        # Attention pattern
        kv_group_num = 1
        if layer.tp_k_head_num > 0:
            kv_group_num = layer.tp_q_head_num // layer.tp_k_head_num
        is_mha = kv_group_num == 1
        use_weak = self.use_weak_gqa or is_mha

        if not use_graph_meta:
            # Step 1: 更新 KV cache 和元数据
            if save_kv_cache:
                nvtx.range_push("quest_update_kv_and_metadata")
                self.quest_update_kv_and_metadata(
                    k_new=k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                    v_new=v.view(-1, layer.tp_k_head_num, layer.v_head_dim),
                    k_buffer=forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                    v_buffer=forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                    k_metadata=forward_batch.token_to_kv_pool.get_metadata_buffer(layer.layer_id),
                    out_cache_loc=forward_batch.out_cache_loc,
                    page_size=self.page_size,
                )
                nvtx.range_pop()

            # Step 2: Estimate - 估算 page-level 注意力得分
            nvtx.range_push("quest_estimate_scores")
            q_3d = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
            req_to_token_pool = forward_batch.req_to_token_pool.req_to_token
            req_pool_indices = forward_batch.req_pool_indices
            if use_weak:
                # Weak GQA / MHA: estimate per Q head -> [B, Hq, max_pages]
                est_buf = self.estimated_scores
                grouped_flag = False
            else:
                # Strong GQA: estimate per KV head -> [B, Hkv, max_pages]
                est_buf = self.estimated_scores[:, : self.num_kv_head, :]
                grouped_flag = True

            if self.quest_estimate_kernel == "triton":
                estimate_splits = self.estimate_splits if self.quest_estimate_split else 0
                self.quest_estimate_scores_triton(
                    q=q_3d,
                    k_metadata=forward_batch.token_to_kv_pool.get_metadata_buffer(layer.layer_id),
                    seq_lens=forward_batch.seq_lens,
                    estimated_scores=est_buf,
                    page_size=self.page_size,
                    req_to_token=req_to_token_pool,
                    req_pool_indices=req_pool_indices,
                    num_page_splits=estimate_splits,
                    grouped=grouped_flag,
                )
            else:
                self.quest_estimate_scores_torch(
                    q=q_3d,
                    k_metadata=forward_batch.token_to_kv_pool.get_metadata_buffer(layer.layer_id),
                    seq_lens=forward_batch.seq_lens,
                    estimated_scores=est_buf,
                    page_size=self.page_size,
                    req_to_token=req_to_token_pool,
                    req_pool_indices=req_pool_indices,
                    num_page_splits=self.estimate_splits,
                    grouped=grouped_flag,
                    max_pages_to_process=self.forward_metadata["max_pages"],
                )
            nvtx.range_pop()

            # Step 3: TopK Selection - 选择重要 pages
            nvtx.range_push("quest_select_topk_pages")
            if use_weak:
                kv_indptr, kv_indices = self.quest_select_topk_pages(
                    estimated_scores=est_buf,
                    seq_lens=forward_batch.seq_lens,
                    req_to_token=forward_batch.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices
                    ],
                    quest_topk=self.quest_topk,
                    page_size=self.page_size,
                    kernel_type=self.quest_topk_kernel,
                )
            else:
                # Strong GQA: TopK per KV head using grouped wrapper
                kv_indptr, kv_indices = self.quest_select_topk_pages(
                    estimated_scores=est_buf,
                    seq_lens=forward_batch.seq_lens,
                    req_to_token=req_to_token_pool,
                    req_pool_indices=req_pool_indices,
                    quest_topk=self.quest_topk,
                    page_size=self.page_size,
                    kernel_type=self.quest_topk_kernel,
                )
            nvtx.range_pop()
            attn_logits = self.attn_logits
            attn_lse = self.attn_lse
            num_kv_splits = torch.full(
                (forward_batch.batch_size,), self.max_kv_splits, dtype=torch.int32, device="cuda"
            )
        else:
            # Graph path: compute update/estimate/select in-graph with static buffers
            kv_indptr = self.forward_metadata["kv_indptr"]
            kv_indices = self.forward_metadata["kv_indices"]
            attn_logits = self.forward_metadata["attn_logits"]
            attn_lse = self.forward_metadata["attn_lse"]
            num_kv_splits = self.forward_metadata["num_kv_splits"]

            # 1) Update KV and page metadata
            if save_kv_cache:
                nvtx.range_push("quest_update_kv_and_metadata")
                self.quest_update_kv_and_metadata(
                    k_new=k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                    v_new=v.view(-1, layer.tp_k_head_num, layer.v_head_dim),
                    k_buffer=forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                    v_buffer=forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                    k_metadata=forward_batch.token_to_kv_pool.get_metadata_buffer(layer.layer_id),
                    out_cache_loc=forward_batch.out_cache_loc,
                    page_size=self.page_size,
                )
                nvtx.range_pop()

            # 2) Estimate page scores into preallocated buffer (static max_pages capacity)
            nvtx.range_push("quest_estimate_scores")
            # Estimate buffer selection under CUDA Graph:
            # - If using split-kernel (estimate_splits>0), we MUST pass the full
            #   buffer to avoid out-of-bounds writes when valid_pages > capture-time pages_cap.
            # - If using page-parallel kernel, keep the pages_cap view to reduce empty CTAs.
            pages_cap = getattr(self, "_cg_estimate_pages", None)
            use_triton_estimate = self.quest_estimate_kernel == "triton"
            estimate_splits = self.estimate_splits if self.quest_estimate_split else 0
            
            if estimate_splits is not None and estimate_splits > 0:
                # For split-kernel we must pass the full buffer in page dim.
                if use_weak:
                    est_buf = self.cuda_graph_estimated_scores[: forward_batch.batch_size]
                else:
                    est_buf = self.cuda_graph_estimated_scores[
                        : forward_batch.batch_size, : self.num_kv_head
                    ]
            else:
                if pages_cap is None:
                    pages_cap = (self.max_context_len + self.page_size - 1) // self.page_size
                if use_weak:
                    est_buf = self.cuda_graph_estimated_scores[
                        : forward_batch.batch_size, :, : pages_cap
                    ]
                else:
                    est_buf = self.cuda_graph_estimated_scores[
                        : forward_batch.batch_size, : self.num_kv_head, : pages_cap
                    ]

            if use_triton_estimate:
                self.quest_estimate_scores_triton(
                    q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k_metadata=forward_batch.token_to_kv_pool.get_metadata_buffer(layer.layer_id),
                    seq_lens=forward_batch.seq_lens,
                    estimated_scores=est_buf,
                    page_size=self.page_size,
                    req_to_token=forward_batch.req_to_token_pool.req_to_token,
                    req_pool_indices=forward_batch.req_pool_indices,
                    num_page_splits=estimate_splits,
                    grouped=not use_weak,
                )
            else:
                self.quest_estimate_scores_torch(
                    q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k_metadata=forward_batch.token_to_kv_pool.get_metadata_buffer(layer.layer_id),
                    seq_lens=forward_batch.seq_lens,
                    estimated_scores=est_buf,
                    page_size=self.page_size,
                    req_to_token=forward_batch.req_to_token_pool.req_to_token,
                    req_pool_indices=forward_batch.req_pool_indices,
                    num_page_splits=self.estimate_splits,
                    grouped=not use_weak,
                    max_pages_to_process=pages_cap,
                )
            nvtx.range_pop()

            # 3) Select+Expand+Pack into graph kv buffers (no allocations)
            nvtx.range_push("quest_select_expand_pack")
            bs_now = forward_batch.batch_size
            kv_indptr.zero_()
            if use_weak:
                # Weak GQA / MHA: per-Q-head select/expand/pack
                self.quest_select_topk_pages_into(
                    estimated_scores=est_buf,
                    seq_lens=forward_batch.seq_lens,
                    req_to_token=forward_batch.req_to_token_pool.req_to_token,
                    req_pool_indices=forward_batch.req_pool_indices,
                    quest_topk=self.quest_topk,
                    page_size=self.page_size,
                    selected_pages=self.cuda_graph_selected_pages[:bs_now],
                    kv_indices_buf=self.cuda_graph_kv_indices_buf[: bs_now * self.num_head],
                    tokens_per_head=self.cuda_graph_tokens_per_head[: bs_now * self.num_head],
                    tokens_per_batch=self.cuda_graph_tokens_per_batch[:bs_now],
                    kv_indptr=kv_indptr[: bs_now + 1],
                    kv_indices=kv_indices,
                    num_heads=self.num_head,
                    kernel_type=self.quest_topk_kernel,
                )
            else:
                # Strong GQA: per-KV-head select/expand/pack，使用 grouped wrapper。
                self.quest_select_topk_pages_into(
                    estimated_scores=est_buf,
                    seq_lens=forward_batch.seq_lens,
                    req_to_token=forward_batch.req_to_token_pool.req_to_token,
                    req_pool_indices=forward_batch.req_pool_indices,
                    quest_topk=self.quest_topk,
                    page_size=self.page_size,
                    selected_pages=self.cuda_graph_selected_pages[:bs_now, : self.num_kv_head],
                    kv_indices_buf=self.cuda_graph_kv_indices_buf[: bs_now * self.num_kv_head],
                    tokens_per_head=self.cuda_graph_tokens_per_head[: bs_now * self.num_kv_head],
                    tokens_per_batch=self.cuda_graph_tokens_per_batch[:bs_now],
                    kv_indptr=kv_indptr[: bs_now + 1],
                    kv_indices=kv_indices,
                    num_heads=self.num_kv_head,
                    kernel_type=self.quest_topk_kernel,
                )
            if num_kv_splits is not None:
                num_kv_splits[:bs_now].fill_(self.max_kv_splits)
            nvtx.range_pop()
        
        # Step 4: Sparse Attention - 对选中 pages 做完整 attention
        # 使用 Quest 专用的 decode kernel（支持 per-head kv_indptr）
        nvtx.range_push("quest_decode_attention_fwd")
        # 调用 Quest decode kernel
        self.quest_decode_attention_fwd(
            q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k_buffer=forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            v_buffer=forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o=o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            kv_indptr=kv_indptr,      # [batch+1, num_heads]
            kv_indices=kv_indices,    # [total_selected_tokens]
            attn_logits=attn_logits,
            attn_lse=attn_lse,
            num_kv_splits=num_kv_splits,
            max_kv_splits=self.max_kv_splits,
            sm_scale=layer.scaling,
            logit_cap=layer.logit_cap,
            layer_id=layer.layer_id,
            force_kv_group_num=1 if use_weak else None,
        )
        nvtx.range_pop()
        
        return o

    # ---------------- CUDA Graph support (decode-only capture) ----------------

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int, kv_indices_buf: Optional[torch.Tensor] = None):
        """Preallocate static buffers used by CUDA Graph replay for decode path."""
        # attn temporaries
        self.cuda_graph_attn_logits = torch.zeros(
            (max_num_tokens, self.num_head, self.max_kv_splits, self.v_head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.cuda_graph_attn_lse = torch.zeros(
            (max_num_tokens, self.num_head, self.max_kv_splits),
            dtype=torch.float32,
            device=self.device,
        )
        self.cuda_graph_num_kv_splits = torch.full(
            (max_num_tokens,), self.max_kv_splits, dtype=torch.int32, device=self.device
        )

        # kv indices capacity: quest_topk pages per head (topk-1 + last)
        tokens_per_head_upper = (self.quest_topk) * self.page_size
        total_capacity = max_num_tokens * self.num_head * tokens_per_head_upper
        if kv_indices_buf is None:
            self.cuda_graph_kv_indices = torch.zeros(
                (total_capacity,), dtype=torch.int32, device=self.device
            )
        else:
            self.cuda_graph_kv_indices = kv_indices_buf

        # csr indptr for batches
        self.kv_indptr = torch.zeros((max_bs + 1,), dtype=torch.int32, device=self.device)

        # Quest-specific scratch for estimate/select/expand without allocations (graph friendly)
        max_pages_cap = (self.max_context_len + self.page_size - 1) // self.page_size
        self.cuda_graph_estimated_scores = torch.zeros(
            (max_bs, self.num_head, max_pages_cap), dtype=torch.float32, device=self.device
        )
        self.cuda_graph_selected_pages = torch.empty(
            (max_bs, self.num_head, self.quest_topk), dtype=torch.int32, device=self.device
        )
        self.cuda_graph_kv_indices_buf = torch.empty(
            (max_bs * self.num_head, tokens_per_head_upper), dtype=torch.int32, device=self.device
        )
        self.cuda_graph_tokens_per_head = torch.zeros(
            (max_bs * self.num_head,), dtype=torch.int32, device=self.device
        )
        self.cuda_graph_tokens_per_batch = torch.zeros(
            (max_bs,), dtype=torch.int32, device=self.device
        )
        self._graph_meta_ready = False

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[None, None]],
    ):
        assert encoder_lens is None, "Not supported"
        assert forward_mode.is_decode(), "Quest Graph capture only supports decode"

        # Fill forward metadata to point to graph buffers (population happens during warmup forward path)
        self.forward_metadata = {
            "kv_indptr": self.kv_indptr[: bs + 1],
            "kv_indices": self.cuda_graph_kv_indices,
            "attn_logits": self.cuda_graph_attn_logits,
            "attn_lse": self.cuda_graph_attn_lse,
            "num_kv_splits": self.cuda_graph_num_kv_splits[:bs],
        }
        # Precompute a tighter pages cap for estimate to reduce overlaunch in graph
        # NOTE: In capture mode, seq_lens might be dummy (e.g. 1), so we cannot rely on it
        # to determine the max_pages_cap for the graph. We must use the global context len
        # to ensure the graph can handle any sequence length during replay.
        self._cg_estimate_pages = (self.max_context_len + self.page_size - 1) // self.page_size
        self._graph_meta_ready = False

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[None, None]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        assert forward_mode.is_decode(), "Quest Graph replay only supports decode"

        # Maintain forward metadata views; population will happen in non-capture forward path if needed
        self.forward_metadata = {
            "kv_indptr": self.kv_indptr[: bs + 1],
            "kv_indices": self.cuda_graph_kv_indices,
            "attn_logits": self.cuda_graph_attn_logits,
            "attn_lse": self.cuda_graph_attn_lse,
            "num_kv_splits": self.cuda_graph_num_kv_splits[:bs],
        }
        if not hasattr(self, "_cg_estimate_pages") or self._cg_estimate_pages is None:
            self._cg_estimate_pages = (self.max_context_len + self.page_size - 1) // self.page_size
        self._graph_meta_ready = False

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    # ---------------- Internal: Page alignment checker ----------------

    def _check_page_alignment(self, forward_batch: ForwardBatch):
        """Assert each request's tokens are page-aligned in KV pool.

        Conditions per batch b:
        - The first token maps to a physical index divisible by page_size.
        - For each logical page [p*P, (p+1)*P), the physical page id (token // P)
          is constant across tokens in that logical page window, except the last
          (possibly partial) page which is handled by the same rule on its span.

        Raises RuntimeError with details on the first violation found.
        """
        req_to_token = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices
        ]
        seq_lens = forward_batch.seq_lens
        if seq_lens.numel() == 0:
            return
        # Small host syncs here are acceptable since this runs only in debug mode
        max_len = int(seq_lens.max().item())
        if max_len <= 0:
            return
        P = int(self.page_size)

        rt = req_to_token[:, :max_len]
        # First-token must start at page boundary
        first_mod = (rt[:, 0] % P).to(torch.int32)
        bad = torch.nonzero(first_mod != 0, as_tuple=False).flatten()
        if bad.numel() > 0:
            raise RuntimeError(
                f"[Quest] Page alignment violated: first token not on boundary for batch indices {bad.tolist()}, page_size={P}"
            )

        page_ids = rt // P
        bs = forward_batch.batch_size
        for b in range(bs):
            L = int(seq_lens[b].item())
            if L == 0:
                continue
            p = 0
            while p * P < L:
                s = p * P
                e = min(s + P, L)
                cur_page_ids = page_ids[b, s:e]
                u = torch.unique(cur_page_ids).cpu()
                if u.numel() > 1:
                    raise RuntimeError(
                        f"[Quest] Page alignment violated: batch {b}, logical_page={p}, physical_pages={u.tolist()}"
                    )
                # Stricter check: within a logical page window, token offsets inside the page
                # must be strictly increasing and contiguous starting from 0.
                # Example (P=4): offsets must be [0,1,2,3] for a full page; for a partial page
                # of length Lp, offsets must be [0,1,...,Lp-1]. Sequences like [0,2,3,1] or
                # [0,2,3] (missing 1) are considered invalid for Quest paged indexing.
                length = e - s
                cur_tokens = rt[b, s:e]
                offs = (cur_tokens % P).to(torch.int32)
                expected = torch.arange(0, length, dtype=torch.int32, device=offs.device)
                if not torch.equal(offs, expected):
                    raise RuntimeError(
                        f"[Quest] In-page order violated: batch {b}, logical_page={p}, "
                        f"offsets={offs.tolist()}, expected={list(range(length))}"
                    )
                p += 1
