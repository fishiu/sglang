"""
Quest Attention Backend - Query-Aware Sparsity for Efficient Long-Context Inference

实现基于 page-level 元数据的三阶段稀疏注意力：
1. Estimate: 用 min/max 元数据快速估算 page 重要性
2. TopK Selection: 选择最重要的 K 个 pages + last page
3. Sparse Attention: 只对选中 pages 做完整注意力

简化版特性：
- 仅支持 batch_size=1
- 仅支持 MHA（num_q_heads == num_kv_heads）
- Decode 阶段使用 Quest，Extend 阶段使用稠密 attention
- 懒初始化：第一次 decode 时从 KV cache 回读计算元数据
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

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
            quest_estimate_scores,
            quest_select_topk_pages,
            quest_update_kv_and_metadata,
            quest_init_all_prefill_pages,
            quest_decode_attention_fwd,
        )
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
        
        self.quest_estimate_scores = quest_estimate_scores
        self.quest_select_topk_pages = quest_select_topk_pages
        self.quest_update_kv_and_metadata = quest_update_kv_and_metadata
        self.quest_init_all_prefill_pages = quest_init_all_prefill_pages
        self.quest_decode_attention_fwd = quest_decode_attention_fwd
        
        # 配置参数
        self.quest_topk = model_runner.server_args.quest_topk
        self.page_size = model_runner.server_args.page_size
        self.num_head = model_runner.model_config.num_attention_heads
        self.head_dim = model_runner.model_config.hidden_size // self.num_head
        
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
        
        # 元数据初始化标记（用于预初始化）
        self.metadata_preinitialized = False
    
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
        Extend 阶段：Delegate 给 TritonAttnBackend
        
        Quest 不在 extend 阶段生效，直接使用标准 Triton extend attention
        """
        return self.triton_backend.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache
        )
    
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
        0. 预初始化：第一次 decode 时初始化所有 prefill pages（仅一次）
        1. 更新 KV cache & 元数据（增量更新）
        2. Estimate: 用元数据估算 page scores
        3. TopK Selection: 选择重要 pages + last page
        4. Sparse Attention: 只对选中 pages 做完整 attention
        """
        # Reshape q（处理 torch.compile 的 3D 输出问题）
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        
        # 分配输出 buffer
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)
        
        # Step 0: 预初始化（仅第一次 decode，第一层时执行）
        if not self.metadata_preinitialized and layer.layer_id == 0:
            # 初始化所有 prefill pages
            batch_idx = 0  # 简化版仅支持 batch_size=1
            seq_len = forward_batch.seq_lens[batch_idx].item() - 1  # 不包括当前 token
            
            if seq_len > 0:  # 有 prefill tokens
                for layer_idx in range(forward_batch.token_to_kv_pool.layer_num):
                    self.quest_init_all_prefill_pages(
                        k_buffer=forward_batch.token_to_kv_pool.get_key_buffer(layer_idx),
                        k_metadata=forward_batch.token_to_kv_pool.get_metadata_buffer(layer_idx),
                        metadata_init=forward_batch.token_to_kv_pool.get_metadata_init_flag(layer_idx),
                        seq_len=seq_len,
                        page_size=self.page_size,
                    )
            
            self.metadata_preinitialized = True
        
        # Step 1: 更新 KV cache 和元数据
        if save_kv_cache:
            self.quest_update_kv_and_metadata(
                k_new=k.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                v_new=v.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                k_buffer=forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                v_buffer=forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                k_metadata=forward_batch.token_to_kv_pool.get_metadata_buffer(layer.layer_id),
                metadata_init=forward_batch.token_to_kv_pool.get_metadata_init_flag(layer.layer_id),
                out_cache_loc=forward_batch.out_cache_loc,
                seq_lens=forward_batch.seq_lens,
                page_size=self.page_size,
            )
        
        # Step 2: Estimate - 估算 page-level 注意力得分
        self.quest_estimate_scores(
            q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k_metadata=forward_batch.token_to_kv_pool.get_metadata_buffer(layer.layer_id),
            seq_lens=forward_batch.seq_lens,
            estimated_scores=self.estimated_scores,
            page_size=self.page_size,
        )
        
        # Step 3: TopK Selection - 选择重要 pages
        kv_indptr, kv_indices = self.quest_select_topk_pages(
            estimated_scores=self.estimated_scores,
            seq_lens=forward_batch.seq_lens,
            quest_topk=self.quest_topk,
            page_size=self.page_size,
        )
        
        # Step 4: Sparse Attention - 对选中 pages 做完整 attention
        # 使用 Quest 专用的 decode kernel（支持 per-head kv_indptr）
        
        # 准备 num_kv_splits（简化版：每个 batch 用相同的 split 数）
        batch_size = self.forward_metadata["batch_size"]
        num_kv_splits = torch.full(
            (batch_size,), self.max_kv_splits, dtype=torch.int32, device="cuda"
        )
        
        # 调用 Quest decode kernel
        self.quest_decode_attention_fwd(
            q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k_buffer=forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            v_buffer=forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o=o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            kv_indptr=kv_indptr,      # [batch+1, num_heads]
            kv_indices=kv_indices,    # [total_selected_tokens]
            attn_logits=self.attn_logits,
            attn_lse=self.attn_lse,
            num_kv_splits=num_kv_splits,
            max_kv_splits=self.max_kv_splits,
            sm_scale=layer.scaling,
            logit_cap=layer.logit_cap,
        )
        
        return o

