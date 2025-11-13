# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for decoding.
It supports page size = 1.

这个文件实现了用于 decoding 阶段的内存高效的 attention 计算，使用 Triton 编写 GPU kernel。
主要特点：
1. 支持 page size = 1 的内存管理
2. 实现了两阶段的 attention 计算：stage1 计算部分结果，stage2 合并结果
3. 支持 MHA (Multi-Head Attention)、GQA (Grouped Query Attention) 和 MQA (Multi-Query Attention)
4. 针对 decoding 阶段优化，处理单个 token 的生成

在 decoding 阶段，模型每次只生成一个 token，query 只有一个位置，但需要与所有历史的 key-value pairs 计算 attention。
这个实现通过分块处理和两阶段计算来优化内存使用和计算效率。
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

import logging

# Triton 是一个用于编写高效 GPU kernel 的 Python 库
import triton
import triton.language as tl  # Triton 的语言扩展，提供了类似 CUDA 的语法
import torch.cuda.nvtx as nvtx

# 检测是否运行在 AMD GPU (HIP) 上，因为需要针对不同硬件做优化
from sglang.srt.utils import is_hip

_is_hip = is_hip()  # 全局变量，标识是否为 AMD GPU

logger = logging.getLogger(__name__)

# KV cache 的最小块大小，用于内存对齐和优化
_MIN_BLOCK_KV = 32


@triton.jit  # Triton JIT 装饰器，将 Python 函数编译为 GPU kernel
def tanh(x):
    """
    实现双曲正切函数 tanh(x)
    
    这是一个自定义的 tanh 实现，因为 Triton 可能没有内置的 tanh 函数。
    使用 sigmoid 函数来实现：tanh(x) = 2 * sigmoid(2x) - 1
    
    在 attention 计算中，当使用 logit_cap 参数时会用到这个函数，
    用于限制 attention scores 的范围，防止数值溢出。
    
    Args:
        x: 输入张量
    
    Returns:
        tanh(x) 的结果
    """
    # Tanh is just a scaled sigmoid
    # 利用 sigmoid 和 tanh 的数学关系：tanh(x) = 2 * sigmoid(2x) - 1
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_stage1(
    # 输入张量（注意：在 Triton kernel 中，张量通过指针访问）
    Q,                  # Query 张量: [batch_size, num_q_heads, head_dim] 
    K_Buffer,           # Key buffer: [total_tokens, num_kv_heads, head_dim] - 存储历史 keys
    V_Buffer,           # Value buffer: [total_tokens, num_kv_heads, v_head_dim] - 存储历史 values
    sm_scale,           # Softmax 缩放因子: float - 通常是 1/sqrt(head_dim)
    # 索引结构（用于变长序列支持）
    kv_indptr,          # KV 指针: [batch_size + 1] - CSR格式，每个序列的起始/结束位置
    kv_indices,         # KV 索引: [total_seq_len] - 每个位置在 buffer 中的实际索引
    # 输出张量
    Att_Out,            # 中间输出: [batch_size, num_q_heads, max_kv_splits, v_head_dim] - Stage1 的部分结果
    Att_Lse,            # LSE 值: [batch_size, num_q_heads, max_kv_splits] - 每个split的log-sum-exp
    num_kv_splits,      # 分块数量: [batch_size] - 每个batch实际使用的split数  7
    # DEBUG: 原子计数器
    loop_counter,       # 循环计数器: [1] - 统计 for 循环总执行次数
    # DEBUG: 每个grid的详细信息
    grid_info,          # 网格信息: [batch, head, split, 4] - 存储每个grid的循环参数
    
    # Stride 参数（用于正确计算多维张量的内存偏移）
    # =====================================================
    # Stride 解释：stride[i] 表示在第i维移动1个位置时内存地址的偏移量
    # 例如：对于 [B, H, D] 张量，stride = [H*D, D, 1]
    stride_qbs,         # Query batch stride: int - Q[b+1] - Q[b] 的内存偏移
    stride_qh,          # Query head stride: int - Q[:, h+1] - Q[:, h] 的内存偏移  
    stride_buf_kbs,     # K buffer batch stride: int - K[token+1] - K[token] 的偏移
    stride_buf_kh,      # K buffer head stride: int - K[:, h+1] - K[:, h] 的偏移
    stride_buf_vbs,     # V buffer batch stride: int - V[token+1] - V[token] 的偏移
    stride_buf_vh,      # V buffer head stride: int - V[:, h+1] - V[:, h] 的偏移
    stride_mid_ob,      # 中间输出 batch stride: int - Out[b+1] - Out[b] 的偏移
    stride_mid_oh,      # 中间输出 head stride: int - Out[:, h+1] - Out[:, h] 的偏移
    stride_mid_os,      # 中间输出 split stride: int - Out[:, :, s+1] - Out[:, :, s] 的偏移
    
    # 编译时常量参数（在 Triton 编译时确定，运行时不可变）
    # ========================================================
    kv_group_num: tl.constexpr,    # KV 分组数: int - num_q_heads // num_kv_heads
    BLOCK_DMODEL: tl.constexpr,    # Key 维度块大小: int - 必须 >= head_dim 且为2的幂  128
    BLOCK_DV: tl.constexpr,        # Value 维度块大小: int - 必须 >= v_head_dim 且为2的幂  128
    BLOCK_N: tl.constexpr,         # 序列块大小: int - 每次处理的 KV pairs 数量 64
    MIN_BLOCK_KV: tl.constexpr,    # 最小KV块: int - 用于对齐，通常是32  32
    logit_cap: tl.constexpr,       # Logit 上限: float - 0表示不启用
    Lk: tl.constexpr,              # Key 实际维度: int - 真实的 head_dim 值  128
    Lv: tl.constexpr,              # Value 实际维度: int - 真实的 v_head_dim 值  128
):
    """
    Attention 计算的第一阶段 kernel (Stage 1)
    
    这是 Flash Attention decoding 的核心 kernel，负责计算 attention 的第一阶段。
    在这个阶段，每个 GPU 线程块处理一部分 KV pairs，计算部分的 attention 结果。
    
    主要工作流程：
    1. 获取当前线程块负责的 batch、head 和 KV split
    2. 加载对应的 query vector
    3. 分块遍历分配给当前线程块的 KV pairs
    4. 对每个 KV 块计算 QK^T，应用 softmax scaling 和 logit capping
    5. 使用 online softmax 技术累积 attention 结果
    6. 存储部分结果和 LSE 值，供 stage2 使用
    
    这个函数被 _decode_att_m_fwd 调用，处理标准的 MHA 情况。
    """
    # 获取当前线程块的三维索引：(batch, head, kv_split)
    # 这是 Triton 的并行执行模型，每个线程块处理一个特定的 (batch, head, split) 组合
    cur_batch = tl.program_id(0)      # 当前处理的 batch 索引
    cur_head = tl.program_id(1)       # 当前处理的 attention head 索引
    split_kv_id = tl.program_id(2)    # 当前处理的 KV split 索引

    # 计算对应的 KV head 索引，用于 GQA/MQA 情况
    # 在 GQA 中，多个 query heads 共享同一个 key-value head
    cur_kv_head = cur_head // kv_group_num

    # 创建维度索引向量，用于访问张量的不同维度
    # =================================================
    # Triton 分块访问详解：
    # - offs_d: [BLOCK_DMODEL] 形状的向量，包含 [0, 1, 2, ..., BLOCK_DMODEL-1]
    # - 用于访问 key/query 的各个维度分量，实现向量化计算
    # - BLOCK_DMODEL 必须是2的幂且 >= 实际的 head_dim，多余部分会被掩码过滤
    offs_d = tl.arange(0, BLOCK_DMODEL)    # Key 维度索引: [BLOCK_DMODEL] 
    offs_dv = tl.arange(0, BLOCK_DV)       # Value 维度索引: [BLOCK_DV]
    
    # 创建掩码，处理维度不是块大小整数倍的情况
    # mask_d[i] = True 表示 offs_d[i] 对应的维度是有效的
    mask_d = offs_d < Lk                   # Key 维度有效掩码: [BLOCK_DMODEL] bool
    mask_dv = offs_dv < Lv                 # Value 维度有效掩码: [BLOCK_DV] bool

    # 从 KV 索引指针中获取当前 batch 的 KV 范围信息
    # kv_indptr 是一个累积索引数组，类似于 CSR 稀疏矩阵的 indptr
    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)          # 当前 batch 的 KV 起始索引  0 | 0
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx  # 当前 batch 的序列长度  260 | 513
    kv_splits = tl.load(num_kv_splits + cur_batch)                   # 当前 batch 的 KV 分块数量  7 | 8

    # 计算 query 张量的内存偏移量
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    # 计算每个 split 应该处理的 KV 长度
    # 首先将序列长度按 split 数量分割，然后向上取整到 MIN_BLOCK_KV 的倍数
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )  ## cdiv(cdiv(260, 7), 32) * 32 = cdiv(38, 32) * 32 = 64, 每个 split 本来只需要读 38，但是为了对齐，需要读 64 | 96
    # 计算当前 split 负责的 KV 范围
    split_kv_start = kv_len_per_split * split_kv_id                    # 当前 split 的起始位置  0
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)  # 当前 split 的结束位置  64    

    # DEBUG: 存储当前grid的循环信息（如果grid_info不为空）
    if grid_info is not None:
        # 计算实际的循环次数
        actual_loop_count = 0
        if split_kv_end > split_kv_start:
            actual_loop_count = (split_kv_end - split_kv_start + BLOCK_N - 1) // BLOCK_N
        
        # 存储到grid_info: [batch, head, split, 4]
        # 4个值分别是：[split_kv_start, split_kv_end, BLOCK_N, actual_loop_count]
        grid_offset = (cur_batch * tl.num_programs(1) * tl.num_programs(2) + 
                      cur_head * tl.num_programs(2) + split_kv_id) * 4
        tl.store(grid_info + grid_offset, split_kv_start)
        tl.store(grid_info + grid_offset + 1, split_kv_end)
        tl.store(grid_info + grid_offset + 2, BLOCK_N)
        tl.store(grid_info + grid_offset + 3, actual_loop_count)

    # 初始化 online softmax 算法的状态变量
    # Online softmax 是一种数值稳定的 softmax 计算方法，避免存储所有中间结果
    e_max = -float("inf")                  # 当前的最大 logit 值
    e_sum = 0.0                           # 当前的指数和
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)  # 累积的 attention 输出

    # 只有当当前 split 有有效的 KV pairs 时才进行计算
    if split_kv_end > split_kv_start:
        # 加载当前 head 的 query vector
        q = tl.load(Q + off_q, mask=mask_d, other=0.0)
        
        # 分块处理核心循环：遍历当前 split 负责的所有 KV pairs
        # =========================================================
        # 分块策略详解：
        # 1. 将长序列分成 BLOCK_N 大小的块（例如 64 个 token 一块）
        # 2. 每次循环处理一个块，避免一次性加载整个序列到内存
        # 3. 对于序列长度1000，BLOCK_N=64，会循环 ⌈1000/64⌉=16 次
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):  # (0, 64, 64)
            # DEBUG: 原子递增循环计数器
            tl.atomic_add(loop_counter, 1)
            
            # 计算当前块中每个位置的序列索引
            # offs_n: [BLOCK_N] - 当前块内的绝对位置索引
            # 例如：第2块时 offs_n = [64, 65, 66, ..., 127]
            offs_n = start_n + tl.arange(0, BLOCK_N)
            
            # 从 kv_indices 中获取实际的 KV 存储位置
            # ==========================================
            # kv_indices 解决了变长序列的索引问题：
            # - 序列位置 i 的实际 token 存储在 buffer[kv_indices[i]]
            # - 支持序列重排、padding 等复杂内存布局
            # kv_loc: [BLOCK_N] - 当前块中每个位置对应的 buffer 索引
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,  # 避免越界访问
                other=0,  # 越界位置填充0
            )

            # 计算 Key buffer 中的内存偏移量
            # =======================================
            # Triton 张量索引详解：
            # - kv_loc[:, None]: [BLOCK_N, 1] - 将 token 索引扩展为列向量
            # - offs_d[None, :]: [1, BLOCK_DMODEL] - 将维度索引扩展为行向量
            # - 广播相加得到: [BLOCK_N, BLOCK_DMODEL] - 每个(token, dim)的内存偏移
            offs_buf_k = (
                kv_loc[:, None] * stride_buf_kbs      # [BLOCK_N, 1] token 位置偏移
                + cur_kv_head * stride_buf_kh         # 标量 - KV head 偏移
                + offs_d[None, :]                     # [1, BLOCK_DMODEL] 维度偏移
            )  # 结果: [BLOCK_N, BLOCK_DMODEL] 内存偏移矩阵
            
            # tl.static_print("(", cur_batch, cur_head, split_kv_id, "), offs_buf_k: ", offs_buf_k.shape)

            # 从 Key buffer 加载当前块的 key vectors
            # k: [BLOCK_N, BLOCK_DMODEL] - 当前块的所有 key vectors
            # 每一行 k[i, :] 是第 i 个 token 的 key vector
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0.0,
            )
            
            # 计算 Query-Key 点积：实现 QK^T 操作
            # ====================================
            # 矩阵运算形状变化：
            # - q: [BLOCK_DMODEL] -> q[None, :]: [1, BLOCK_DMODEL] 
            # - k: [BLOCK_N, BLOCK_DMODEL]
            # - q[None, :] * k: [1, BLOCK_DMODEL] * [BLOCK_N, BLOCK_DMODEL] = [BLOCK_N, BLOCK_DMODEL] (广播)
            # - tl.sum(..., 1): 沿维度1求和 -> [BLOCK_N]
            # 结果：qk[i] = query · key[i] （每个 token 的 attention score）
            qk = tl.sum(q[None, :] * k, 1)  # [BLOCK_N] attention scores
            # 应用 attention 缩放因子，通常是 1/sqrt(head_dim)
            qk *= sm_scale

            # 如果启用了 logit capping，则应用 tanh 函数限制 attention scores 的范围
            # 这有助于防止 attention scores 过大导致的数值不稳定
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            # 对超出有效范围的位置设置为负无穷，确保 softmax 后为 0
            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            # 计算 Value buffer 中的内存偏移量
            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs      # KV 位置的 batch stride
                + cur_kv_head * stride_buf_vh         # KV head 的 stride  
                + offs_dv[None, :]                    # Value 维度索引
            )
            # 从 Value buffer 加载当前块的 value vectors
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            # Online Softmax 算法的核心部分
            # =====================================
            # Online Softmax 解决了长序列 softmax 的内存问题：
            # 传统方法：需要存储所有 attention scores，然后计算 softmax
            # Online 方法：逐块处理，动态维护 max 和 sum，最终得到相同结果
            
            # 更新全局最大值，用于数值稳定的 softmax 计算
            # tl.max(qk, 0): 标量 - 当前块的最大 attention score
            # e_max: 标量 - 之前所有块的最大值
            # n_e_max: 标量 - 更新后的全局最大值
            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            
            # 计算重缩放因子，用于调整之前累积的结果
            # 当发现更大的 score 时，需要重新缩放之前的累积结果
            re_scale = tl.exp(e_max - n_e_max)  # 标量 - 重缩放因子
            
            # 计算当前块的 softmax 权重
            # p: [BLOCK_N] - 当前块的归一化权重
            p = tl.exp(qk - n_e_max)
            
            # 重缩放之前累积的结果
            # acc: [BLOCK_DV] - 累积的 attention 输出
            acc *= re_scale  # [BLOCK_DV] 重缩放
            
            # 累加当前块的加权 value：实现 sum(p_i * v_i)
            # ============================================
            # 张量运算形状变化：
            # - p: [BLOCK_N] -> p[:, None]: [BLOCK_N, 1]
            # - v: [BLOCK_N, BLOCK_DV] 
            # - p[:, None] * v: [BLOCK_N, 1] * [BLOCK_N, BLOCK_DV] = [BLOCK_N, BLOCK_DV] (广播)
            # - tl.sum(..., 0): 沿维度0求和 -> [BLOCK_DV]
            acc += tl.sum(p[:, None] * v, 0)  # [BLOCK_DV] 累加加权 values

            # 更新指数和与最大值，为下一轮迭代做准备
            e_sum = e_sum * re_scale + tl.sum(p, 0)  # 标量 - 更新指数和
            e_max = n_e_max  # 标量 - 更新最大值

        # 计算中间输出张量的内存偏移量
        # 这是存储当前 split 的部分 attention 结果的位置
        offs_mid_o = (
            cur_batch * stride_mid_ob     # batch 维度偏移
            + cur_head * stride_mid_oh    # head 维度偏移
            + split_kv_id * stride_mid_os # split 维度偏移
            + offs_dv                     # value 维度偏移
        )

        # 存储归一化的部分 attention 结果
        # acc / e_sum 是当前 split 的 softmax 归一化结果
        # 这个结果将在 stage2 中与其他 split 的结果合并
        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=(mask_dv),
        )

        # 计算 LSE (Log-Sum-Exp) 值的存储位置
        # LSE 值用于 stage2 中正确合并多个 split 的结果
        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        # 存储 LSE 值：log(sum(exp(x_i))) = max + log(sum(exp(x_i - max)))
        # 这是 online softmax 算法中用于后续合并的关键信息
        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


@triton.jit
def _fwd_kernel_stage1_layer0(
    # 输入张量（注意：在 Triton kernel 中，张量通过指针访问）
    Q,                  # Query 张量: [batch_size, num_q_heads, head_dim] 
    K_Buffer,           # Key buffer: [total_tokens, num_kv_heads, head_dim] - 存储历史 keys
    V_Buffer,           # Value buffer: [total_tokens, num_kv_heads, v_head_dim] - 存储历史 values
    sm_scale,           # Softmax 缩放因子: float - 通常是 1/sqrt(head_dim)
    # 索引结构（用于变长序列支持）
    kv_indptr,          # KV 指针: [batch_size + 1] - CSR格式，每个序列的起始/结束位置
    kv_indices,         # KV 索引: [total_seq_len] - 每个位置在 buffer 中的实际索引
    # 输出张量
    Att_Out,            # 中间输出: [batch_size, num_q_heads, max_kv_splits, v_head_dim] - Stage1 的部分结果
    Att_Lse,            # LSE 值: [batch_size, num_q_heads, max_kv_splits] - 每个split的log-sum-exp
    num_kv_splits,      # 分块数量: [batch_size] - 每个batch实际使用的split数  7
    # DEBUG: 原子计数器
    loop_counter,       # 循环计数器: [1] - 统计 for 循环总执行次数
    # DEBUG: 每个grid的详细信息
    grid_info,          # 网格信息: [batch, head, split, 4] - 存储每个grid的循环参数
    
    # Stride 参数（用于正确计算多维张量的内存偏移）
    # =====================================================
    # Stride 解释：stride[i] 表示在第i维移动1个位置时内存地址的偏移量
    # 例如：对于 [B, H, D] 张量，stride = [H*D, D, 1]
    stride_qbs,         # Query batch stride: int - Q[b+1] - Q[b] 的内存偏移
    stride_qh,          # Query head stride: int - Q[:, h+1] - Q[:, h] 的内存偏移  
    stride_buf_kbs,     # K buffer batch stride: int - K[token+1] - K[token] 的偏移
    stride_buf_kh,      # K buffer head stride: int - K[:, h+1] - K[:, h] 的偏移
    stride_buf_vbs,     # V buffer batch stride: int - V[token+1] - V[token] 的偏移
    stride_buf_vh,      # V buffer head stride: int - V[:, h+1] - V[:, h] 的偏移
    stride_mid_ob,      # 中间输出 batch stride: int - Out[b+1] - Out[b] 的偏移
    stride_mid_oh,      # 中间输出 head stride: int - Out[:, h+1] - Out[:, h] 的偏移
    stride_mid_os,      # 中间输出 split stride: int - Out[:, :, s+1] - Out[:, :, s] 的偏移
    
    # 编译时常量参数（在 Triton 编译时确定，运行时不可变）
    # ========================================================
    kv_group_num: tl.constexpr,    # KV 分组数: int - num_q_heads // num_kv_heads
    BLOCK_DMODEL: tl.constexpr,    # Key 维度块大小: int - 必须 >= head_dim 且为2的幂  128
    BLOCK_DV: tl.constexpr,        # Value 维度块大小: int - 必须 >= v_head_dim 且为2的幂  128
    BLOCK_N: tl.constexpr,         # 序列块大小: int - 每次处理的 KV pairs 数量 64
    MIN_BLOCK_KV: tl.constexpr,    # 最小KV块: int - 用于对齐，通常是32  32
    logit_cap: tl.constexpr,       # Logit 上限: float - 0表示不启用
    Lk: tl.constexpr,              # Key 实际维度: int - 真实的 head_dim 值  128
    Lv: tl.constexpr,              # Value 实际维度: int - 真实的 v_head_dim 值  128
):
    """
    Attention 计算的第一阶段 kernel (Stage 1)
    
    这是 Flash Attention decoding 的核心 kernel，负责计算 attention 的第一阶段。
    在这个阶段，每个 GPU 线程块处理一部分 KV pairs，计算部分的 attention 结果。
    
    主要工作流程：
    1. 获取当前线程块负责的 batch、head 和 KV split
    2. 加载对应的 query vector
    3. 分块遍历分配给当前线程块的 KV pairs
    4. 对每个 KV 块计算 QK^T，应用 softmax scaling 和 logit capping
    5. 使用 online softmax 技术累积 attention 结果
    6. 存储部分结果和 LSE 值，供 stage2 使用
    
    这个函数被 _decode_att_m_fwd 调用，处理标准的 MHA 情况。
    """
    # 获取当前线程块的三维索引：(batch, head, kv_split)
    # 这是 Triton 的并行执行模型，每个线程块处理一个特定的 (batch, head, split) 组合
    cur_batch = tl.program_id(0)      # 当前处理的 batch 索引
    cur_head = tl.program_id(1)       # 当前处理的 attention head 索引
    split_kv_id = tl.program_id(2)    # 当前处理的 KV split 索引

    # 计算对应的 KV head 索引，用于 GQA/MQA 情况
    # 在 GQA 中，多个 query heads 共享同一个 key-value head
    cur_kv_head = cur_head // kv_group_num

    # 创建维度索引向量，用于访问张量的不同维度
    # =================================================
    # Triton 分块访问详解：
    # - offs_d: [BLOCK_DMODEL] 形状的向量，包含 [0, 1, 2, ..., BLOCK_DMODEL-1]
    # - 用于访问 key/query 的各个维度分量，实现向量化计算
    # - BLOCK_DMODEL 必须是2的幂且 >= 实际的 head_dim，多余部分会被掩码过滤
    offs_d = tl.arange(0, BLOCK_DMODEL)    # Key 维度索引: [BLOCK_DMODEL] 
    offs_dv = tl.arange(0, BLOCK_DV)       # Value 维度索引: [BLOCK_DV]
    
    # 创建掩码，处理维度不是块大小整数倍的情况
    # mask_d[i] = True 表示 offs_d[i] 对应的维度是有效的
    mask_d = offs_d < Lk                   # Key 维度有效掩码: [BLOCK_DMODEL] bool
    mask_dv = offs_dv < Lv                 # Value 维度有效掩码: [BLOCK_DV] bool

    # 从 KV 索引指针中获取当前 batch 的 KV 范围信息
    # kv_indptr 是一个累积索引数组，类似于 CSR 稀疏矩阵的 indptr
    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)          # 当前 batch 的 KV 起始索引  0 | 0
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx  # 当前 batch 的序列长度  260 | 513
    kv_splits = tl.load(num_kv_splits + cur_batch)                   # 当前 batch 的 KV 分块数量  7 | 8

    # 计算 query 张量的内存偏移量
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    # 计算每个 split 应该处理的 KV 长度
    # 首先将序列长度按 split 数量分割，然后向上取整到 MIN_BLOCK_KV 的倍数
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )  ## cdiv(cdiv(260, 7), 32) * 32 = cdiv(38, 32) * 32 = 64, 每个 split 本来只需要读 38，但是为了对齐，需要读 64 | 96
    # 计算当前 split 负责的 KV 范围
    split_kv_start = kv_len_per_split * split_kv_id                    # 当前 split 的起始位置  0
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)  # 当前 split 的结束位置  64    

    # DEBUG: 存储当前grid的循环信息（如果grid_info不为空）
    if grid_info is not None:
        # 计算实际的循环次数
        actual_loop_count = 0
        if split_kv_end > split_kv_start:
            actual_loop_count = (split_kv_end - split_kv_start + BLOCK_N - 1) // BLOCK_N
        
        # 存储到grid_info: [batch, head, split, 4]
        # 4个值分别是：[split_kv_start, split_kv_end, BLOCK_N, actual_loop_count]
        grid_offset = (cur_batch * tl.num_programs(1) * tl.num_programs(2) + 
                      cur_head * tl.num_programs(2) + split_kv_id) * 4
        tl.store(grid_info + grid_offset, split_kv_start)
        tl.store(grid_info + grid_offset + 1, split_kv_end)
        tl.store(grid_info + grid_offset + 2, BLOCK_N)
        tl.store(grid_info + grid_offset + 3, actual_loop_count)

    # 初始化 online softmax 算法的状态变量
    # Online softmax 是一种数值稳定的 softmax 计算方法，避免存储所有中间结果
    e_max = -float("inf")                  # 当前的最大 logit 值
    e_sum = 0.0                           # 当前的指数和
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)  # 累积的 attention 输出

    # 只有当当前 split 有有效的 KV pairs 时才进行计算
    if split_kv_end > split_kv_start:
        # 加载当前 head 的 query vector
        q = tl.load(Q + off_q, mask=mask_d, other=0.0)
        
        # 分块处理核心循环：遍历当前 split 负责的所有 KV pairs
        # =========================================================
        # 分块策略详解：
        # 1. 将长序列分成 BLOCK_N 大小的块（例如 64 个 token 一块）
        # 2. 每次循环处理一个块，避免一次性加载整个序列到内存
        # 3. 对于序列长度1000，BLOCK_N=64，会循环 ⌈1000/64⌉=16 次
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):  # (0, 64, 64)
            # DEBUG: 原子递增循环计数器
            tl.atomic_add(loop_counter, 1)
            
            # 计算当前块中每个位置的序列索引
            # offs_n: [BLOCK_N] - 当前块内的绝对位置索引
            # 例如：第2块时 offs_n = [64, 65, 66, ..., 127]
            offs_n = start_n + tl.arange(0, BLOCK_N)
            
            # 从 kv_indices 中获取实际的 KV 存储位置
            # ==========================================
            # kv_indices 解决了变长序列的索引问题：
            # - 序列位置 i 的实际 token 存储在 buffer[kv_indices[i]]
            # - 支持序列重排、padding 等复杂内存布局
            # kv_loc: [BLOCK_N] - 当前块中每个位置对应的 buffer 索引
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,  # 避免越界访问
                other=0,  # 越界位置填充0
            )

            # 计算 Key buffer 中的内存偏移量
            # =======================================
            # Triton 张量索引详解：
            # - kv_loc[:, None]: [BLOCK_N, 1] - 将 token 索引扩展为列向量
            # - offs_d[None, :]: [1, BLOCK_DMODEL] - 将维度索引扩展为行向量
            # - 广播相加得到: [BLOCK_N, BLOCK_DMODEL] - 每个(token, dim)的内存偏移
            offs_buf_k = (
                kv_loc[:, None] * stride_buf_kbs      # [BLOCK_N, 1] token 位置偏移
                + cur_kv_head * stride_buf_kh         # 标量 - KV head 偏移
                + offs_d[None, :]                     # [1, BLOCK_DMODEL] 维度偏移
            )  # 结果: [BLOCK_N, BLOCK_DMODEL] 内存偏移矩阵
            
            # tl.static_print("(", cur_batch, cur_head, split_kv_id, "), offs_buf_k: ", offs_buf_k.shape)

            # 从 Key buffer 加载当前块的 key vectors
            # k: [BLOCK_N, BLOCK_DMODEL] - 当前块的所有 key vectors
            # 每一行 k[i, :] 是第 i 个 token 的 key vector
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0.0,
            )
            
            # 计算 Query-Key 点积：实现 QK^T 操作
            # ====================================
            # 矩阵运算形状变化：
            # - q: [BLOCK_DMODEL] -> q[None, :]: [1, BLOCK_DMODEL] 
            # - k: [BLOCK_N, BLOCK_DMODEL]
            # - q[None, :] * k: [1, BLOCK_DMODEL] * [BLOCK_N, BLOCK_DMODEL] = [BLOCK_N, BLOCK_DMODEL] (广播)
            # - tl.sum(..., 1): 沿维度1求和 -> [BLOCK_N]
            # 结果：qk[i] = query · key[i] （每个 token 的 attention score）
            qk = tl.sum(q[None, :] * k, 1)  # [BLOCK_N] attention scores
            # 应用 attention 缩放因子，通常是 1/sqrt(head_dim)
            qk *= sm_scale

            # 如果启用了 logit capping，则应用 tanh 函数限制 attention scores 的范围
            # 这有助于防止 attention scores 过大导致的数值不稳定
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            # 对超出有效范围的位置设置为负无穷，确保 softmax 后为 0
            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            # 计算 Value buffer 中的内存偏移量
            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs      # KV 位置的 batch stride
                + cur_kv_head * stride_buf_vh         # KV head 的 stride  
                + offs_dv[None, :]                    # Value 维度索引
            )
            # 从 Value buffer 加载当前块的 value vectors
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            # Online Softmax 算法的核心部分
            # =====================================
            # Online Softmax 解决了长序列 softmax 的内存问题：
            # 传统方法：需要存储所有 attention scores，然后计算 softmax
            # Online 方法：逐块处理，动态维护 max 和 sum，最终得到相同结果
            
            # 更新全局最大值，用于数值稳定的 softmax 计算
            # tl.max(qk, 0): 标量 - 当前块的最大 attention score
            # e_max: 标量 - 之前所有块的最大值
            # n_e_max: 标量 - 更新后的全局最大值
            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            
            # 计算重缩放因子，用于调整之前累积的结果
            # 当发现更大的 score 时，需要重新缩放之前的累积结果
            re_scale = tl.exp(e_max - n_e_max)  # 标量 - 重缩放因子
            
            # 计算当前块的 softmax 权重
            # p: [BLOCK_N] - 当前块的归一化权重
            p = tl.exp(qk - n_e_max)
            
            # 重缩放之前累积的结果
            # acc: [BLOCK_DV] - 累积的 attention 输出
            acc *= re_scale  # [BLOCK_DV] 重缩放
            
            # 累加当前块的加权 value：实现 sum(p_i * v_i)
            # ============================================
            # 张量运算形状变化：
            # - p: [BLOCK_N] -> p[:, None]: [BLOCK_N, 1]
            # - v: [BLOCK_N, BLOCK_DV] 
            # - p[:, None] * v: [BLOCK_N, 1] * [BLOCK_N, BLOCK_DV] = [BLOCK_N, BLOCK_DV] (广播)
            # - tl.sum(..., 0): 沿维度0求和 -> [BLOCK_DV]
            acc += tl.sum(p[:, None] * v, 0)  # [BLOCK_DV] 累加加权 values

            # 更新指数和与最大值，为下一轮迭代做准备
            e_sum = e_sum * re_scale + tl.sum(p, 0)  # 标量 - 更新指数和
            e_max = n_e_max  # 标量 - 更新最大值

        # 计算中间输出张量的内存偏移量
        # 这是存储当前 split 的部分 attention 结果的位置
        offs_mid_o = (
            cur_batch * stride_mid_ob     # batch 维度偏移
            + cur_head * stride_mid_oh    # head 维度偏移
            + split_kv_id * stride_mid_os # split 维度偏移
            + offs_dv                     # value 维度偏移
        )

        # 存储归一化的部分 attention 结果
        # acc / e_sum 是当前 split 的 softmax 归一化结果
        # 这个结果将在 stage2 中与其他 split 的结果合并
        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=(mask_dv),
        )

        # 计算 LSE (Log-Sum-Exp) 值的存储位置
        # LSE 值用于 stage2 中正确合并多个 split 的结果
        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        # 存储 LSE 值：log(sum(exp(x_i))) = max + log(sum(exp(x_i - max)))
        # 这是 online softmax 算法中用于后续合并的关键信息
        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


def _decode_att_m_fwd(
    q,              # Query 张量: [batch_size, num_q_heads, head_dim]
    k_buffer,       # Key buffer: [total_tokens, num_kv_heads, head_dim] 
    v_buffer,       # Value buffer: [total_tokens, num_kv_heads, v_head_dim]
    att_out,        # 中间输出: [batch_size, num_q_heads, max_kv_splits, v_head_dim]
    att_lse,        # LSE 张量: [batch_size, num_q_heads, max_kv_splits]
    kv_indptr,      # KV 指针: [batch_size + 1]
    kv_indices,     # KV 索引: [total_seq_len]
    num_kv_splits,  # 分块数量: [batch_size]
    max_kv_splits,  # 最大分块数: int
    sm_scale,       # 缩放因子: float
    logit_cap,      # Logit上限: float
    layer_id=None,  # Layer ID for debugging: int
):
    """
    Multi-Head Attention 的 decoding 前向传播函数 (Stage 1)
    
    这是处理标准 MHA 的主要函数，负责启动 stage1 kernel 的执行。
    它配置 Triton kernel 的执行参数，包括网格大小、块大小、线程数等。
    
    主要功能：
    1. 根据输入张量的形状确定执行配置
    2. 处理不同硬件平台 (NVIDIA/AMD) 的优化参数
    3. 启动 _fwd_kernel_stage1 kernel 进行实际计算
    
    这个函数被 decode_attention_fwd_normal 调用，专门处理 kv_group_num == 1 的情况。
    """
    # BLOCK 详解：这是 Triton 分块计算的核心参数
    # ========================================
    # BLOCK 是什么？
    # - BLOCK 指定了在序列长度维度上的分块大小，即每次处理多少个 KV pairs
    # - 这是 Flash Attention 的核心思想：将大的 attention 矩阵分割成小块逐个处理
    # 
    # 为什么需要分块？
    # 1. 内存限制：完整的 attention 矩阵 [seq_len, seq_len] 对长序列会消耗巨大内存
    # 2. 计算效率：GPU 的 shared memory 和寄存器有限，分块能最大化利用
    # 3. 数值稳定性：online softmax 算法需要逐块处理
    #
    # BLOCK=64 意味着什么？
    # - 每个 Triton 线程块一次处理 64 个连续的 KV pairs
    # - 如果序列长度是 1000，会分成 ⌈1000/64⌉ = 16 个块来处理
    # - 每个块计算的是 query [1, head_dim] 与 keys [64, head_dim] 的点积
    BLOCK = 64  # 序列维度的分块大小：每次处理 64 个 KV pairs
    
    # 针对 AMD MI3xx GPU 的 SGPR (Scalar General Purpose Register) 限制进行优化
    # AMD GPU 的寄存器限制更严格，需要使用更小的块大小
    if _is_hip:
        BLOCK = 8  # AMD GPU 使用更小的块：每次处理 8 个 KV pairs
    MAX_KV_SPLITS = max_kv_splits  # 最大分块数量
    Lk = k_buffer.shape[-1]        # Key 的实际维度大小
    Lv = v_buffer.shape[-1]        # Value 的实际维度大小

    # 获取 batch 大小和 head 数量
    batch, head_num = kv_indptr.shape[0] - 1, q.shape[1]

    # 定义 Triton kernel 的执行网格：(batch, head, kv_split)
    # 每个网格点对应一个线程块，处理特定的 (batch, head, split) 组合
    grid = (batch, head_num, MAX_KV_SPLITS)
    # 计算 KV 分组数量，用于判断是否为 GQA/MQA
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    # DEBUG: 创建原子计数器来统计 for 循环的总执行次数
    import torch
    loop_counter = torch.zeros(1, dtype=torch.int32, device=q.device)
    
    # DEBUG: 为每个grid存储详细的循环信息（仅在layer_id=0时启用）
    grid_info = None
    if layer_id == 0:
        # 每个grid存储4个值：[split_kv_start, split_kv_end, BLOCK_N, actual_loop_count]
        grid_info = torch.zeros(batch, head_num, MAX_KV_SPLITS, 4, dtype=torch.int32, device=q.device)

    # 根据 KV 分组情况设置线程束 (warp) 数量
    # 更多的 warp 可以提高并行度，但也会增加资源消耗
    if kv_group_num == 1:
        num_warps = 4  # MHA 情况，使用更多 warp
    else:
        num_warps = 2  # GQA/MQA 情况，使用较少 warp
        if _is_hip:
            num_warps = 1  # AMD GPU 进一步减少 warp 数量

    # 计算块大小，必须是 2 的幂次，用于 Triton 的内存访问优化
    BLOCK_DMODEL = triton.next_power_of_2(Lk)  # Key 维度的块大小
    BLOCK_DV = triton.next_power_of_2(Lv)      # Value 维度的块大小

    if layer_id == 0:
        # print(f"PUSH attnl0")
        nvtx.range_push(f"attnl0")

    fwd_kernel_stage1 = _fwd_kernel_stage1_layer0 if layer_id == 0 else _fwd_kernel_stage1

    # 启动 Triton kernel 进行 attention 计算的第一阶段
    # [grid] 语法指定了 kernel 的执行网格，每个网格点启动一个线程块
    fwd_kernel_stage1[grid](
        # 输入张量
        q,              # Query 张量
        k_buffer,       # Key buffer
        v_buffer,       # Value buffer
        sm_scale,       # Softmax 缩放因子
        kv_indptr,      # KV 索引指针
        kv_indices,     # KV 索引
        # 输出张量
        att_out,        # 中间输出结果
        att_lse,        # LSE 值
        num_kv_splits,  # KV 分块数量
        # DEBUG: 计数器
        loop_counter,   # 循环计数器
        # DEBUG: 网格信息
        grid_info,      # 网格详细信息
        # Stride 参数，用于正确访问多维张量
        q.stride(0),           # Query 的 batch stride
        q.stride(1),           # Query 的 head stride
        k_buffer.stride(0),    # K buffer 的 batch stride
        k_buffer.stride(1),    # K buffer 的 head stride
        v_buffer.stride(0),    # V buffer 的 batch stride
        v_buffer.stride(1),    # V buffer 的 head stride
        att_out.stride(0),     # 输出的 batch stride
        att_out.stride(1),     # 输出的 head stride
        att_out.stride(2),     # 输出的 split stride
        # 编译时常量参数
        kv_group_num=kv_group_num,      # KV 分组数量
        BLOCK_DMODEL=BLOCK_DMODEL,      # Key 维度块大小
        BLOCK_DV=BLOCK_DV,              # Value 维度块大小
        BLOCK_N=BLOCK,                  # 序列维度块大小
        MIN_BLOCK_KV=_MIN_BLOCK_KV,     # 最小 KV 块大小
        logit_cap=logit_cap,            # Logit 上限
        # Triton 执行配置参数
        num_warps=num_warps,            # 线程束数量
        num_stages=2,                   # 流水线阶段数
        Lk=Lk,                          # Key 实际维度
        Lv=Lv,                          # Value 实际维度
    )

    if layer_id == 0:
        nvtx.range_pop()

    # DEBUG: 打印循环执行次数统计
    if False and layer_id == 0:
        total_loops = loop_counter.item()
        # 打印调试信息，合并为一行输出，包括循环总执行次数、Grid大小和BLOCK_N
        print(f"[DEBUG] Layer {layer_id}: fwd_kernel_stage1 for loops = {total_loops}, Grid = (bsz={batch}, head={head_num}, splits={MAX_KV_SPLITS}), BLOCK_N = {BLOCK}")
        
        # DEBUG: 打印每个grid的详细循环信息
        if grid_info is not None:
            print("\n" + "="*80)
            print(f"[DEBUG] Layer {layer_id}: 每个Grid的详细For循环信息")
            print("="*80)
            print(f"{'Grid(b,h,s)':<12} {'Start':<6} {'End':<6} {'BLOCK_N':<8} {'Loops':<6} {'有效':<4}")
            print("-"*80)
            
            grid_data = grid_info.cpu().numpy()
            total_active_grids = 0
            total_detailed_loops = 0
            
            for b in range(batch):
                for h in range(head_num):
                    for s in range(MAX_KV_SPLITS):
                        start, end, block_n, loops = grid_data[b, h, s]
                        is_active = "是" if end > start else "否"
                        if end > start:
                            total_active_grids += 1
                            total_detailed_loops += loops
                        
                        # 只打印前50个grid和所有有效的grid
                        if (b * head_num * MAX_KV_SPLITS + h * MAX_KV_SPLITS + s) < 50 or end > start:
                            print(f"({b:2d},{h:2d},{s:2d})    {start:6d} {end:6d} {block_n:8d} {loops:6d} {is_active:4s}")
            
            print("-"*80)
            print(f"总计: {total_active_grids} 个有效Grid, 详细循环统计={total_detailed_loops}, 原子计数器={total_loops}")
            if total_detailed_loops != total_loops:
                print(f"⚠️  警告: 详细统计({total_detailed_loops}) != 原子计数器({total_loops})")
            print("="*80)


@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
        if BLOCK_DPE > 0:
            qpe = tl.load(
                Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
            )
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
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
            qk = tl.dot(q, k.to(q.dtype))
            if BLOCK_DPE > 0:
                offs_buf_kpe = (
                    kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[:, None]
                )
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

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

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

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


def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
):
    BLOCK = 32
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    # [TODO] work around shmem limit on MI3xx
    if _is_hip and Lk >= 576:
        BLOCK = 16

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    batch, head_num = kv_indptr.shape[0] - 1, q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    BLOCK_H = 16
    MAX_KV_SPLITS = max_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        MAX_KV_SPLITS,
    )

    extra_kargs = {}
    num_stages = 2
    if _is_hip:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1

    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        num_warps=4,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        **extra_kargs,
    )


@triton.jit
def _fwd_kernel_stage2(
    # 输入张量：Stage1 的输出结果
    Mid_O,          # Stage1 部分结果: [batch_size, num_q_heads, max_kv_splits, v_head_dim]
    Mid_O_1,        # Stage1 LSE 值: [batch_size, num_q_heads, max_kv_splits] 
    # 输出张量：最终的 attention 结果
    O,              # 最终输出: [batch_size, num_q_heads, v_head_dim]
    # 索引和控制信息
    kv_indptr,      # KV 指针: [batch_size + 1] - 序列长度信息
    num_kv_splits,  # 实际分块数: [batch_size] - 每个序列实际使用的 split 数
    
    # Stride 参数：用于多维张量内存访问
    # =====================================
    stride_mid_ob,  # 中间结果 batch stride: int - Mid_O[b+1] - Mid_O[b]
    stride_mid_oh,  # 中间结果 head stride: int - Mid_O[:, h+1] - Mid_O[:, h]  
    stride_mid_os,  # 中间结果 split stride: int - Mid_O[:, :, s+1] - Mid_O[:, :, s]
    stride_obs,     # 最终输出 batch stride: int - O[b+1] - O[b]
    stride_oh,      # 最终输出 head stride: int - O[:, h+1] - O[:, h]
    
    # 编译时常量参数
    # ===============
    MAX_KV_SPLITS: tl.constexpr,   # 最大分块数: int - 预分配的 split 维度大小
    MIN_BLOCK_KV: tl.constexpr,    # 最小KV块: int - 对齐参数
    BLOCK_DV: tl.constexpr,        # Value 维度块: int - 必须 >= v_head_dim 且为2的幂
    Lv: tl.constexpr,              # Value 实际维度: int - 真实的 v_head_dim
):
    """
    Attention 计算的第二阶段 kernel (Stage 2)
    
    这是 Flash Attention decoding 的第二阶段，负责合并 stage1 产生的多个部分结果。
    Stage1 将长序列分割成多个 split 并行处理，每个 split 产生一个部分的 attention 结果。
    Stage2 需要将这些部分结果正确合并，得到最终的 attention 输出。
    
    主要工作流程：
    1. 获取当前线程块负责的 batch 和 head
    2. 遍历该 (batch, head) 对应的所有 split
    3. 使用 online softmax 技术合并各个 split 的结果
    4. 存储最终的 attention 输出
    
    关键算法：使用 LSE 值正确合并多个 softmax 结果
    如果有两个 softmax 结果 s1, s2，对应的 LSE 值为 lse1, lse2，
    合并后的结果为：(s1 * exp(lse1 - max_lse) + s2 * exp(lse2 - max_lse)) / (exp(lse1 - max_lse) + exp(lse2 - max_lse))
    """
    # 获取当前线程块的二维索引：(batch, head)
    # Stage2 只需要按 batch 和 head 并行，不需要 split 维度的并行
    cur_batch = tl.program_id(0)  # 当前 batch 索引
    cur_head = tl.program_id(1)   # 当前 head 索引

    # 获取当前 batch 的序列长度和分块数量
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(
        kv_indptr + cur_batch
    )
    kv_splits = tl.load(num_kv_splits + cur_batch)

    # 创建 Value 维度的索引和掩码
    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    # 初始化合并算法的状态变量
    # 这些变量用于实现 online softmax 合并多个 split 的结果
    e_sum = 0.0                                   # 当前的指数和
    e_max = -float("inf")                         # 当前的最大 LSE 值
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)  # 累积的 attention 输出

    # 计算中间结果的内存偏移量基地址
    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d  # Value 结果的基地址
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv  # LSE 值的基地址
    
    # 计算每个 split 的长度（与 stage1 保持一致）
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    # Stage2 合并循环：遍历所有 split 并合并结果
    # =============================================
    # Split 合并策略详解：
    # - 每个 split 都有自己的部分结果 (tv) 和 LSE 值 (tlogic)
    # - 需要使用 online softmax 算法正确合并这些部分结果
    # - 最终得到等价于完整序列 attention 的结果
    for split_kv_id in range(0, MAX_KV_SPLITS):
        # 计算当前 split 的范围，判断是否有效
        # （与 Stage1 保持一致的分割策略）
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        # 只处理有效的 split（包含实际数据的 split）
        if split_kv_end > split_kv_start:
            # 加载当前 split 的部分 attention 结果
            # tv: [BLOCK_DV] - 当前 split 的归一化 attention 输出
            # 这是 Stage1 中 acc/e_sum 的结果
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, 
                mask=mask_d, 
                other=0.0
            )
            
            # 加载当前 split 的 LSE 值
            # tlogic: 标量 - 当前 split 的 log-sum-exp 值
            # LSE = max + log(sum)，包含了归一化所需的信息
            tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)
            
            # Online softmax 合并算法的核心
            # 更新全局最大 LSE 值
            n_e_max = tl.maximum(tlogic, e_max)

            # 计算之前累积结果的重缩放因子
            old_scale = tl.exp(e_max - n_e_max)
            # 重缩放之前的累积结果
            acc *= old_scale
            # 计算当前 split 的权重
            exp_logic = tl.exp(tlogic - n_e_max)
            # 加权累加当前 split 的结果
            acc += exp_logic * tv

            # 更新指数和与最大值
            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    # 存储最终的 attention 输出结果
    # acc / e_sum 是正确归一化的 attention 输出
    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def _decode_softmax_reducev_fwd(
    logits,         # Stage1 的中间 attention 结果 [batch, head, splits, head_dim]
    lse,            # Stage1 的 LSE 值 [batch, head, splits]
    q,              # Query 张量（用于获取形状信息）
    o,              # 最终输出张量 [batch, head, head_dim]
    v_buffer,       # Value buffer（用于获取维度信息）
    kv_indptr,      # KV 索引指针
    num_kv_splits,  # 每个 batch 的实际分块数量
    max_kv_splits,  # 最大分块数量
):
    """
    Stage2 的包装函数：合并多个 split 的 attention 结果
    
    这是 decode attention 的第二阶段包装函数，负责配置和启动 stage2 kernel。
    它的主要作用是：
    1. 根据输入张量确定执行配置参数
    2. 针对不同硬件平台设置优化参数
    3. 启动 _fwd_kernel_stage2 进行实际的结果合并
    
    被 decode_attention_fwd_normal 和 decode_attention_fwd_grouped 调用，
    是两阶段 attention 计算中不可缺少的第二步。
    """
    # 获取执行配置参数
    batch, head_num = q.shape[0], q.shape[1]  # batch 大小和 head 数量
    Lv = v_buffer.shape[-1]                   # Value 的实际维度
    BLOCK_DV = triton.next_power_of_2(Lv)     # Value 维度的块大小

    MAX_KV_SPLITS = max_kv_splits

    # AMD GPU 的特殊优化参数
    extra_kargs = {}
    if _is_hip:
        # 针对 AMD ROCm 平台的特定优化参数
        # waves_per_eu: 每个执行单元的波数
        # matrix_instr_nonkdim: 矩阵指令的非K维度
        # kpack: K维度的打包因子
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    # Stage2 只需要按 (batch, head) 并行，不需要 split 维度
    grid = (batch, head_num)
    
    # 启动 Stage2 kernel 进行结果合并
    _fwd_kernel_stage2[grid](
        # 输入张量
        logits,         # Stage1 的中间结果
        lse,            # Stage1 的 LSE 值
        o,              # 最终输出张量
        kv_indptr,      # KV 索引指针
        num_kv_splits,  # 分块数量
        # Stride 参数
        logits.stride(0),  # 中间结果的 batch stride
        logits.stride(1),  # 中间结果的 head stride
        logits.stride(2),  # 中间结果的 split stride
        o.stride(0),       # 输出的 batch stride
        o.stride(1),       # 输出的 head stride
        # 编译时常量
        MAX_KV_SPLITS=MAX_KV_SPLITS,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        # Triton 执行配置
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )


def decode_attention_fwd_normal(
    q,              # Query 张量
    k_buffer,       # Key buffer
    v_buffer,       # Value buffer
    o,              # 输出张量
    kv_indptr,      # KV 索引指针
    kv_indices,     # KV 索引
    attn_logits,    # 中间结果张量
    attn_lse,       # LSE 张量
    num_kv_splits,  # KV 分块数量
    max_kv_splits,  # 最大分块数量
    sm_scale,       # Softmax 缩放因子
    logit_cap=0.0,  # Logit 上限
    layer_id=None,  # Layer ID for debugging
):
    """
    标准 MHA (Multi-Head Attention) 的 decode attention 实现
    
    用于处理 kv_group_num == 1 的情况，即每个 query head 都有对应的 key-value head。
    执行两阶段计算：
    1. Stage1: 调用 _decode_att_m_fwd 计算部分 attention 结果
    2. Stage2: 调用 _decode_softmax_reducev_fwd 合并最终结果
    """
    # Stage1: 计算分块的 attention 结果
    # 每个 split 独立计算部分的 QK^T @ V，产生中间结果和 LSE 值
    _decode_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        attn_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        logit_cap,
        layer_id,
    )
    # Stage2: 合并所有 split 的结果
    # 使用 LSE 值正确合并多个部分结果，得到最终的 attention 输出
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_buffer,
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
    )


def decode_attention_fwd_grouped(
    q,              # Query 张量
    k_buffer,       # Key buffer
    v_buffer,       # Value buffer
    o,              # 输出张量
    kv_indptr,      # KV 索引指针
    kv_indices,     # KV 索引
    attn_logits,    # 中间结果张量
    attn_lse,       # LSE 张量
    num_kv_splits,  # KV 分块数量
    max_kv_splits,  # 最大分块数量
    sm_scale,       # Softmax 缩放因子
    logit_cap=0.0,  # Logit 上限
):
    """
    分组 GQA/MQA (Grouped/Multi-Query Attention) 的 decode attention 实现
    
    用于处理 kv_group_num > 1 的情况，即多个 query heads 共享 key-value heads。
    这种设计可以显著减少 KV cache 的内存使用，同时保持模型性能。
    
    执行流程与标准 MHA 相同，但使用专门优化的 grouped kernel：
    1. Stage1: 调用 _decode_grouped_att_m_fwd 处理分组情况
    2. Stage2: 调用 _decode_softmax_reducev_fwd 合并结果
    """
    # Stage1: 使用分组优化的 kernel 计算部分 attention 结果
    # 处理多个 query heads 共享 key-value heads 的情况
    _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        attn_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        logit_cap,
    )
    # Stage2: 合并结果（与标准 MHA 相同）
    # 分组只影响 stage1 的计算，stage2 的合并逻辑保持一致
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_buffer,
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
    )


def decode_attention_fwd(
    q,              # Query 张量: [batch_size, num_q_heads, head_dim]
    k_buffer,       # Key buffer: [total_tokens, num_kv_heads, head_dim] - 存储所有历史 key vectors
    v_buffer,       # Value buffer: [total_tokens, num_kv_heads, v_head_dim] - 存储所有历史 value vectors
    o,              # 输出张量: [batch_size, num_q_heads, v_head_dim] - attention 的最终输出
    kv_indptr,      # KV 索引指针: [batch_size + 1] - CSR格式，指示每个序列的KV范围
    kv_indices,     # KV 索引: [total_seq_len] - 每个位置对应的实际token索引  
    attn_logits,    # 中间结果张量: [batch_size, num_q_heads, max_kv_splits, v_head_dim] - Stage1的部分结果
    attn_lse,       # LSE 张量: [batch_size, num_q_heads, max_kv_splits] - 每个split的log-sum-exp值
    num_kv_splits,  # 分块数量: [batch_size] - 每个batch实际使用的split数量
    max_kv_splits,  # 最大分块数量: int - 预分配的最大split数量
    sm_scale,       # Softmax 缩放因子: float - 通常是 1/sqrt(head_dim)
    logit_cap=0.0,  # Logit 上限: float - 用于数值稳定性，0表示不启用
    layer_id=None,  # Layer ID for debugging
):
    """
    Decode Attention 的主要入口函数
    
    这是整个 decode attention 计算的统一入口点，根据 attention 类型自动选择合适的实现：
    - MHA (Multi-Head Attention): kv_group_num == 1，所有 query heads 都有对应的 key-value heads
    - GQA (Grouped Query Attention): kv_group_num > 1，多个 query heads 共享一个 key-value head
    - MQA (Multi-Query Attention): 特殊的 GQA 情况，所有 query heads 共享一个 key-value head
    
    算法流程：
    1. 判断 attention 类型（MHA vs GQA/MQA）
    2. 调用对应的实现函数
    3. 执行两阶段计算：Stage1 并行计算部分结果，Stage2 合并最终结果
    
    Tensor Shape 详解：
    ==================
    输入张量形状关系：
    - batch_size: 当前处理的序列数量
    - num_q_heads: Query head 的数量（通常是 num_kv_heads 的倍数）
    - num_kv_heads: Key-Value head 的数量
    - head_dim: Key 和 Query 的维度
    - v_head_dim: Value 的维度（可能与 head_dim 不同）
    - total_tokens: KV cache 中存储的总 token 数量
    - total_seq_len: 所有序列的总长度（可能小于 total_tokens）
    
    关键关系：
    - kv_group_num = num_q_heads // num_kv_heads（决定 attention 类型）
    - kv_indptr[i+1] - kv_indptr[i] = 第i个序列的长度
    - sum(kv_indptr[i+1] - kv_indptr[i]) = total_seq_len
    
    在 SGLang 的调用链中的位置：
    - 被 triton_backend.py 中的 TritonAttnBackend.forward_decode() 调用
    - 是 decoding 阶段 attention 计算的核心实现
    - 与 prefill 阶段的 attention 计算分离，专门优化单 token 生成场景
    
    Args:
        q: 当前时刻的 query [batch_size, num_q_heads, head_dim]
        k_buffer: 历史 key buffer [total_tokens, num_kv_heads, head_dim]
        v_buffer: 历史 value buffer [total_tokens, num_kv_heads, v_head_dim]
        o: attention 输出 [batch_size, num_q_heads, v_head_dim]
        kv_indptr: 序列范围指针 [batch_size + 1]，CSR格式索引
        kv_indices: token 索引 [total_seq_len]，指向 buffer 中的实际位置
        attn_logits: 中间结果 [batch_size, num_q_heads, max_kv_splits, v_head_dim]
        attn_lse: LSE 值 [batch_size, num_q_heads, max_kv_splits]
        num_kv_splits: 实际分块数 [batch_size]
        max_kv_splits: 最大分块数 (int)
        sm_scale: 缩放因子 (float)
        logit_cap: logit 上限 (float)
    """
    # 输入验证：确保张量形状的一致性
    assert max_kv_splits == attn_logits.shape[2]  # 分块数量与中间张量一致
    assert q.shape[0] <= kv_indptr.shape[0] - 1   # batch 数量不超过 KV 索引范围
    assert q.shape[0] <= attn_logits.shape[0]     # batch 数量与输出张量一致

    # 计算 KV 分组数量，用于判断 attention 类型
    # kv_group_num = query_heads / kv_heads
    kv_group_num = q.shape[1] // v_buffer.shape[1]

    if kv_group_num == 1:
        # MHA (Multi-Head Attention) 情况
        # 每个 query head 都有对应的 key-value head，使用标准实现
        decode_attention_fwd_normal(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            logit_cap=logit_cap,
            layer_id=layer_id,
        )
    else:
        # GQA (Grouped Query Attention) / MQA (Multi-Query Attention) / MLA 情况
        # 多个 query heads 共享 key-value heads，需要特殊的分组处理
        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            logit_cap=logit_cap,
        )
