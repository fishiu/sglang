# Quest GQA Design (Weak vs Strong)

> This document is for the next agent to quickly understand the current
> Quest + GQA design, what has been implemented, what is still messy, and
> what pitfalls we already hit in experiments.

The current repo has two logical GQA modes on top of Quest:

- **Weak GQA**: each Q head behaves almost like MHA – it selects pages
  and tokens independently, but reads from a shared KV‑head KV cache.
- **Strong GQA**: Q heads that share the same KV head act as a *group*.
  They jointly decide which pages to keep (grouped estimate/top‑k) and
  the decode kernel computes attention for the whole group at once,
  reusing K/V loads.

Both modes are implemented only for **decode**; extend/prefill still use
the Triton backend (dense attention).

Below `Hq` = per‑TP query heads, `Hkv` = per‑TP KV heads,
`g = kv_group_num = Hq / Hkv`, `P` = number of pages, and we assume
GQA: `Hq % Hkv == 0`.

---

## 1. Control knobs and high‑level flow

### 1.1 Server args and backend flags

File: `python/sglang/srt/server_args.py`

- Config fields:
  - `enable_quest: bool`
  - `quest_topk: int`
  - `quest_estimate_splits: int`
  - **New:** `quest_use_weak_gqa: bool = False`
    - CLI flag: `--quest-use-weak-gqa` (store_true)
    - When `True`, GQA uses the *weak* behavior.
    - When `False` (default), GQA uses *strong* behavior.

File: `python/sglang/srt/layers/attention/quest_backend.py`

- In `QuestAttnBackend.__init__`:
  - `tp_size = get_attention_tp_size()`
  - `self.num_q_head = model_config.num_attention_heads // tp_size`
  - `self.num_kv_head = model_config.get_num_kv_heads(tp_size)`
  - `self.num_head = self.num_q_head` (legacy alias for Q head count)
  - `self.head_dim = model_config.head_dim`
  - `self.use_weak_gqa = model_runner.server_args.quest_use_weak_gqa`

### 1.2 Mode selection per layer

In `QuestAttnBackend.forward_decode`:

```python
kv_group_num = 1
if layer.tp_k_head_num > 0:
    kv_group_num = layer.tp_q_head_num // layer.tp_k_head_num
is_mha = kv_group_num == 1
use_weak = self.use_weak_gqa or is_mha
```

- **MHA**: `kv_group_num == 1` → `is_mha=True` → always `use_weak=True`.
- **GQA weak mode**: `kv_group_num > 1` and `quest_use_weak_gqa=True`.
- **GQA strong mode**: `kv_group_num > 1` and `quest_use_weak_gqa=False`.

Additionally:

- KV update / metadata are always done per KV head, regardless of mode:
  ```python
  k_new = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
  v_new = v.view(-1, layer.tp_k_head_num, layer.v_head_dim)
  quest_update_kv_and_metadata(...)
  ```

The differences between weak and strong modes are in **estimate** and
**decode**.

---

## 2. Weak GQA: behavior and implementation

Weak GQA tries to keep the implementation close to MHA, mostly for
correctness and as a baseline.

### 2.1 Estimate (weak GQA)

File: `python/sglang/srt/layers/attention/triton_ops/quest_attention.py`

Function: `quest_estimate_scores(...)` (with `grouped=False`).

Inputs:

- `q: [B, Hq, D]`
- `k_metadata: [num_pages, Hkv, D, 2]` (min/max per page per KV head)
- `estimated_scores: [B, Hq, max_pages]`

Steps:

1. Compute `BLOCK_DMODEL = next_power_of_2(D)`.
2. `grid = (B, Hq, NUM_SPLITS)`; kernel `quest_estimate_split_kernel`
   loops over pages:
   - For each `(b, q_head=hq, split)`:
     - Map `hq` to KV head `kv_h = hq // kv_group_num`.
     - Load `q[b, hq, :]` and (min,max) for `(page, kv_h)` from metadata.
     - Compute upper bound:
       `score(page) = Σ_d max(q_d * k_min_d, q_d * k_max_d)`.
3. Write into `estimated_scores[b, hq, pid]` for all pages `pid`.

In weak mode:

- `grouped=False` → no extra reduction.
- Each Q head gets its own score for each page, even though metadata is
  per KV head.

### 2.2 Select + CSR (weak GQA)

Functions:

- `quest_select_topk_pages(...)`
- `quest_select_topk_pages_into(...)`

Inputs:

- `estimated_scores: [B, Hq, max_pages]`
- `seq_lens: [B]`, `req_to_token: [B, max_context_len]`
- `quest_topk`, `page_size`

High level:

1. For each `(b, h)` independently:
   - Select `quest_topk-1` best non‑last pages + always append last page.
2. Expand pages to tokens via `req_to_token` into `kv_indices_buf`:
   - `kv_indices_buf`: `[B*Hq, tokens_cap]`.
   - `tokens_per_head`: `[B*Hq]` count for each `(b,h)`.
3. Because Quest decode kernel expects **same token count per head in a
   batch**, do:
   - `tokens_per_batch[b] = tokens_per_head[b, 0]`  
     (we assert all heads have equal counts; with current logic they do).
   - `kv_indptr[b+1] = kv_indptr[b] + tokens_per_batch[b]`.
4. Pack into final `kv_indices` layout expected by decode kernel:
   - Each Q head has `tokens_per_batch[b]` tokens.
   - `kv_indices` shape is flat 1D; indexing scheme is documented in
     comments inside `_quest_decode_kernel_stage1`.

Important: in weak GQA, heads **do not necessarily select the same
pages**, but they do have the same *number* of tokens per batch. The
decode kernel handles per‑head indices separately (no assumption of
shared tokens).

### 2.3 Decode (weak GQA / MHA)

Kernel: `_quest_decode_kernel_stage1` (single‑head version, unchanged).

Wrapper: `quest_decode_attention_fwd(...)` with
`force_kv_group_num=1` in backend when `use_weak=True`:

```python
self.quest_decode_attention_fwd(
    q=...,
    k_buffer=...,
    v_buffer=...,
    ...,
    force_kv_group_num=1 if use_weak else None,
)
```

Inside `quest_decode_attention_fwd`:

- For `kv_group_num == 1`:
  - `grid = (batch, head_num, MAX_KV_SPLITS)`.
  - Call `_quest_decode_kernel_stage1[...]` exactly as in the original
    Quest MHA implementation:
    - For given `(b, q_head, split)`:
      - `cur_kv_head = q_head // kv_group_num` → same as `q_head` here.
      - Use `kv_indptr[b]` and `kv_indices` to compute the token window
        for this head and split.
      - Accumulate `QK` and `PV` via online softmax into
        `Att_Out[b, q_head, split, :]` and `Att_Lse`.
  - Stage2 `_quest_decode_kernel_stage2` merges splits.

Thus weak GQA decode is basically “per head Quest decode” where Q→KV
mapping is used only to pick the KV head index for K/V loads, but each
head still has its own CSR tokens.

---

## 3. Strong GQA: semantics, current implementation, pitfalls

Strong GQA tries to exploit that Q heads in each GQA group share the
same KV head in the model. The intended semantics:

- For each KV group (size = `g = kv_group_num`), we want to:
  1. Let all Q heads in the group contribute to the *page score*.
  2. Use the aggregated group score to decide which pages to keep.
  3. Ensure that all Q heads in the group see the *same tokens*.
  4. In decode, compute attention for the whole group in one kernel,
     reusing K/V loads.

### 3.1 Estimate in strong GQA (group‑reduce on scores, not Q)

Important: we purposely **do not** pre‑sum Q vectors, because the
estimate formula is nonlinear:

```text
score(q, K_min, K_max) = Σ_d max(q_d * k_min_d, q_d * k_max_d)
```

In general:

```text
score(q1 + q2, K) ≠ score(q1, K) + score(q2, K)
```

So the correct way to get a “group score” is:

1. Compute per‑head scores: `score(q_i, K_p)` for each head `i` and
   page `p`.
2. Sum over heads in the same GQA group:
   `score_group(p) = Σ_i score(q_i, K_p)` (over i in group).
3. Use `score_group` for TopK.

This is what the current implementation does.

#### Implementation

File: `quest_attention.py:565`, `quest_estimate_scores(...)`:

```python
def quest_estimate_scores(..., grouped: bool = False, ...):
    ...
    num_kv_heads = k_metadata.shape[1]
    kv_group_num = num_q_heads // num_kv_heads
    quest_estimate_split_kernel[grid](...)  # per-Q-head scores
    ...
    if grouped and kv_group_num > 1:
        est_view = estimated_scores.view(batch_size, num_kv_heads, kv_group_num, max_pages)
        est_group = est_view.sum(dim=2, keepdim=True)
        est_view.copy_(est_group.expand_as(est_view))
```

Backend (`quest_backend.py`) passes:

- Non‑graph:
  ```python
  self.quest_estimate_scores(..., grouped=not use_weak)
  ```
- Graph:
  ```python
  self.quest_estimate_scores(..., estimated_scores=est_buf, ..., grouped=not use_weak)
  ```

So:

- **weak GQA / MHA**: `grouped=False`, behavior identical to per‑head
  version.
- **strong GQA**: `grouped=True`, scores of all heads in a KV group are
  summed per page and then broadcast to each Q head in the group.

Resulting semantics:

- After estimate, `estimated_scores[b, h, :]` is identical for all
  Q heads `h` in the same KV group. This is the key to ensure they
  choose identical pages in the next phase.

### 3.2 Select + CSR in strong GQA

We deliberately did **not** introduce a new TopK kernel. Instead:

- Both weak and strong GQA use the same `quest_select_topk_pages` /
  `quest_select_topk_pages_into`.
- In strong GQA, because the `estimated_scores` for heads in the same KV
  group are equal, they will produce identical `selected_pages` and
  identical CSR tokens per head.

Important note:

- There are helper wrappers
  `quest_select_topk_pages_grouped` / `quest_select_topk_pages_into_grouped`
  currently defined but not used; the actual behavior is fully driven by
  score broadcasting before calling the original selection functions.
- The CSR layout is still “per head”: for each batch `b`, each head
  `h` gets a contiguous block of `tokens_per_batch[b]` tokens.

This is a design choice for now: it keeps the index layout unchanged and
lets the grouped decode kernel derive a per‑KV‑head token view from the
per‑head layout, instead of changing the CSR format.

### 3.3 Grouped decode kernel

File: `quest_attention.py:198`, `_quest_decode_grouped_kernel_stage1`.

Wrapper: `quest_decode_attention_fwd(...)` (same file:430) chooses
kernel as follows:

```python
kv_group_num = q.shape[1] // k_buffer.shape[1]
if force_kv_group_num is not None:
    kv_group_num = force_kv_group_num

if kv_group_num == 1:
    # MHA / weak GQA → _quest_decode_kernel_stage1
else:
    # strong GQA → _quest_decode_grouped_kernel_stage1
```

`QuestAttnBackend.forward_decode` calls:

```python
self.quest_decode_attention_fwd(
    q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
    ...,
    force_kv_group_num=1 if use_weak else None,
)
```

So:

- `use_weak=True` (MHA or weak GQA) → forced `kv_group_num=1` →
  single‑head kernel.
- `use_weak=False` (strong GQA) → `kv_group_num=Hq/Hkv` →
  grouped kernel.

#### CSR layout assumptions

Given `kv_indptr[b]` and `kv_indices`, the original Quest decode kernel
assumes:

- Per batch `b`, `tokens_per_batch = kv_indptr[b+1] - kv_indptr[b]`.
- Each head has `tokens_per_batch` tokens.
- Tokens for different heads are laid out as:

```text
for b in [0..B-1]:
  base_tokens_before = kv_indptr[b]
  for h in [0..Hq-1]:
    for t in [0..tokens_per_batch-1]:
      idx = base_tokens_before * Hq + h * tokens_per_batch + t
      kv_indices[idx] = token_id(b, h, t)
```

This is encoded inside `_quest_decode_kernel_stage1` via:

```python
cur_batch_seq_len = tokens_per_batch
cur_batch_base_tokens = tl.load(kv_indptr + cur_batch)
cur_batch_kv_start_idx = cur_head * cur_batch_seq_len + cur_batch_base_tokens * head_num
```

The grouped kernel reuses this layout: it treats the first head in a KV
group as the canonical token sequence and ignores the duplicate copies
for the other heads.

#### `_quest_decode_grouped_kernel_stage1` behavior

Grid: `grid = (batch, num_kv_heads, MAX_KV_SPLITS)`:

- `cur_batch = program_id(0)`
- `cur_kv_head = program_id(1)` → KV head index
- `split_kv_id = program_id(2)`

Within the kernel:

- Head group:
  ```python
  BLOCK_H = kv_group_num
  offs_h = tl.arange(0, BLOCK_H)          # [0..g-1]
  cur_head = cur_kv_head * kv_group_num + offs_h   # Q head indices in this group
  mask_h = cur_head < num_q_heads
  ```

- Q load:
  ```python
  offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
  q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
  # q: [BLOCK_H, BLOCK_DMODEL]
  ```

- Tokens:
  ```python
  cur_batch_token_start = kv_indptr[cur_batch]
  cur_batch_seq_len = kv_indptr[cur_batch+1] - cur_batch_token_start
  kv_len_per_split = ...
  split_kv_start, split_kv_end = ...

  # base offset for group head 0
  base_head = cur_kv_head * kv_group_num
  base = cur_batch_token_start * num_q_heads + base_head * cur_batch_seq_len
  ```
  This uses exactly the packing scheme described above.

- Inside split loop:

  ```python
  offs_n = start_n + tl.arange(0, BLOCK_N)
  kv_loc = tl.load(
      kv_indices + base + offs_n,
      mask=offs_n < split_kv_end,
      other=0,
  )
  # K load: [BLOCK_N, BLOCK_DMODEL]
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

  # QK: [BLOCK_H, BLOCK_N]
  qk = tl.sum(q[:, None, :] * k[None, :, :], axis=2)
  qk *= sm_scale
  ...

  # V load & PV
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
  p = tl.exp(qk - n_e_max[:, None])
  acc *= re_scale[:, None]
  acc += tl.sum(p[:, :, None] * v[None, :, :], axis=1)
  ```

Key points:

- All Q heads in the group share the same `kv_loc` token sequence
  (coming from group head 0).
- K/V are loaded once per `(batch, kv_head, split)` and reused for all
  Q heads in the group.
- Online softmax is fully vectorized over `BLOCK_H` (head) dimension.

Final store:

```python
offs_mid_o = (
    cur_batch * stride_mid_ob
    + cur_head[:, None] * stride_mid_oh
    + split_kv_id * stride_mid_os
    + offs_dv[None, :]
)
tl.store(Att_Out + offs_mid_o, acc / e_sum[:, None], mask=...)
...
tl.store(Att_Lse + offs_mid_o_1, e_max + tl.log(e_sum), mask=mask_h)
```

So `Att_Out` / `Att_Lse` are filled for all Q heads in the group.
Stage2 remains `_quest_decode_kernel_stage2` and works unchanged.

### 3.4 CUDA Graph compatibility

Graph path has been updated to mirror eager behavior:

- Estimate: `grouped=not use_weak` is also used for `est_buf`.
- Select: same `quest_select_topk_pages_into(est_buf, ...)` after score
  broadcasting.  
  Implementation: see `quest_backend.py:360–423`.

There is no special graph‑only grouped kernel; the same kernels are
called from within graph capture.

---

## 4. Known issues / pitfalls

This section is to warn the next agent where we already tripped.

### 4.1 tl.dot constraints (fixed)

We initially tried to use `tl.dot` inside
`_quest_decode_grouped_kernel_stage1`:

- `qk = tl.dot(q, k.to(q.dtype))`
- `acc += tl.dot(p.to(v.dtype), v)`

Triton requires `M, N, K >= 16` for `tl.dot` and strict `[M,K] @ [K,N]`
shapes. Our grouped kernel had shapes like `[g,D]` and `g < 16`, and K/V
dims were not in `[M,K]` layout, causing compile‑time assertions.

Fix: we replaced these with explicit broadcast + sum:

```python
qk = tl.sum(q[:, None, :] * k[None, :, :], axis=2)
acc += tl.sum(p[:, :, None] * v[None, :, :], axis=1)
```

This is slower than an ideal TensorCore kernel, but correct and
constraint‑free. A future optimization could selectively dispatch to
`tl.dot` when shapes are large enough.

### 4.2 Broadcast masks on tl.load (fixed)

In grouped kernel K/V loads, we initially used masks with wrong
broadcast order, e.g.:

```python
offs_buf_k: [BLOCK_N, BLOCK_DMODEL]
mask = (offs_n[None, :] < split_kv_end) & (mask_d[:, None])
```

This broadcasts to `[BLOCK_DMODEL, BLOCK_N]` and conflicts with
`offs_buf_k` shape, triggering Triton `Cannot broadcast` errors.

Fix: align masks with `offs_buf_k`:

```python
mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :])
```

Same for V.

### 4.3 Performance still not ideal

Even after grouped decode:

- Strong GQA Quest is still not clearly faster than weak GQA or MHA
  Quest in benchmarks like Qwen3‑4B, `bs=64, input_len=1000`.  
- Likely reasons:
  - Additional work in estimate (group sum + broadcast).
  - Grouped decode kernel not tuned (BLOCK_N/BLOCK_DMODEL/num_warps
    currently chosen conservatively).
  - Extra ATen kernels (`torch.zeros` for debug_info,
    tokens_per_batch/cumsum, etc.) causing bubbles under CUDA Graph.

This is not a correctness issue but a tuning problem; see next section
for suggestions.

---

## 5. Suggestions for the next agent

If you take over this work, here are concrete directions.

### 5.1 Clean up / clarify strong vs weak

- Decide whether to keep the unused wrappers:
  - `quest_select_topk_pages_grouped`
  - `quest_select_topk_pages_into_grouped`
  They are currently dead code; either wire them up properly or delete
  them to avoid confusion.
- Consider splitting kernels explicitly like Triton backend:
  - `_quest_decode_kernel_stage1_mha`
  - `_quest_decode_kernel_stage1_grouped`
  instead of overloading names, purely for clarity and profiling.

### 5.2 Verify CSR layout with small tests

Write a small unit test (or debug script) that:

- Fix `B=1`, `Hq=4`, `Hkv=2`, `g=2`, small `tokens_per_batch`.
- Manually build `kv_indices` using existing `quest_select_topk_pages`.
- In Python, reverse‑engineer what the grouped kernel thinks as
  “base token sequence” per KV head and verify it matches the first Q
  head’s tokens in that group.

This will give high confidence that the indexing math in
`_quest_decode_grouped_kernel_stage1` is correct.

### 5.3 Performance tuning

Once correctness is solid, focus on:

- **Decode kernel parameters**:
  - Tune `BLOCK_N`, `BLOCK_DMODEL`, `BLOCK_DV`, `num_warps` separately
    for grouped kernel (currently we mostly mirror `_quest_decode_kernel_stage1`).
  - Check occupancy / DRAM throughput vs Triton’s
    `_fwd_grouped_kernel_stage1`.
- **Remove debug allocations**:
  - `debug_info = torch.zeros((batch, head_num, max_kv_splits, 5), ...)`
    is always allocated but never used; removing or guarding it by an
    env/flag will cut one big memset kernel.
- **Reduce ATen micro‑kernels in select path**:
  - `tokens_per_batch.copy_`, `torch.cumsum`, etc., may cause bubbles
    under CUDA Graph. Consider moving cumsum into a Triton kernel or
    tightly merging with expand/pack.

### 5.4 CUDA Graph specific care

Under graph capture:

- Ensure `kv_group_num` and `use_weak` are kept consistent between
  capture and replay.
- Avoid shape‑dependent branches that might differ between capture
  batches (but current design uses static `max_bs`/`max_num_tokens`, so
  this should be OK).

---

## 6. Summary

- **Weak GQA**: per‑Q‑head estimate/select/decode, only KV cache is
  shrunk to Hkv; decode is single‑head Quest kernel with Q→KV mapping.
- **Strong GQA**:  
  - Estimate: same per‑head kernel, then group sum over heads in each KV
    group and broadcast back to all heads → group decides pages.  
  - Select: same kernels, but because scores are identical within each
    group, resulting pages/tokens are identical per group.  
  - Decode: grouped kernel `_quest_decode_grouped_kernel_stage1` reuses
    one token/K/V stream per KV head and computes all group heads
    together.
- Current code implements this semantics both for eager and CUDA Graph
  paths, but performance is not yet tuned; the grouped kernel is
  functionally correct but not fully optimized.

This document should help you orient yourself quickly; if you want a
single starting point, focus on:

- `quest_backend.py:forward_decode`
- `quest_attention.py:quest_estimate_scores`,
  `_quest_decode_kernel_stage1`,
  `_quest_decode_grouped_kernel_stage1`,
  `quest_decode_attention_fwd`.

