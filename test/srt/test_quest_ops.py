import unittest

import torch

from sglang.test.test_utils import CustomTestCase
from sglang.srt.layers.attention.triton_ops.quest_attention import (
    quest_update_kv_and_metadata,
    quest_estimate_scores,
    quest_select_topk_pages_into,
)


class TestQuestOps(CustomTestCase):
    def test_update_kv_and_metadata_minmax(self):
        device = "cuda"
        dtype = torch.float32
        B, H, D = 3, 4, 8
        page_size = 4

        # Prepare new K/V for B sequences
        k_new = torch.randn(B, H, D, device=device, dtype=dtype)
        v_new = torch.randn(B, H, D, device=device, dtype=dtype)

        # Place each token on a distinct global page to avoid races when updating page metadata.
        # With page_size=4, locations [0, 4, 8, ...] map to unique pages [0, 1, 2, ...].
        out_cache_loc = (torch.arange(B, dtype=torch.int32, device=device) * page_size)
        size = int(out_cache_loc.max().item()) + 1

        k_buf = torch.empty(size, H, D, device=device, dtype=v_new.dtype)
        v_buf = torch.empty_like(k_buf)

        num_pages = (size + page_size - 1) // page_size
        # Production initializes metadata to (+inf, -inf). Mirror that here.
        k_metadata = torch.empty(num_pages, H, D, 2, device=device, dtype=torch.float32)
        k_metadata[..., 0].fill_(float("inf"))   # min
        k_metadata[..., 1].fill_(float("-inf"))  # max

        quest_update_kv_and_metadata(
            k_new, v_new, k_buf, v_buf, k_metadata, out_cache_loc, page_size
        )

        # KV cache is updated exactly at out_cache_loc
        self.assertTrue(torch.allclose(k_buf[out_cache_loc.long()], k_new))
        self.assertTrue(torch.allclose(v_buf[out_cache_loc.long()], v_new))

        # Since old metadata is effectively +/-inf and each page receives exactly one token here,
        # new min/max equals that token for its page
        for b in range(B):
            page_idx = int(out_cache_loc[b].item()) // page_size
            self.assertTrue(
                torch.allclose(k_metadata[page_idx, :, :, 0], k_new[b].to(torch.float32))
            )
            self.assertTrue(
                torch.allclose(k_metadata[page_idx, :, :, 1], k_new[b].to(torch.float32))
            )

    def test_estimate_excludes_last_page(self):
        device = "cuda"
        B, H, D = 2, 2, 16
        page_size = 4
        seq_lens = torch.tensor([7, 8], dtype=torch.int32, device=device)
        max_pages = int(((seq_lens.max().item()) + page_size - 1) // page_size)

        q = torch.randn(B, H, D, device=device, dtype=torch.float32)

        # Fake metadata
        num_pages_global = max_pages * 2
        k_meta = torch.randn(num_pages_global, H, D, 2, device=device, dtype=torch.float32)

        estimated_scores = torch.zeros(B, H, max_pages, device=device, dtype=torch.float32)
        # Identity req_to_token mapping for simplicity
        max_ctx = max_pages * page_size
        req_to_token = torch.arange(max_ctx, device=device, dtype=torch.int32).repeat(B, 1)

        quest_estimate_scores(
            q,
            k_meta,
            seq_lens,
            estimated_scores,
            page_size,
            req_to_token,
        )

        for b in range(B):
            num_pages = int(((int(seq_lens[b].item())) + page_size - 1) // page_size)
            # last page index = num_pages - 1 should remain 0 (not written)
            self.assertTrue(torch.all(estimated_scores[b, :, num_pages - 1] == 0))

    def test_select_expand_pack_into(self):
        device = "cuda"
        B, H, D = 2, 3, 16
        page_size = 4
        quest_topk = 3  # total = (topk-1) + last
        # Two sequences with different lens (ensure partial last page)
        seq_lens = torch.tensor([7, 10], dtype=torch.int32, device=device)
        max_pages = int(((seq_lens.max().item()) + page_size - 1) // page_size)

        # random estimated scores
        estimated_scores = torch.randn(B, H, max_pages, device=device, dtype=torch.float32)
        max_ctx = max_pages * page_size
        req_to_token = torch.arange(max_ctx, device=device, dtype=torch.int32).repeat(B, 1)

        selected_pages = torch.empty(B, H, quest_topk, device=device, dtype=torch.int32)
        tokens_cap = quest_topk * page_size
        kv_indices_buf = torch.empty(B * H, tokens_cap, device=device, dtype=torch.int32)
        tokens_per_head = torch.zeros(B * H, device=device, dtype=torch.int32)
        tokens_per_batch = torch.zeros(B, device=device, dtype=torch.int32)
        kv_indptr = torch.zeros(B + 1, device=device, dtype=torch.int32)
        kv_indices = torch.empty(B * H * tokens_cap, device=device, dtype=torch.int32)

        quest_select_topk_pages_into(
            estimated_scores,
            seq_lens,
            req_to_token,
            quest_topk,
            page_size,
            selected_pages,
            kv_indices_buf,
            tokens_per_head,
            tokens_per_batch,
            kv_indptr,
            kv_indices,
            H,
        )

        # Validate kv_indptr consistency
        self.assertEqual(int(kv_indptr[0].item()), 0)
        self.assertEqual(int(kv_indptr[-1].item()), int(tokens_per_batch.sum().item()))

        # Validate tokens_per_batch equals min over heads of per-head token counts
        for b in range(B):
            # Compute expected per head token counts from selected pages
            per_head = []
            for h in range(H):
                tok = 0
                for i in range(quest_topk):
                    p = int(selected_pages[b, h, i].item())
                    if p < 0:
                        continue
                    # Page length: full except possibly last page
                    num_pages = int(((int(seq_lens[b].item())) + page_size - 1) // page_size)
                    start = p * page_size
                    end = min(start + page_size, int(seq_lens[b].item()))
                    tok += max(0, end - start)
                per_head.append(tok)
            self.assertEqual(int(tokens_per_batch[b].item()), min(per_head))


if __name__ == "__main__":
    unittest.main()
