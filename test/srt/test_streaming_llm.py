import os
import shlex
import unittest
from types import SimpleNamespace

import torch
import requests

from sglang.srt.utils import kill_process_tree
from sglang.srt.layers.attention.streamingllm_utils import update_streamingllm_buffer
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.run_eval import run_eval


def _cuda_only():
    return torch.cuda.is_available()


class TestStreamingLLM_A_Unit(unittest.TestCase):
    @unittest.skipUnless(_cuda_only(), "CUDA is required for this test")
    def test_update_streamingllm_buffer_basic(self):
        print("[StreamingLLM][Unit] test_update_streamingllm_buffer_basic: start", flush=True)
        # Build a tiny req_to_token table on CUDA
        device = "cuda"
        bs = 2
        max_context_len = 16
        req_pool_size = 8
        req_to_token = torch.full(
            (req_pool_size, max_context_len), -1, dtype=torch.int32, device=device
        )
        # Two sequences: map to distinctive indices for easy assertion
        # Row 0: indices 0.., Row 1: indices 1000..
        for s in range(max_context_len):
            req_to_token[0, s] = s
            req_to_token[1, s] = 1000 + s

        seq_lens = torch.tensor([10, 5], dtype=torch.int32, device=device)
        req_pool_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)

        S, W = 2, 3
        kv_indptr, kv_indices, kv_lens = update_streamingllm_buffer(
            req_to_token=req_to_token,
            sink_tokens=S,
            window_tokens=W,
            seq_lens=seq_lens,
            req_pool_indices=req_pool_indices,
            bs=bs,
            device=device,
        )

        # Expected per-seq union: first S + last W without overlap
        # seq0 (L=10): [0,1] + [7,8,9]
        # seq1 (L=5):  [1000,1001] + [1002,1003,1004]
        self.assertTrue(torch.equal(kv_indptr, torch.tensor([0, 5, 10], device=device)))
        self.assertTrue(torch.equal(kv_lens, torch.tensor([5, 5], device=device)))
        expected = torch.tensor(
            [0, 1, 7, 8, 9, 1000, 1001, 1002, 1003, 1004], dtype=torch.int32, device=device
        )
        self.assertTrue(torch.equal(kv_indices, expected))
        print("[StreamingLLM][Unit] test_update_streamingllm_buffer_basic: ok", flush=True)


class _DummyTokenToKVPool:
    def __init__(self, size, kv_heads, head_dim, device):
        self.k = torch.zeros((size + 1, kv_heads, head_dim), dtype=torch.bfloat16, device=device)
        self.v = torch.zeros((size + 1, kv_heads, head_dim), dtype=torch.bfloat16, device=device)

    def get_value_buffer(self, layer_id: int):
        return self.v

    def get_key_buffer(self, layer_id: int):
        return self.k


class _DummyModelConfig:
    def __init__(self, num_heads=8, kv_heads=8, head_dim=64, context_len=2048):
        self.num_attention_heads = num_heads
        self._kv_heads = kv_heads
        self.head_dim = head_dim
        self.context_len = context_len
        self.is_encoder_decoder = False

    def get_num_kv_heads(self, tp_size: int):
        return self._kv_heads


class _DummyReqToTokenPool:
    def __init__(self, size, max_context_len, device):
        self.size = size
        self.req_to_token = torch.full(
            (size, max_context_len), -1, dtype=torch.int32, device=device
        )


class _DummyServerArgs:
    def __init__(self):
        self.triton_attention_num_kv_splits = 8
        self.enable_streaming_llm = True
        self.streaming_llm_window_length = 3
        self.streaming_llm_num_sink_tokens = 2
        # fields used by TritonAttnBackend init path
        self.speculative_num_draft_tokens = 1
        self.speculative_num_steps = 1


class _DummyModelRunner:
    def __init__(self, device="cuda"):
        self.device = device
        self.gpu_id = 0
        self.server_args = _DummyServerArgs()
        self.model_config = _DummyModelConfig()
        self.req_to_token_pool = _DummyReqToTokenPool(size=4, max_context_len=16, device=device)
        self.sliding_window_size = None  # disable built-in SW
        kv_heads = self.model_config.get_num_kv_heads(1)
        self.token_to_kv_pool = _DummyTokenToKVPool(size=64, kv_heads=kv_heads, head_dim=self.model_config.head_dim, device=device)


class TestStreamingLLM_B_Integration(unittest.TestCase):
    @unittest.skipUnless(_cuda_only(), "CUDA is required for this test")
    def test_backend_metadata_streaming_indices(self):
        print("[StreamingLLM][Integration] test_backend_metadata_streaming_indices: start", flush=True)
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
        # Ensure dp_attention is initialized for tests (TP size = 1)
        try:
            from sglang.srt.layers import dp_attention as _dp
            if getattr(_dp, "_ATTN_TP_SIZE", None) is None:
                _dp._ATTN_TP_SIZE = 1
        except Exception:
            pass
        device = "cuda"
        runner = _DummyModelRunner(device=device)

        # Prepare a tiny mapping for two reqs
        for s in range(16):
            runner.req_to_token_pool.req_to_token[0, s] = s
            runner.req_to_token_pool.req_to_token[1, s] = 1000 + s

        attn_backend = TritonAttnBackend(runner)

        # Build a minimal ForwardBatch in DECODE mode
        seq_lens = torch.tensor([10, 5], dtype=torch.int32, device=device)
        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=2,
            input_ids=torch.zeros((2,), dtype=torch.long, device=device),
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32, device=device),
            seq_lens=seq_lens,
            out_cache_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
            seq_lens_sum=int(seq_lens.sum().item()),
        )

        attn_backend.init_forward_metadata(fb)
        fm = attn_backend.forward_metadata

        # Verify StreamingLLM metadata exists and matches expectations
        self.assertIsNotNone(fm.streaming_kv_indptr)
        self.assertIsNotNone(fm.streaming_kv_indices)
        self.assertTrue(torch.equal(fm.streaming_kv_indptr, torch.tensor([0, 5, 10], device=device)))
        expected = torch.tensor([0, 1, 7, 8, 9, 1000, 1001, 1002, 1003, 1004], dtype=torch.int32, device=device)
        self.assertTrue(torch.equal(fm.streaming_kv_indices, expected))
        print("[StreamingLLM][Integration] test_backend_metadata_streaming_indices: ok", flush=True)


@unittest.skip("Temporarily disable StreamingLLM E2E smoke test")
class TestStreamingLLME2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _cuda_only():
            raise unittest.SkipTest("CUDA is required for this test")
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        print("[StreamingLLM][E2E] Launching server...", flush=True)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-streaming-llm",
                "--streaming-llm-window-length",
                "256",
                "--streaming-llm-num-sink-tokens",
                "4",
                "--attention-backend",
                "triton",
                "--max-total-tokens",
                "200000",
            ],
        )
        print("[StreamingLLM][E2E] Server launched.", flush=True)

    @classmethod
    def tearDownClass(cls):
        print("[StreamingLLM][E2E] Shutting down server...", flush=True)
        kill_process_tree(cls.process.pid)
        print("[StreamingLLM][E2E] Server terminated.", flush=True)

    def test_smoke_generate(self):
        print("[StreamingLLM][E2E] test_smoke_generate: start", flush=True)
        # Simple /generate smoke call to avoid dataset downloads
        payload = {
            "text": "System: You are a helpful assistant.\nUser: Say hello in one word.\nAssistant:",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 8,
            },
        }
        resp = requests.post(f"{self.base_url}/generate", json=payload, timeout=60)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("text", data)
        print(f"[StreamingLLM][E2E] /generate ok. Output: {data['text'][:120]}", flush=True)
        print("[StreamingLLM][E2E] test_smoke_generate: ok", flush=True)


class TestStreamingLlmAcc(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        # Allow overriding server args from environment for flexible CLI usage
        # Full override: set SGL_TEST_SERVER_OTHER_ARGS to a shell-style string
        env_other_args = os.getenv("SGL_TEST_SERVER_OTHER_ARGS")
        if env_other_args:
            other_args = shlex.split(env_other_args)
        else:
            # Fine-grained overrides via individual env vars
            window_len = os.getenv("SGL_TEST_STREAMING_WINDOW_LENGTH", "1024")
            sink_tokens = os.getenv("SGL_TEST_STREAMING_SINK_TOKENS", "4")
            attention_backend = os.getenv("SGL_TEST_ATTENTION_BACKEND", "triton")
            max_total_tokens = os.getenv("SGL_TEST_MAX_TOTAL_TOKENS", "200000")
            disable_cuda_graph_env = os.getenv("SGL_TEST_DISABLE_CUDA_GRAPH", "0")

            other_args = [
                "--enable-streaming-llm",
                "--streaming-llm-window-length",
                str(window_len),
                "--streaming-llm-num-sink-tokens",
                str(sink_tokens),
                "--attention-backend",
                str(attention_backend),
                "--max-total-tokens",
                str(max_total_tokens),
                "--random-seed",
                "1107",
            ]
            print(f"[StreamingLLM][Acc] other_args: {other_args}", flush=True)
            # Include or exclude --disable-cuda-graph based on env flag (default: include)
            if disable_cuda_graph_env.lower() in {"1", "true", "yes", "y"}:
                other_args.append("--disable-cuda-graph")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.65)


if __name__ == "__main__":
    unittest.main()
