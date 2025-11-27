"""
Usage:
python3 -m unittest test_quest_backend.TestQuestAttnBackend.test_mmlu
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MHA_MODEL_NAME_FOR_TEST,
    DEFAULT_MHA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestQuestAttnBackend(CustomTestCase):
    def test_mmlu(self):
        # model = DEFAULT_MODEL_NAME_FOR_TEST
        # model = DEFAULT_SMALL_MHA_MODEL_NAME_FOR_TEST
        model = DEFAULT_MHA_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "triton",
                "--page-size",
                16,
                "--enable-quest",
                "--quest-topk",
                16,
                "--mem-fraction-static",
                0.5,
                "--random-seed",
                "1107",
            ],
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_eval(args)
            # Keep threshold conservative to avoid flakiness across environments
            self.assertGreaterEqual(metrics["score"], 0.50)
        finally:
            kill_process_tree(process.pid)


class TestQuestGqaAttnBackend(CustomTestCase):
    def test_mmlu_gqa(self):
        model = "Qwen/Qwen3-4B-Instruct-2507"
        # model = DEFAULT_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "triton",
                "--page-size",
                16,
                "--enable-quest",
                "--quest-topk",
                16,
                "--mem-fraction-static",
                0.5,
                "--random-seed",
                "1107",
                "--context-length",
                4096,
                "--quest-estimate-kernel",
                "triton",
                "--quest-topk-kernel",
                "max",
            ],
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_eval(args)
            # Keep threshold conservative to avoid flakiness across environments
            self.assertGreaterEqual(metrics["score"], 0.50)
        finally:
            kill_process_tree(process.pid)

    def test_mmlu_gqa_estimate_torch(self):
        model = "Qwen/Qwen3-4B-Instruct-2507"
        # model = DEFAULT_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "triton",
                "--page-size",
                16,
                "--enable-quest",
                "--quest-topk",
                16,
                "--mem-fraction-static",
                0.5,
                "--random-seed",
                "1107",
                "--context-length",
                4096,
                "--quest-estimate-kernel",
                "torch",
                "--quest-topk-kernel",
                "max",
            ],
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_eval(args)
            # Keep threshold conservative to avoid flakiness across environments
            self.assertGreaterEqual(metrics["score"], 0.50)
        finally:
            kill_process_tree(process.pid)

    def test_mmlu_gqa_topk_sort(self):
        model = "Qwen/Qwen3-4B-Instruct-2507"
        # model = DEFAULT_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "triton",
                "--page-size",
                16,
                "--enable-quest",
                "--quest-topk",
                16,
                "--mem-fraction-static",
                0.5,
                "--random-seed",
                "1107",
                "--context-length",
                4096,
                "--quest-estimate-kernel",
                "triton",
                "--quest-topk-kernel",
                "sort",
            ],
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_eval(args)
            # Keep threshold conservative to avoid flakiness across environments
            self.assertGreaterEqual(metrics["score"], 0.50)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()

