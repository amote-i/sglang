import unittest

import requests

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestAscendMMAttentionBackend(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = (
            "/root/.cache/modelscope/hub/models/LLM-Research/Llama-4-Scout-17B-16E-Instruct"
            if is_npu()
            else "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        )
        other_args = (
            [
                "--chat-template",
                "llama-4",
                "--mem-fraction-static",
                "0.8",
                "--tp-size=8",
                "--context-length=8192",
                "--mm-attention-backend",
                "fa3",
                "--cuda-graph-max-bs",
                "4",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ]
            if is_npu()
            else [
                "--chat-template",
                "llama-4",
                "--mem-fraction-static",
                "0.8",
                "--tp-size=8",
                "--context-length=8192",
                "--mm-attention-backend",
                "fa3",
                "--cuda-graph-max-bs",
                "4",
            ]
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mm_attention_backend(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)


if __name__ == "__main__":
    unittest.main()
