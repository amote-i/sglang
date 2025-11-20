import unittest

import requests

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestL2Cache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if is_npu():
            cls.model = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-32B"
        else:
            cls.model = "Qwen/Qwen3-32B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        common_args = [
            "--enable-hierarchical-cache",
            "--mem-fraction-static",
            0.8,
            "--tp-size",
            2,
        ]
        other_args = common_args + (
            [
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ]
            if is_npu()
            else []
        )

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_L2_cache_01(self):
        for i in range(2):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?What is The capital of France?",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 10,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            if i == 0:
                self.assertEqual(int(response.json()["meta_info"]["cached_tokens"]), 0)
            else:
                self.assertGreater(
                    int(response.json()["meta_info"]["cached_tokens"]), 0
                )


if __name__ == "__main__":
    unittest.main()
