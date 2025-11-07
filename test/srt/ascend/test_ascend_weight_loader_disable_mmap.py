import os
import time
import unittest

import requests

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestWeightLoaderDisableMmap(CustomTestCase):
    def test_weight_loader_mmap(self):
        other_args = [
            (
                [
                    "--weight-loader-disable-mmap",
                    "--attention-backend",
                    "ascend",
                    "--disable-cuda-graph",
                    "--mem-fraction-static",
                    0.9,
                ]
                if is_npu()
                else ["----weight-loader-disable-mmap"]
            ),
            (
                [
                    "--attention-backend",
                    "ascend",
                    "--disable-cuda-graph",
                    "--mem-fraction-static",
                    0.9,
                ]
                if is_npu()
                else []
            ),
        ]
        start_succes_time_list = []
        for i in other_args:
            start_time = time.perf_counter()
            process = popen_launch_server(
                (
                    "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-11B-Vision-Instruct"
                    if is_npu()
                    else DEFAULT_SMALL_MODEL_NAME_FOR_TEST
                ),
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=i,
            )
            start_succes_time = time.perf_counter() - start_time
            start_succes_time_list.append(start_succes_time)
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
            kill_process_tree(process.pid)
        print("-----------start_succes_time_list is", start_succes_time_list)
        self.assertTrue(start_succes_time_list[0] > start_succes_time_list[1])


if __name__ == "__main__":
    unittest.main()
