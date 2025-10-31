import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    popen_launch_server,
)


class TestDPAttentionMinimumTokenLoadBalance(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if is_npu():
            cls.model = (
                "/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-R1-0528-W8A8"
            )
        else:
            cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                [
                    "--trust-remote-code",
                    "--tp",
                    "16",
                    "--enable-dp-attention",
                    "--dp",
                    "1",
                    "--enable-torch-compile",
                    "--torch-compile-max-bs",
                    "2",
                    "--load-balance-method",
                    "minimum_tokens",
                    "--attention-backend",
                    "ascend",
                    "--disable-cuda-graph",
                    "--quantization",
                    "w8a8_int8",
                    "--mem-fraction-static",
                    "0.75",
                ]
                if is_npu()
                else [
                    "--trust-remote-code",
                    "--tp",
                    "2",
                    "--enable-dp-attention",
                    "--dp",
                    "2",
                    "--enable-torch-compile",
                    "--torch-compile-max-bs",
                    "2",
                    "--load-balance-method",
                    "minimum_tokens",
                    "--quantization",
                    "w8a8_int8",
                    "--mem-fraction-static",
                    "0.75",
                ]
            ),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mgsm_en(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=10,
            num_threads=1024,
        )

        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.5)


if __name__ == "__main__":
    unittest.main()
