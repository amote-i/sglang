import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestC4AI(CustomTestCase):
    model = "/root/.cache/modelscope/hub/models/CohereForAI/c4ai-command-r-v01"
    accuracy = 0.05

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        chat_template_path = (
            "../../examples/chat_template/tool_chat_template_c4ai_command_r_v01.jinja"
        )
        if not os.path.exists(chat_template_path) or not os.path.isfile(
            chat_template_path
        ):
            chat_template_path = "/__w/sglang/sglang/test/srt/ascend/examples/chat_template/tool_chat_template_c4ai_command_r_v01.jinja"

        other_args = (
            [
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.8",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--chat-template",
                chat_template_path,
                "--tp-size",
                "2",
            ]
            if is_npu()
            else []
        )
        if is_npu():
            env = os.environ.copy()
            env.update(
                {
                    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24666",
                    "HCCL_BUFFSIZE": "200",
                    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "24",
                    "USE_VLLM_CUSTOM_ALLREDUCE": "1",
                    "HCCL_EXEC_TIMEOUT": "200",
                    "STREAMS_PER_DEVICE": "32",
                    "SGLANG_ENABLE_TORCH_COMPILE": "1",
                }
            )
        else:
            env = None

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        self.assertGreater(
            metrics["accuracy"],
            self.accuracy,
            f'Accuracy of {self.model} is {str(metrics["accuracy"])}, is lower than {self.accuracy}',
        )


# class TestERNIE_4_5_21B_A3B_PT(TestC4AI):
#     model = "/root/.cache/modelscope/hub/models/PaddlePaddle/ERNIE-4.5-21B-A3B-PT"

#     @classmethod
#     def setUpClass(cls):
#         cls.base_url = DEFAULT_URL_FOR_TEST
#         other_args = (
#             [
#                 "--trust-remote-code",
#                 "--mem-fraction-static",
#                 "0.8",
#                 "--attention-backend",
#                 "ascend",
#                 "--disable-cuda-graph",
#                 "--tp-size",
#                 "2",
#             ]
#             if is_npu()
#             else []
#         )
#         if is_npu():
#             env = os.environ.copy()
#             env.update(
#                 {
#                     "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
#                     "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24666",
#                     "HCCL_BUFFSIZE": "200",
#                     "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "24",
#                     "USE_VLLM_CUSTOM_ALLREDUCE": "1",
#                     "HCCL_EXEC_TIMEOUT": "200",
#                     "STREAMS_PER_DEVICE": "32",
#                 }
#             )
#         else:
#             env = None

#         cls.process = popen_launch_server(
#             cls.model,
#             cls.base_url,
#             timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
#             other_args=other_args,
#             env=env,
#         )


if __name__ == "__main__":
    unittest.main()
