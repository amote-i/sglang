import unittest

from test_ascend_llm_models_2 import TestLlama_2_7B


class TestStablelm_2_1_6b(TestLlama_2_7B):
    model = "/root/.cache/modelscope/hub/models/Xenova/stablelm-2-1_6b"
    accuracy = 0.6


if __name__ == "__main__":
    unittest.main()
