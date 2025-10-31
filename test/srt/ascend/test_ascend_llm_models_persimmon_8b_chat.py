from test_ascend_llm_models_2 import TestLlama_2_7B


class TestPersimmon_8b_chat(TestLlama_2_7B):
    model = "/root/.cache/modelscope/hub/models/Howeee/persimmon-8b-chat"
    accuracy = 0.05
