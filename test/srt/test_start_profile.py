"""
Usage:
python3 -m unittest test_srt_engine.TestSRTEngine.test_4_sync_async_stream_combination
"""

import os
import shutil
import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DEFAULT_SMALL_MODEL_NAME_FOR_TEST_NPU = (
    "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
)

OUTPUT_DIR = "./profiler_dir"


class TestStartProfile(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        envs.SGLANG_TORCH_PROFILER_DIR.set(OUTPUT_DIR)
        cls.model = (
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST
            if not is_npu()
            else DEFAULT_SMALL_MODEL_NAME_FOR_TEST_NPU
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = (
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

    def setUp(self):
        self._clear_profile_dir()

    def test_start_profile_1(self):
        """Test /start_profile with start_step and num_steps argument. This have to be the first test for start_step to work"""
        response = self._start_profile(start_step="15", num_steps=5)

        self._post_request()

        self._check_empty_profile_dir()

    def test_start_profile_2(self):
        """Test /start_profile with no argument"""
        self._clear_profile_dir()
        # Before /stop_profile, the profile directory should be empty
        self._check_empty_profile_dir()
        response = self._start_profile()
        self._post_request()
        # Post /stop_profile and check the profile directory is non-empty
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/stop_profile",
        )

    def test_start_profile_3(self):
        """Test /start_profile with num_steps argument"""
        response = self._start_profile(num_steps=5)
        self._post_request()
        self._check_empty_profile_dir()

    def _start_profile(self, **kwargs):
        """Start profiling with optional parameters."""
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/start_profile",
            json=kwargs if kwargs else None,
        )
        self.assertEqual(response.status_code, 200)

    def _post_request(self):
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

    def _clear_profile_dir(self):
        if os.path.isdir(OUTPUT_DIR):
            # Remove the directory and all its contents
            shutil.rmtree(OUTPUT_DIR)

    def _check_non_empty_profile_dir(self):
        self.assertTrue(os.path.isdir(OUTPUT_DIR), "Output directory does not exist.")
        self.assertNotEqual(
            len(os.listdir(OUTPUT_DIR)), 0, "Output directory is empty!"
        )

    def _check_empty_profile_dir(self):
        if os.path.isdir(OUTPUT_DIR):
            self.assertEqual(
                len(os.listdir(OUTPUT_DIR)), 0, "Output directory is non-empty!"
            )


if __name__ == "__main__":
    unittest.main()
