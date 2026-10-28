"""NPU e2e test for --enable-watermark / --watermark-config.

Covers the NPU-specific device dispatch (host-level is_cuda gating routes the
watermark force/append path to the torch reference implementation instead of
the CUDA Triton selector) and verifies end to end that generated text is
detectable by the watermark detector.

Trimmed mirror of test/registered/e2e/openai_server/test_watermark.py, which
owns the server-config matrix (default_enabled, enforce_all, structured
output) and stays CUDA-only. Keep the request payload and detector thresholds
in sync with that file.
"""

import json
import os
import sys
import tempfile

import pytest
import requests

from sglang.srt.sampling.watermarking import WatermarkDetector
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

# NPU-only: local Llama-3.2 mirror, Ascend attention backend, and the torch
# reference selector (host-level is_cuda dispatch) instead of the Triton one.
register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)

_MODEL = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
_KEY_A = "0123456789abcdef"
_KEY_B = "fedcba9876543210"
_OMITTED = object()


def _watermark_statistics(choice, key):
    return (
        WatermarkDetector(key)
        .detect_tokens(
            choice["response_token_ids"],
            prompt_token_ids=choice["prompt_token_ids"],
        )
        .combined
    )


def _chat_payload(watermark=_OMITTED, *, max_tokens):
    payload = {
        "model": _MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a long, varied field guide to urban trees. Include many "
                    "species, observations, and practical examples."
                ),
            }
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "return_token_ids": True,
    }
    if watermark is not _OMITTED:
        payload["watermark"] = watermark
    return payload


def _assert_detected(response, key, *, other_key=None):
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    statistics = _watermark_statistics(choice, key)
    assert statistics.num_contexts >= 100
    assert statistics.z_score >= 5.0
    if other_key is not None:
        other = _watermark_statistics(choice, other_key)
        assert statistics.z_score - other.z_score >= 4.0


class TestNpuWatermarkEndpoint(CustomTestCase):
    """Testcase: verify --enable-watermark produces NPU-generated text that the
    watermark detector attributes to the server/request key.

    [Test Category] Parameter
    [Test Target] --enable-watermark, --watermark-config, per-request watermark
    """

    process = None
    config_file = None

    @classmethod
    def setUpClass(cls):
        cls.config_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump({"key": _KEY_A, "context_window": 4}, cls.config_file)
        cls.config_file.close()
        os.chmod(cls.config_file.name, 0o600)
        cls.process = popen_launch_server(
            _MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--enable-watermark",
                "--watermark-config",
                cls.config_file.name,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)
        if cls.config_file is not None:
            os.unlink(cls.config_file.name)

    def test_per_request_watermark_is_detectable(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
            json=_chat_payload({"enabled": True}, max_tokens=512),
            timeout=300,
        )
        _assert_detected(response, _KEY_A, other_key=_KEY_B)

    def test_per_request_key_override_is_detectable(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
            json=_chat_payload({"key": _KEY_B}, max_tokens=512),
            timeout=300,
        )
        _assert_detected(response, _KEY_B, other_key=_KEY_A)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
