"""
Test --enable-dynamic-chunking on NPU with pipeline parallelism.

The flag enables dynamic chunk-size adjustment for pipeline parallelism
(pp_size > 1): at startup the scheduler profiles synthetic prefills and fits
a quadratic latency model, then sizes every chunk of an in-flight chunked
prefill so each forward takes about as long as a first static-size chunk.
Without pp_size > 1 the flag is a hard no-op — the chunk sizer is only
created for pipeline parallelism — so this test runs at --pp-size 2.

Scope (PP=2 functional coverage): the server accepts the flag, the startup
profiling + fit actually succeed, and a long prompt still completes across
multiple chunks. The "actually succeed" part matters because every sizer
failure path (profiling error, non-positive quadratic coefficient, failed
broadcast) degrades *silently* to static chunk sizes with only a warning —
the "[PP Dynamic Chunk] ... Predictor ready" log line is the only observable
proof the feature is live, so this test captures the server log and asserts
on it. Per-chunk execution-time consistency itself is a performance
observation and belongs to the perf suites, not to CI assertions.

The chunk size is 1024 (not smaller) on purpose: the profiler samples
latencies up to 1.25 x chunked_prefill_size, and at tiny chunk sizes the
attention term of the latency model is inside the fit noise, so the fitted
quadratic coefficient is not guaranteed positive and the sizer would
legitimately disable itself — a flaky assertion, not a product bug.

[Test Category] Parameter
[Test Target] --enable-dynamic-chunking
"""

import os
import tempfile
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="full-2-npu-a3", nightly=True)


class TestNpuEnableDynamicChunking(CustomTestCase):
    """Verify --enable-dynamic-chunking is live at PP=2 and does not break
    multi-chunk prefill serving.

    [Test Category] Parameter
    [Test Target] --enable-dynamic-chunking
    """

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    chunked_prefill_size = 1024

    @classmethod
    def setUpClass(cls):
        # Private mkstemp log pair instead of test_utils' shared fixed names
        # (/tmp/stdout.txt, /tmp/stderr.txt), so concurrent manual runs of
        # this case next to others cannot clobber each other's captured log.
        out_fd, cls.stdout_path = tempfile.mkstemp(suffix=".out.log")
        err_fd, cls.stderr_path = tempfile.mkstemp(suffix=".err.log")
        os.close(out_fd)
        os.close(err_fd)
        cls.stdout = open(cls.stdout_path, "w+", encoding="utf-8")
        cls.stderr = open(cls.stderr_path, "w+", encoding="utf-8")
        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--disable-cuda-graph",
                    "--attention-backend",
                    "ascend",
                    "--tp-size",
                    "2",
                    "--pp-size",
                    "2",
                    "--enable-dynamic-chunking",
                    # Large enough that the profiling range reaches the
                    # quadratic regime (see module docstring).
                    "--chunked-prefill-size",
                    str(cls.chunked_prefill_size),
                ],
                return_stdout_stderr=(cls.stdout, cls.stderr),
            )
        except Exception:
            cls.stdout.close()
            cls.stderr.close()
            os.remove(cls.stdout_path)
            os.remove(cls.stderr_path)
            raise

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        if hasattr(cls, "stdout") and cls.stdout:
            cls.stdout.close()
        if hasattr(cls, "stderr") and cls.stderr:
            cls.stderr.close()
        for path in (
            getattr(cls, "stdout_path", None),
            getattr(cls, "stderr_path", None),
        ):
            if path and os.path.exists(path):
                os.remove(path)

    def _server_log(self) -> str:
        self.stdout.flush()
        self.stderr.flush()
        self.stdout.seek(0)
        self.stderr.seek(0)
        return self.stdout.read() + self.stderr.read()

    def test_flag_reported(self):
        info = requests.get(f"{self.base_url}/server_info", timeout=30).json()
        self.assertTrue(info["enable_dynamic_chunking"])

    def test_predictor_ready(self):
        """The sizer profiled and fitted successfully — not silently
        degraded to static chunking."""
        log = self._server_log()
        ready_count = log.count("Predictor ready (quadratic)")
        self.assertGreaterEqual(
            ready_count,
            2,
            f"Expected both PP ranks to report a ready predictor "
            f"(found {ready_count} 'Predictor ready (quadratic)' lines): "
            f"at pp_size 2 every rank profiles the same broadcast samples "
            f"and fits them, so a missing line means that rank never "
            f"created/fitted its sizer or silently degraded",
        )
        self.assertNotIn("Dynamic chunking will be disabled", log)

    def test_long_prompt_chunked_generate(self):
        """A prompt of several chunks completes under the dynamic sizer."""
        long_text = "The history of France is long and rich. " * 400
        gen_resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": long_text,
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=120,
        )
        self.assertEqual(gen_resp.status_code, 200)
        data = gen_resp.json()
        # Guard the multi-chunk precondition: if the tokenizer ever gets
        # more efficient, this test must fail loudly instead of silently
        # exercising a single-chunk prefill.
        self.assertGreater(
            data["meta_info"]["prompt_tokens"], 2 * self.chunked_prefill_size
        )
        self.assertGreater(data["meta_info"]["completion_tokens"], 0)

        self.assertIsNone(self.process.poll(), "Server crashed during test")


if __name__ == "__main__":
    unittest.main()
