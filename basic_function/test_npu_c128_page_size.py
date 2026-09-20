"""
Test --c128-page-size on NPU with a real DeepSeek-V4 (C128 sparse attention)
model: DeepSeek-V4-Flash at tp=4 x dp=4 (fp8), the smallest proven recipe for
this model family (mirrors the nightly-acc-4-npu-a5 gpqa case, minus the
benchmark-only knobs).

The parameter sets the physical page size of the NPU DSV4 C128 KV cache
(the page granularity of the sparse-attention operator and of the
req→C128-sidecar mapping). It is consumed only on the DSV4 path
(`--attention-backend dsv4` + a DeepSeek-V4-family model), so a dense model
with the plain ascend backend can observe nothing but the echo.

Scope (functional coverage on the live path):

  - the multiple-of-16 constraint is enforced by the DSV4 memory pool itself
    (dsv4_memory_pool), so --c128-page-size 24 must abort startup with the
    pool's own error — an assertion that only fires when the C128 pool is
    actually being built;
  - a non-default page width (32 → C128 groups of 128*32 = 4096 tokens)
    serves a ~12k-token needle prompt across chunked prefill boundaries
    with the right answer, i.e. the paged C128 layout the flag selects is
    correct end to end;
  - the DSV4 pool configurator's own startup log ("DSV4 SWA sizing:")
    proves the pool under test is the DSV4 one.

[Test Category] Parameter
[Test Target] --c128-page-size
"""

import os
import tempfile
import time
import unittest

import requests

from sglang.test.ascend.test_ascend_utils import MODEL_WEIGHTS_DIR
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_npu_ci(est_time=3600, suite="nightly-acc-4-npu-a5", nightly=True)

DEEPSEEK_V4_FLASH_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "deepseek-ai/DeepSeek-V4-Flash"
)

TP_SIZE = 4
DP_SIZE = 4
C128_PAGE_SIZE = 32
CHUNKED_PREFILL_SIZE = 8192

# The DSV4 pool's own init-time constraint (dsv4_memory_pool), asserted
# verbatim by the validation case.
C128_POOL_VALIDATION_ERROR = "c128_page_size must be a positive multiple of 16"

# Launch recipe mirrored from the proven nightly-acc-4-npu-a5 case
# (test_deepseek_v4_flash_fp8_4p_gpqa_a5), minus benchmark-only knobs
# (EAGLE, reasoning parser, giant prefill budgets, cuda-graph bs list).
# The env block is part of what makes this config boot on A5.
# popen_launch_server appends --device itself (auto_config_device).
DSV4_4P_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "TASK_QUEUE_ENABLE": "1",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    # HCCL deepep
    "HCCL_BUFFSIZE": "1024",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "16",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
    # dsv4
    "IS_DEEPSEEK_V4": "1",
    "USE_FUSED_HC_PRE_ASCENDC": "1",
    "SGLANG_DSV4_NPU_FUSED_COMPRESSOR": "1",
    "SGLANG_DSV4_NPU_FUSED_COMPRESSOR_PREFILL": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "True",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    "SGLANG_OPT_FP8_WO_A_GEMM": "False",
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
}

DSV4_4P_OTHER_ARGS = [
    "--page-size",
    "128",
    "--tp-size",
    str(TP_SIZE),
    "--trust-remote-code",
    "--attention-backend",
    "dsv4",
    "--watchdog-timeout",
    "9000",
    "--mem-fraction-static",
    "0.72",
    "--max-running-requests",
    "8",
    "--chunked-prefill-size",
    str(CHUNKED_PREFILL_SIZE),
    "--max-prefill-tokens",
    "16384",
    "--enable-dp-lm-head",
    "--disable-radix-cache",
    "--enable-dp-attention",
    "--dp-size",
    str(DP_SIZE),
    "--quantization",
    "fp8",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
]

NEEDLE = "XQ7-4421"
_FILLER = (
    "The survey team logged the depth of every borehole along the northern "
    "ridge and filed the readings with the regional archive. "
)
# ~4.8k tokens per half: the needle prompt spans several C128 groups
# (128 * 32 = 4096 tokens each) and multiple prefill chunks.
_NEEDLE_REPEATS = 200


def _needle_prompt() -> str:
    body = _FILLER * _NEEDLE_REPEATS
    return (
        f"{body}The vault code word is {NEEDLE}. {body}"
        "Question: What is the vault code word?\nAnswer:"
    )


def _wait_for_log(logs, needle, timeout=60):
    """Poll already-open capture files until `needle` shows up.

    popen_launch_server's pump threads flush the server output line by line,
    so give them a bounded window instead of asserting immediately.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        for log in logs:
            log.seek(0)
            if needle in log.read():
                return True
        time.sleep(2)
    return False


class TestNpuC128PageSizeFunctional(CustomTestCase):
    """Verify a non-default --c128-page-size is plumbed into the DSV4 pool
    and serves correctly.

    [Test Category] Parameter
    [Test Target] --c128-page-size
    """

    model = DEEPSEEK_V4_FLASH_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    server_launch_timeout = 2400

    @classmethod
    def setUpClass(cls):
        out_fd, cls.out_path = tempfile.mkstemp(suffix=".out.log")
        err_fd, cls.err_path = tempfile.mkstemp(suffix=".err.log")
        os.close(out_fd)
        os.close(err_fd)
        cls.out_log = open(cls.out_path, "w+", encoding="utf-8")
        cls.err_log = open(cls.err_path, "w+", encoding="utf-8")
        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=cls.server_launch_timeout,
                other_args=DSV4_4P_OTHER_ARGS
                + ["--c128-page-size", str(C128_PAGE_SIZE)],
                env={**os.environ, **DSV4_4P_ENVS},
                return_stdout_stderr=(cls.out_log, cls.err_log),
            )
        except Exception:
            cls.out_log.close()
            cls.err_log.close()
            os.remove(cls.out_path)
            os.remove(cls.err_path)
            raise

    @classmethod
    def tearDownClass(cls):
        process = getattr(cls, "process", None)
        if process is not None:
            # SIGTERM first: the validation class below relaunches on the same
            # cards, and a bare SIGKILL can leave NPU memory held for minutes
            # after the tree is reaped (see terminate_and_kill_process_tree).
            terminate_and_kill_process_tree(process)
        for log, path in ((cls.out_log, cls.out_path), (cls.err_log, cls.err_path)):
            log.close()
            if os.path.exists(path):
                os.remove(path)

    def test_value_reported_and_dsv4_pool_built(self):
        info = requests.get(f"{self.base_url}/server_info", timeout=60).json()
        self.assertEqual(info["c128_page_size"], C128_PAGE_SIZE)
        # The DSV4 pool configurator's own log proves the C128 pool under
        # test is actually the DSV4 one (absent on the plain ascend path).
        self.assertTrue(
            _wait_for_log((self.out_log, self.err_log), "DSV4 SWA sizing:"),
            "The DSV4 pool sizing log is missing — the C128 pool was not "
            "built on the DSV4 path this test relies on",
        )

    def test_long_prompt_across_c128_groups(self):
        """A prompt spanning several C128 groups (128 * page-size tokens each)
        and multiple prefill chunks must come back with the right answer
        under the page width the flag selected."""
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": _needle_prompt(),
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
            timeout=600,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertIn(NEEDLE, resp.json()["text"])
        self.assertGreater(resp.json()["meta_info"]["prompt_tokens"], 2 * 4096)
        self.assertIsNone(self.process.poll(), "Server crashed during test")


class TestNpuC128PageSizeValidation(CustomTestCase):
    """The DSV4 memory pool's own constraint, exercised on a bare launch (no
    concurrent server holding the cards).

    [Test Category] Parameter
    [Test Target] --c128-page-size
    """

    model = DEEPSEEK_V4_FLASH_WEIGHTS_PATH
    # The rejection fires at DSV4 pool init, i.e. after weight load, so this
    # budget only has to cover load + the failed pool build — not a healthy
    # server's full readiness ramp. Well under the functional class's 2400s,
    # DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH magnitude.
    server_launch_timeout = 1200

    def test_non_multiple_of_16_rejected(self):
        """--c128-page-size 24 → the DSV4 memory pool itself rejects it at
        startup. This validation only exists on the DSV4 path, so the
        rejection doubles as proof the pool consumes the flag."""
        out_fd, out_path = tempfile.mkstemp(suffix=".out.log")
        err_fd, err_path = tempfile.mkstemp(suffix=".err.log")
        os.close(out_fd)
        os.close(err_fd)
        out_log = open(out_path, "w+", encoding="utf-8")
        err_log = open(err_path, "w+", encoding="utf-8")
        try:
            process, launch_error = None, None
            try:
                process = popen_launch_server(
                    self.model,
                    DEFAULT_URL_FOR_TEST,
                    timeout=self.server_launch_timeout,
                    other_args=DSV4_4P_OTHER_ARGS + ["--c128-page-size", "24"],
                    env={**os.environ, **DSV4_4P_ENVS},
                    return_stdout_stderr=(out_log, err_log),
                )
            except Exception as exc:
                launch_error = exc

            if process is not None:
                # popen_launch_server only returns on a healthy boot, so the
                # pool never saw 24: the flag was silently dropped or the
                # pool's multiple-of-16 validation is missing.
                terminate_and_kill_process_tree(process)
                self.fail(
                    "--c128-page-size 24 was accepted and the server booted "
                    "healthy: the flag likely never reached the DSV4 pool, or "
                    "its multiple-of-16 validation is missing"
                )

            # The scheduler subprocess raises ValueError at pool init and
            # SIGQUITs the launcher (run_scheduler_process's except branch),
            # so the launch must have failed because the process died — not
            # because it merely ran out of time.
            self.assertIsNotNone(launch_error)
            self.assertIn(
                "exited",
                str(launch_error),
                f"Expected the server process to die on the DSV4 pool's "
                f"rejection, got: {launch_error}",
            )

            # The ValueError traceback is logged by the scheduler subprocess
            # into the captured stderr; the pump threads may still be draining
            # the pipes, so poll with a bounded window before asserting.
            self.assertTrue(
                _wait_for_log((out_log, err_log), C128_POOL_VALIDATION_ERROR),
                f"The DSV4 pool's rejection message did not appear in the "
                f"captured server output (launch error: {launch_error})",
            )
        finally:
            for log, path in ((out_log, out_path), (err_log, err_path)):
                log.close()
                if os.path.exists(path):
                    os.remove(path)


if __name__ == "__main__":
    unittest.main()
