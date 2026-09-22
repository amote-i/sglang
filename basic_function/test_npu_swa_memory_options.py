"""
Test the SWA memory pool options on NPU with a hybrid-SWA model
(Llama-4-Scout-17B-16E-Instruct: Llama4ForConditionalGeneration is in
is_hybrid_swa_model's allowlist, and every 4th layer is a full-attention
layer — 48 layers split into 12 full + 36 sliding-window, window 8192):
--swa-full-tokens-ratio and --disable-hybrid-swa-memory.

Both options configure the hybrid SWA memory pool:
  - --swa-full-tokens-ratio sets the ratio of SWA-layer KV tokens to
    full-layer KV tokens (0 < ratio <= 1, fallback 0.8);
  - --disable-hybrid-swa-memory turns the hybrid SWA pool off, so the SWA
    layers store full history and the per-token cost rises sharply.

Scope (functional coverage): with a dense model the hybrid pool never
activates and both flags are runtime no-ops, so this test uses a real
hybrid-SWA model and asserts on the pool sizes the configurator itself
logs at startup:

  - ratio 0.5 → "Use sliding window memory pool. full_layer_tokens=A,
    swa_layer_tokens=B" with B ~= A * 0.5 (page-aligned), and serving
    works;
  - ratio 1.0 → swa_layer_tokens ~= full_layer_tokens, i.e. the ratio is
    the knob that moved, not something else about the launch;
  - --disable-hybrid-swa-memory → the hybrid-pool log is absent and the
    scheduler's max_total_num_tokens drops below the hybrid run's
    full_layer_tokens;
  - an out-of-range ratio is rejected at startup for every model.

Model note: this is the smallest hybrid-SWA model with NPU runtime
evidence (meta-llama/Llama-4-Scout-17B-16E-Instruct is ~109B total MoE
params, so it runs TP4 on the 4-NPU runner). The launch flags mirror the
known-good NPU launch in
test/registered/npu/llm_models/test_npu_llama4_scount_17b_16e.py
(chat template, TP4, pinned context, ascend attention backend) and are
identical across all launches below, so pool-size comparisons stay
apples-to-apples. The ratio is a capacity knob independent of the
swa:full layer counts, so every assertion below is model-agnostic;
Llama-4's override only auto-selects an attention backend (we pass one
explicitly) and cannot disable the hybrid pool. The SWA-chunk-cap pool
mode, which would bypass --swa-full-tokens-ratio, requires an explicit
--max-running-requests, which these launches never pass.

[Test Category] Parameter
[Test Target] --swa-full-tokens-ratio;--disable-hybrid-swa-memory
"""

import os
import re
import subprocess
import sys
import tempfile
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_4_SCOUT_17B_16E_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    auto_config_device,
    popen_launch_server,
)

# Four full server launches of a TP4 109B MoE; launch timeout itself is
# 1000s, matching the known-good Llama-4 NPU run.
register_npu_ci(est_time=3600, suite="full-4-npu-a3", nightly=True)

SWA_POOL_LOG = re.compile(
    r"Use sliding window memory pool\. full_layer_tokens=(\d+), "
    r"swa_layer_tokens=(\d+)"
)
MAX_TOKENS_LOG = re.compile(r"max_total_num_tokens=(\d+),")

# Mirrors timeout_for_server_launch=1000 of the known-good Llama-4 NPU
# launch (DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH=600 is sized for small models).
LAUNCH_TIMEOUT = 1000


class TestNpuSwaMemoryOptions(CustomTestCase):
    """Verify the SWA memory pool options change the actual pool layout on a
    hybrid-SWA model and do not break serving.

    [Test Category] Parameter
    [Test Target] --swa-full-tokens-ratio;--disable-hybrid-swa-memory
    """

    model = LLAMA_4_SCOUT_17B_16E_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def _launch_with_logs(cls, extra_args):
        """Launch on a fresh log pair; returns (process, read_log, paths)."""
        out_fd, out_path = tempfile.mkstemp(suffix=".out.log")
        err_fd, err_path = tempfile.mkstemp(suffix=".err.log")
        os.close(out_fd)
        os.close(err_fd)
        out_log = open(out_path, "w+", encoding="utf-8")
        err_log = open(err_path, "w+", encoding="utf-8")
        try:
            process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=LAUNCH_TIMEOUT,
                other_args=[
                    # Known-good Llama-4 NPU launch, identical across all
                    # runs below (see module docstring).
                    "--chat-template",
                    "llama-4",
                    "--tp-size",
                    "4",
                    "--context-length",
                    "8192",
                    "--attention-backend",
                    "ascend",
                    "--disable-cuda-graph",
                    "--mem-fraction-static",
                    "0.9",
                    "--disable-radix-cache",
                ]
                + extra_args,
                return_stdout_stderr=(out_log, err_log),
            )
        except Exception:
            out_log.close()
            err_log.close()
            os.remove(out_path)
            os.remove(err_path)
            raise

        def read_log():
            out_log.flush()
            err_log.flush()
            out_log.seek(0)
            err_log.seek(0)
            return out_log.read() + err_log.read()

        return process, read_log, (out_path, err_path), (out_log, err_log)

    @staticmethod
    def _close_logs(logs, paths):
        for log in logs:
            log.close()
        for path in paths:
            if os.path.exists(path):
                os.remove(path)

    @staticmethod
    def _wait_for_log(needle, read_log, timeout=60):
        """The teed output lands asynchronously; poll for the needle."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            text = read_log()
            if needle in text:
                return text
            time.sleep(2)
        return read_log()

    @staticmethod
    def _assert_generate_ok():
        """Smoke-check that the oddly-configured server still serves: HTTP
        200 with a non-empty completion proves prefill/decode/detokenize
        run on the configured pool. The pool geometry itself is what the
        numeric log assertions observe; a knowledge probe (e.g. expecting
        a specific fact) would only make this parameter test flaky."""
        gen_resp = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
        )
        assert gen_resp.status_code == 200
        assert gen_resp.json()["text"].strip(), "Serving returned an empty completion"

    def _run_hybrid_launch(self, extra_args, expected_ratio):
        """Launch with the hybrid pool on, parse its logged split, and check
        swa_layer_tokens matches the configured ratio."""
        process, read_log, paths, logs = self._launch_with_logs(extra_args)
        try:
            log_text = self._wait_for_log("Use sliding window memory pool", read_log)
            matches = SWA_POOL_LOG.search(log_text)
            self.assertIsNotNone(
                matches,
                "The hybrid SWA pool log is missing — either the model stopped "
                "being hybrid-SWA or the pool init path changed",
            )
            full_tokens, swa_tokens = int(matches.group(1)), int(matches.group(2))
            ratio = swa_tokens / full_tokens
            # Page alignment shifts the split by at most page_size over the
            # whole pool (page_size/full_tokens, well under 1%) — orders of
            # magnitude inside the 0.05 delta for any real memory budget.
            self.assertAlmostEqual(
                ratio,
                expected_ratio,
                delta=0.05,
                msg=f"swa_layer_tokens/full_layer_tokens should track "
                f"--swa-full-tokens-ratio {expected_ratio}, got {swa_tokens}/"
                f"{full_tokens} = {ratio:.3f}",
            )

            self._assert_generate_ok()
            self.assertIsNone(process.poll(), "Server crashed during test")
            return full_tokens
        finally:
            kill_process_tree(process.pid)
            self._close_logs(logs, paths)

    def test_swa_full_tokens_ratio_half(self):
        """--swa-full-tokens-ratio 0.5 → the SWA pool gets ~half the full
        pool's tokens."""
        self._run_hybrid_launch(["--swa-full-tokens-ratio", "0.5"], 0.5)

    def test_swa_full_tokens_ratio_one(self):
        """--swa-full-tokens-ratio 1.0 → the SWA pool matches the full pool,
        proving the ratio is the knob that moved between the two runs."""
        self._run_hybrid_launch(["--swa-full-tokens-ratio", "1.0"], 1.0)

    def test_disable_hybrid_swa_memory(self):
        """--disable-hybrid-swa-memory → no hybrid pool split, and the total
        token capacity falls below the hybrid run's full-layer tokens
        because the SWA layers now pay full per-token cost.

        Baseline is a ratio-0.5 run. The direction is model-independent:
        with f full and s > 0 SWA layers, the hybrid run fits f*F + s*S
        layer-token units of KV into the VRAM budget (F = logged
        full_layer_tokens, S = swa_layer_tokens ~= 0.5*F < F), while the
        disabled run spreads that same budget over f+s layers per token,
        so its capacity is exactly (f*F + s*S)/(f+s) < F. No architecture
        layer-ratio constant, so this holds for any hybrid-SWA model."""
        hybrid_full_tokens = self._run_hybrid_launch(
            ["--swa-full-tokens-ratio", "0.5"], 0.5
        )

        process, read_log, paths, logs = self._launch_with_logs(
            ["--disable-hybrid-swa-memory"]
        )
        try:
            log_text = self._wait_for_log("max_total_num_tokens=", read_log)
            self.assertNotIn(
                "Use sliding window memory pool",
                log_text,
                "The hybrid SWA pool was built despite --disable-hybrid-swa-memory",
            )
            matches = MAX_TOKENS_LOG.search(log_text)
            self.assertIsNotNone(matches, "max_total_num_tokens log line missing")
            disabled_tokens = int(matches.group(1))
            self.assertLess(
                disabled_tokens,
                hybrid_full_tokens,
                f"With the hybrid pool disabled the SWA layers pay full "
                f"per-token cost, so the capacity ({disabled_tokens}) must "
                f"drop below the hybrid ratio-0.5 run's full-layer tokens "
                f"({hybrid_full_tokens}): the same budget that held "
                f"f*{hybrid_full_tokens} + s*S layer-tokens now spreads "
                f"over f+s layers per token, which is strictly less than F "
                f"whenever s > 0 and S < F",
            )

            self._assert_generate_ok()
            self.assertIsNone(process.poll(), "Server crashed during test")
        finally:
            kill_process_tree(process.pid)
            self._close_logs(logs, paths)

    @classmethod
    def _expect_startup_failure(cls, extra_args, expected_error):
        """Arg-validation failures must abort startup with a clear message."""
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            cls.model,
            "--device",
            auto_config_device(),
            "--attention-backend",
            "ascend",
        ] + extra_args
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            _, stderr = process.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            kill_process_tree(process.pid)
            raise AssertionError(f"Server should reject {extra_args} at startup")

        assert process.returncode != 0, f"Server unexpectedly accepted {extra_args}"
        assert expected_error in stderr, (
            f"Expected {expected_error!r} in stderr, got:\n{stderr[-2000:]}"
        )

    def test_ratio_out_of_range_rejected(self):
        """--swa-full-tokens-ratio 1.5 → rejected by the range validation
        (0 < ratio <= 1), which runs for every model at startup, before any
        weights load (so no TP sizing is needed for this launch)."""
        self._expect_startup_failure(
            ["--swa-full-tokens-ratio", "1.5"],
            "--swa-full-tokens-ratio should be in range (0, 1.0].",
        )


if __name__ == "__main__":
    unittest.main()
