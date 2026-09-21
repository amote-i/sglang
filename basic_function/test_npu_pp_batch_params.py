"""
Test the pipeline parallelism batch control options on NPU at PP=2:
--pp-max-micro-batch-size and --pp-async-batch-depth.

  - --pp-max-micro-batch-size caps the micro batch size of pipeline
    parallelism (positive integer; None auto-computes it);
  - --pp-async-batch-depth sets how many micro batches may be in flight
    asynchronously (0 = synchronous).

Scope (functional coverage at PP=2, the 2-NPU suite):

  - --pp-max-micro-batch-size feeds the per-micro-batch prefill admission
    budget (get_num_allocatable_reqs: pp_budget = pp_max_micro_batch_size -
    running_bs). The scheduler logs every prefill micro-batch as
    "Prefill batch ... #new-seq: N", so the test launches with the cap set
    to 1, sends eight requests released together by a barrier, and asserts
    from the captured server log that no micro-batch admitted more than one
    new request while all eight were admitted and completed. At pp_size=1
    the admission
    mechanism still runs, but the micro-batch dimension it controls does
    not — the cap's actual subject only exists at pp_size > 1.
  - --pp-async-batch-depth is a hard no-op at pp_size=1: every consumer is
    in the PP event loop (scheduler_pp_mixin), which is dispatched only for
    pp_size > 1. At PP=2 the loop grows to pp_size + depth slots and takes
    the early output-exchange branch on every iteration, so requests are
    served under the async exchange ordering. The loop slot count is logged
    at startup ("PP loop slots: N (pp_size=... + pp_async_batch_depth=...)"),
    so the test asserts the configured depth reached the live loop — a depth
    that stopped plumbing through would log the default slot count and fail
    the assertion. The degree of overlap that depth buys is a timing
    observation and belongs to the perf suites; concurrent requests still
    must complete correctly under it.

[Test Category] Parameter
[Test Target] --pp-max-micro-batch-size;--pp-async-batch-depth
"""

import os
import re
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

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

register_npu_ci(est_time=700, suite="full-2-npu-a3", nightly=True)

# 8 (not 4) so that even if the scheduler drains a couple of requests into
# separate micro-batches while the rest arrive, several requests still contend
# for one admission instant -- a broken cap must get the chance to admit >1
# into a single micro-batch for the cap assertion to bite.
NUM_CONCURRENT_REQUESTS = 8


class TestNpuPpBatchParams(CustomTestCase):
    """Verify --pp-max-micro-batch-size and --pp-async-batch-depth are
    consumed by the PP=2 scheduler and do not break serving.

    [Test Category] Parameter
    [Test Target] --pp-max-micro-batch-size;--pp-async-batch-depth
    """

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @staticmethod
    def _generate_one():
        return requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=120,
        )

    @classmethod
    def _launch_with_logs(cls, extra_args):
        """Launch on a fresh log pair; returns (process, out_log, err_log, paths)."""
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
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--attention-backend", "ascend", "--pp-size", "2"]
                + extra_args,
                return_stdout_stderr=(out_log, err_log),
            )
        except Exception:
            out_log.close()
            err_log.close()
            os.remove(out_path)
            os.remove(err_path)
            raise
        return process, out_log, err_log, (out_path, err_path)

    @staticmethod
    def _close_logs(out_log, err_log, paths):
        out_log.close()
        err_log.close()
        for path in paths:
            if os.path.exists(path):
                os.remove(path)

    @staticmethod
    def _server_log(out_log, err_log) -> str:
        out_log.flush()
        err_log.flush()
        out_log.seek(0)
        err_log.seek(0)
        return out_log.read() + err_log.read()

    @classmethod
    def _send_concurrent_and_check(cls, process):
        """Overlap requests so the scheduler runs several micro-batches, and
        require every request to complete on a healthy server.

        The threads line up on a Barrier and fire in the same instant, so
        several requests sit in the waiting queue at one scheduling moment --
        that is what gives a broken cap the chance to admit >1 request into
        a single prefill micro-batch, which is the condition the cap
        assertion in test_pp_max_micro_batch_size_caps_micro_batch detects.
        Residual timing assumption: even after the barrier, the HTTP/
        tokenizer path can still deliver requests a scheduler loop apart, so
        requests may occasionally be spread across micro-batches anyway; the
        barrier makes a broken cap overwhelmingly more likely to be caught,
        but does not make the observation fully deterministic.
        """
        barrier = threading.Barrier(NUM_CONCURRENT_REQUESTS)

        def _fire(_):
            # The pool runs exactly NUM_CONCURRENT_REQUESTS workers, so every
            # thread reaches wait(); the timeout only turns a pathological
            # stall into a loud failure instead of a hang.
            barrier.wait(timeout=60)
            return cls._generate_one()

        with ThreadPoolExecutor(max_workers=NUM_CONCURRENT_REQUESTS) as pool:
            responses = list(pool.map(_fire, range(NUM_CONCURRENT_REQUESTS)))
        for resp in responses:
            assert resp.status_code == 200, resp.text
            assert "Paris" in resp.text
        assert process.poll() is None, "Server crashed during test"

    @staticmethod
    def _wait_for_log(needle: str, out_log, err_log, timeout=30) -> str:
        """The server output is teed into the log files by a background
        thread, so poll until the line lands (or timeout), then return the
        full log text."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            log_text = TestNpuPpBatchParams._server_log(out_log, err_log)
            if needle in log_text:
                return log_text
            time.sleep(1)
        return TestNpuPpBatchParams._server_log(out_log, err_log)

    def test_pp_max_micro_batch_size_caps_micro_batch(self):
        """--pp-max-micro-batch-size 1 → no prefill micro-batch admits more
        than one new request, yet all eight barrier-released requests
        complete."""
        process, out_log, err_log, paths = self._launch_with_logs(
            ["--pp-max-micro-batch-size", "1"]
        )
        try:
            info = requests.get(f"{self.base_url}/server_info", timeout=30).json()
            self.assertEqual(info["pp_max_micro_batch_size"], 1)

            # Depth left at its default 0 here, so the loop must have been
            # built with exactly pp_size slots.
            log_text = self._wait_for_log(
                "PP loop slots: 2 (pp_size=2 + pp_async_batch_depth=0)",
                out_log,
                err_log,
            )
            self.assertIn(
                "PP loop slots: 2 (pp_size=2 + pp_async_batch_depth=0)",
                log_text,
                "The PP loop slot log is missing — the PP scheduler init "
                "path changed and the depth observation is stale",
            )

            self._send_concurrent_and_check(process)

            # The scheduler reports each prefill micro-batch with its
            # admission count; give the teed output a moment to settle, then
            # require the lines to exist (else the log mechanism changed and
            # the cap assertion below would be vacuous).
            log_text = self._wait_for_log("Prefill batch", out_log, err_log)

            new_seqs = [
                int(m)
                for line in log_text.splitlines()
                if "Prefill batch" in line
                for m in re.findall(r"#new-seq: (\d+)", line)
            ]
            self.assertTrue(
                new_seqs,
                "No prefill micro-batch was logged, so the cap cannot be "
                "verified from the scheduler's own admission counts",
            )
            self.assertGreaterEqual(
                sum(new_seqs),
                NUM_CONCURRENT_REQUESTS,
                "Fewer requests were admitted to prefill micro-batches than were sent",
            )
            self.assertEqual(
                max(new_seqs),
                1,
                f"A prefill micro-batch admitted more than the cap of 1: "
                f"admission counts {sorted(new_seqs)}",
            )
        finally:
            kill_process_tree(process.pid)
            self._close_logs(out_log, err_log, paths)

    def test_pp_async_batch_depth(self):
        """--pp-async-batch-depth 2 with --pp-max-micro-batch-size 2 → the
        async exchange path (pp_size + depth = 4 loop slots) serves
        overlapping requests correctly, and the live loop reports the
        depth-expanded slot count."""
        process, out_log, err_log, paths = self._launch_with_logs(
            ["--pp-async-batch-depth", "2", "--pp-max-micro-batch-size", "2"]
        )
        try:
            info = requests.get(f"{self.base_url}/server_info", timeout=30).json()
            self.assertEqual(info["pp_async_batch_depth"], 2)
            self.assertEqual(info["pp_max_micro_batch_size"], 2)

            # The flag's functional footprint: the async PP event loop is
            # sized pp_size + depth and takes the early output-exchange
            # branch. Without this assertion the test could not tell a
            # consumed depth from an ignored one.
            expected = "PP loop slots: 4 (pp_size=2 + pp_async_batch_depth=2)"
            log_text = self._wait_for_log(expected, out_log, err_log)
            self.assertIn(
                expected,
                log_text,
                "PP loop slots log does not reflect --pp-async-batch-depth 2 "
                "(expected 4 slots); the depth never reached the live loop",
            )
            self.assertNotIn(
                "PP loop slots: 2 (pp_size=2 + pp_async_batch_depth=0)",
                log_text,
                "The loop was built with the default depth despite "
                "--pp-async-batch-depth 2",
            )

            self._send_concurrent_and_check(process)
        finally:
            kill_process_tree(process.pid)
            self._close_logs(out_log, err_log, paths)


if __name__ == "__main__":
    unittest.main()
