"""
Test --pp-max-micro-batch-size and --pp-async-batch-depth on NPU at PP=2
with one directly observable functional case.

  - --pp-max-micro-batch-size caps the new requests a single prefill
    micro-batch may admit (positive integer; None auto-computes it);
  - --pp-async-batch-depth lets a PP rank send a finished micro-batch's
    outputs before the next forward completes (0 = synchronous exchange).

Direct observations, no scheduler instrumentation:

  - the cap's effect is visible in the pre-existing per-prefill-batch stats
    line ("Prefill batch ... #new-seq: N", metrics_reporter.py): launched
    with the cap set to 1, no prefill micro-batch may report more than one
    new sequence, while the eight barrier-released requests must all be
    admitted (sum of #new-seq >= 8) and complete — a broken cap must get
    the chance to admit >1 into one micro-batch for the assertion to bite;
  - for the depth there is no external trace of the loop-slot count, so the
    direct observation is functional: with depth 2 every PP iteration takes
    the early output-exchange branch (scheduler_pp_mixin.py keys off
    depth > 0), and the eight concurrent requests must all complete with
    correct content under it — a corrupted or deadlocked async exchange
    shows up right there. Whether the configured depth value (as opposed to
    any depth > 0) reached the loop is not externally observable and is
    left to the perf suites.

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

register_npu_ci(est_time=800, suite="full-2-npu-a3", nightly=True)

# 8 (not fewer) so that even if the scheduler drains a couple of requests
# into separate micro-batches while the rest arrive, several requests still
# contend for one admission instant -- a broken cap must get the chance to
# admit >1 into a single micro-batch for the cap assertion to bite.
NUM_CONCURRENT_REQUESTS = 8


class TestNpuPpBatchParams(CustomTestCase):
    """Verify --pp-max-micro-batch-size caps prefill micro-batch admission
    and serving stays correct under --pp-async-batch-depth at PP=2.

    [Test Category] Parameter
    [Test Target] --pp-max-micro-batch-size;--pp-async-batch-depth
    """

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    def test_micro_batch_cap_and_async_depth(self):
        """cap=1 + depth=2 at PP=2 → every prefill micro-batch admits at
        most one new request, and all eight barrier-released requests
        complete with correct content on the async exchange path."""
        out_fd, out_path = tempfile.mkstemp(suffix=".out.log")
        err_fd, err_path = tempfile.mkstemp(suffix=".err.log")
        os.close(out_fd)
        os.close(err_fd)
        out_log = open(out_path, "w+", encoding="utf-8")
        err_log = open(err_path, "w+", encoding="utf-8")
        try:
            process = popen_launch_server(
                self.model,
                self.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--attention-backend",
                    "ascend",
                    "--pp-size",
                    "2",
                    "--pp-max-micro-batch-size",
                    "1",
                    "--pp-async-batch-depth",
                    "2",
                ],
                return_stdout_stderr=(out_log, err_log),
            )
            try:
                info = requests.get(f"{self.base_url}/server_info", timeout=30).json()
                self.assertEqual(info["pp_max_micro_batch_size"], 1)
                self.assertEqual(info["pp_async_batch_depth"], 2)

                # The threads line up on a Barrier and fire in the same
                # instant, so several requests sit in the waiting queue at
                # one scheduling moment -- that is what gives a broken cap
                # the chance to admit >1 request into a single prefill
                # micro-batch, which is the condition the cap assertion
                # below detects.
                barrier = threading.Barrier(NUM_CONCURRENT_REQUESTS)

                def _fire(_):
                    # The pool runs exactly NUM_CONCURRENT_REQUESTS workers,
                    # so every thread reaches wait(); the timeout only turns
                    # a pathological stall into a loud failure instead of a
                    # hang.
                    barrier.wait(timeout=60)
                    return requests.post(
                        f"{self.base_url}/generate",
                        json={
                            "text": "The capital of France is",
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": 16,
                            },
                        },
                        timeout=120,
                    )

                with ThreadPoolExecutor(max_workers=NUM_CONCURRENT_REQUESTS) as pool:
                    responses = list(pool.map(_fire, range(NUM_CONCURRENT_REQUESTS)))
                for resp in responses:
                    self.assertEqual(resp.status_code, 200, resp.text)
                    self.assertIn("Paris", resp.text)
                self.assertIsNone(process.poll(), "Server crashed during test")

                # The scheduler reports each prefill micro-batch with its
                # admission count; give the teed output a moment to settle,
                # then require the lines to exist (else the log mechanism
                # changed and the cap assertion below would be vacuous).
                deadline = time.time() + 30
                while True:
                    out_log.flush()
                    err_log.flush()
                    out_log.seek(0)
                    err_log.seek(0)
                    log_text = out_log.read() + err_log.read()
                    if "Prefill batch" in log_text or time.time() > deadline:
                        break
                    time.sleep(1)

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
                    "Fewer requests were admitted to prefill micro-batches "
                    "than were sent",
                )
                self.assertEqual(
                    max(new_seqs),
                    1,
                    f"A prefill micro-batch admitted more than the cap of 1: "
                    f"admission counts {sorted(new_seqs)}",
                )
            finally:
                kill_process_tree(process.pid)
        finally:
            out_log.close()
            err_log.close()
            os.remove(out_path)
            os.remove(err_path)


if __name__ == "__main__":
    unittest.main()
