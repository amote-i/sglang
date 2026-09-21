"""
Test --retraction-policy on NPU with one directly observable functional case.

--retraction-policy selects the decode retraction policy used when the KV
cache is full:
  - "length" (default): retracts short-output, long-input requests first;
  - "priority": retracts lower-priority requests first, and therefore
    requires --enable-priority-scheduling.

Direct observation, no scheduler instrumentation: SGLANG_TEST_RETRACT (a
pre-existing test hook in scheduler.py) force-retracts the policy's chosen
victim every SGLANG_TEST_RETRACT_INTERVAL forwards. The victim loses its
decode turn and pays a re-prefill per event while non-victims decode
uninterrupted, so the STANDING VICTIM finishes last. For the request matrix
below (input length grows with the priority value, max_new_tokens uniform)
the two policies predict different standing victims:

  - priority policy: _get_decode_retraction_order pops the lowest priority
    first, so the priority-0 request is the victim of every event and must
    finish last;
  - length policy (what a no-op --retraction-policy degrades to): in
    lockstep decode the output lengths stay identical, so the
    (len(output_ids), -len(origin_input_ids)) key falls to the LONGEST
    input and the priority-3 request finishes last instead.

The completion-order assertion below (priority-0 is the last of the four to
finish) therefore fails if --retraction-policy stops steering victim
choice. The pre-existing "Testing retraction." warning proves the forced
events actually reached the scheduler, so the ordering check cannot pass
vacuously.

[Test Category] Parameter
[Test Target] --retraction-policy
"""

import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=900, suite="full-1-npu-a3", nightly=True)


class TestNpuRetractionPolicy(CustomTestCase):
    """Verify --retraction-policy priority actually holds back the lowest
    priority request, observed through completion order.

    [Test Category] Parameter
    [Test Target] --retraction-policy
    """

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    # Input length grows with the priority value (p0 shortest ... p3
    # longest) while max_new_tokens stays uniform: under the priority policy
    # the standing victim is p0, under the length policy it is p3, so the
    # last-finisher assertion discriminates the two policies. Every prompt
    # ends with "...the capital of France is" so greedy decoding continues
    # with " Paris".
    RETRACT_PROMPTS = {
        0: "The capital of France is",
        1: "Yes, the capital of France is",
        2: "As everyone knows, the capital of France is",
        3: "It is widely known and taught in schools that the capital of France is",
    }
    # Uniform and generous: no request can hit its cap (or finish early)
    # before enough forced events have accumulated to hold the standing
    # victim clearly behind the other three.
    RETRACT_MAX_NEW_TOKENS = 64

    def test_priority_policy_holds_back_lowest_priority(self):
        """--retraction-policy priority under forced retraction → the
        priority-0 request is the last of the four to finish."""
        out_fd, out_path = tempfile.mkstemp(suffix=".out.log")
        err_fd, err_path = tempfile.mkstemp(suffix=".err.log")
        os.close(out_fd)
        os.close(err_fd)
        out_log = open(out_path, "w+", encoding="utf-8")
        err_log = open(err_path, "w+", encoding="utf-8")
        try:
            with envs.SGLANG_TEST_RETRACT.override(True):
                process = popen_launch_server(
                    self.model,
                    self.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=[
                        "--attention-backend",
                        "ascend",
                        "--retraction-policy",
                        "priority",
                        "--enable-priority-scheduling",
                    ],
                    return_stdout_stderr=(out_log, err_log),
                )
            try:
                info = requests.get(f"{self.base_url}/server_info").json()
                self.assertEqual(info["retraction_policy"], "priority")

                num_requests = len(self.RETRACT_PROMPTS)
                barrier = threading.Barrier(num_requests)
                finish_times = {}

                def send(priority):
                    # The pool runs exactly num_requests workers, so every
                    # thread reaches wait(); the timeout only turns a
                    # pathological stall into a loud failure instead of a
                    # hang.
                    barrier.wait(timeout=60)
                    resp = requests.post(
                        f"{self.base_url}/generate",
                        json={
                            "text": self.RETRACT_PROMPTS[priority],
                            "priority": priority,
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": self.RETRACT_MAX_NEW_TOKENS,
                            },
                        },
                        timeout=300,
                    )
                    self.assertEqual(resp.status_code, 200, resp.text)
                    self.assertIn("Paris", resp.text)
                    finish_times[priority] = time.monotonic()

                with ThreadPoolExecutor(max_workers=num_requests) as pool:
                    list(pool.map(send, self.RETRACT_PROMPTS))

                self.assertIsNone(process.poll(), "Server crashed during test")

                # The forced events must have reached the scheduler (this
                # warning line is pre-existing), else the ordering assertion
                # below would pass vacuously.
                out_log.flush()
                err_log.flush()
                out_log.seek(0)
                err_log.seek(0)
                log_text = out_log.read() + err_log.read()
                self.assertIn(
                    "Testing retraction.",
                    log_text,
                    "SGLANG_TEST_RETRACT did not reach the scheduler, so no "
                    "retraction event ever fired",
                )

                last_finisher = max(finish_times, key=finish_times.get)
                self.assertEqual(
                    last_finisher,
                    0,
                    f"The last request to finish should be the priority-0 "
                    f"one: the lowest priority is the standing victim of "
                    f"every forced retraction, while a no-op policy would "
                    f"hold back the longest input (priority 3) instead. "
                    f"Finish order was "
                    f"{sorted(finish_times, key=finish_times.get)}",
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
