"""
Test --pp-max-micro-batch-size and --pp-async-batch-depth on NPU at PP=2
with one directly observable functional case.

  - --pp-max-micro-batch-size caps the requests a single prefill
    micro-batch may admit (positive integer; None auto-computes it);
  - --pp-async-batch-depth builds the PP event loop out of
    pp_size + depth micro-batch slots and, when > 0, sends a finished
    micro-batch's outputs before the next forward is launched
    (0 = synchronous exchange).

Direct observations:

  - the cap's effect is visible in the pre-existing per-prefill-batch stats
    line ("Prefill batch ... #new-seq: N", metrics_reporter.py), where N is
    len(adder.can_run_list) -- the admission count of that micro-batch, the
    number the cap's budget (get_num_allocatable_reqs: pp_budget =
    pp_max_micro_batch_size - running_bs) bounds. Launched with the cap set
    to 1, no prefill micro-batch may report more than one new sequence, and
    at least one of those lines must show #queue-req >= 1: a request held in
    the queue while the micro-batch admitted only one is what makes the <= 1
    result the cap's admission decision instead of requests simply arriving
    one at a time. All eight barrier-released requests must complete with
    correct content;
  - the depth only reshapes the PP event loop, so it is asserted where that
    is deterministic: the CPU unit test of the loop
    (test_disagg_idle_step_counters.test_pp_idle_cycles) runs the real
    init_pp_loop_state over a parametrized depth and asserts the loop is
    built with pp_size + depth slots. What this case adds is the NPU/e2e
    half: the eight concurrent requests must complete with correct content
    under the depth > 0 early-exchange branch, and the value must be present
    in the config the scheduler processes run with. How much overlap the
    extra slots buy is a timing observation and belongs to the perf suites.

Both values are additionally read back from the scheduler processes
(/server_info -> internal_states, one entry per scheduler, each holding the
config that process runs with), one step closer to the point of use than the
top-level fields echoed by the tokenizer manager.

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
from typing import List, Tuple

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

PP_SIZE = 2
PP_MAX_MICRO_BATCH_SIZE = 1
PP_ASYNC_BATCH_DEPTH = 2


def _parse_prefill_admissions(log_text: str) -> List[Tuple[int, int]]:
    """(new sequences admitted, requests still queued) per prefill
    micro-batch, read from the "#new-seq: N ... #queue-req: M" stats line."""
    admissions = []
    for line in log_text.splitlines():
        if "Prefill batch" not in line:
            continue
        new_seq = re.search(r"#new-seq: (\d+)", line)
        queue_req = re.search(r"#queue-req: (\d+)", line)
        if new_seq and queue_req:
            admissions.append((int(new_seq.group(1)), int(queue_req.group(1))))
    return admissions


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
        most one new request while requests are still queued, and all eight
        barrier-released requests complete with correct content on the async
        exchange path."""
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
                    str(PP_SIZE),
                    "--pp-max-micro-batch-size",
                    str(PP_MAX_MICRO_BATCH_SIZE),
                    "--pp-async-batch-depth",
                    str(PP_ASYNC_BATCH_DEPTH),
                ],
                return_stdout_stderr=(out_log, err_log),
            )
            try:
                info = requests.get(f"{self.base_url}/server_info", timeout=30).json()
                # internal_states 是各 scheduler 进程 resolved_server_args_dict()
                # 的回显：与这些进程里的 get_parallel() 同源，也就是
                # init_pp_loop_state / get_num_allocatable_reqs 真正读的那份运行时
                # 配置。顶层字段只回显 tokenizer manager 的 ServerArgs，证明力更弱。
                states = info["internal_states"]
                self.assertTrue(states, "No scheduler internal state was reported")
                for state in states:
                    self.assertEqual(
                        state["pp_max_micro_batch_size"], PP_MAX_MICRO_BATCH_SIZE
                    )
                    self.assertEqual(
                        state["pp_async_batch_depth"], PP_ASYNC_BATCH_DEPTH
                    )

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
                # admission count and the queue depth at log time; give the
                # teed output a moment to settle, then require the lines to
                # exist (else the log mechanism changed and the cap assertion
                # below would be vacuous).
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

                admissions = _parse_prefill_admissions(log_text)
                self.assertTrue(
                    admissions,
                    "No prefill micro-batch was logged, so the cap cannot be "
                    "verified from the scheduler's own admission counts",
                )
                # 参数核心观测点：#new-seq 就是该 prefill micro-batch 的准入数
                # (len(adder.can_run_list)，上界由 pp_budget = pp_max_micro_batch_size
                # - running_bs 决定)，所以 cap=1 时任何一条 prefill 日志都不允许
                # 出现 #new-seq >= 2；cap 失效时八个并发请求会被同一轮塞进一个
                # micro-batch。
                self.assertEqual(
                    max(new_seq for new_seq, _ in admissions),
                    PP_MAX_MICRO_BATCH_SIZE,
                    "A prefill micro-batch admitted more than the cap of "
                    f"{PP_MAX_MICRO_BATCH_SIZE}: admission counts "
                    f"{sorted(new_seq for new_seq, _ in admissions)}",
                )
                # 同一行的 #queue-req 把"cap 在起作用"变成日志证据：至少有一行是
                # "队列里还压着请求、而该 micro-batch 只收了 1 个"。缺了这一行，
                # 请求逐个到达也能得到同样的 <=1 结果，上面的断言就是空转的。
                self.assertTrue(
                    any(queue_req >= 1 for _, queue_req in admissions),
                    "Every logged prefill micro-batch saw an empty queue, so "
                    "the <= 1 admission counts say nothing about the cap: "
                    f"{admissions}",
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
