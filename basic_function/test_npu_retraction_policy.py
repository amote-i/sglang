"""
Test --retraction-policy on NPU.

The parameter selects the decode retraction policy used when the KV cache
is full:
  - "length" (default): retracts short-output, long-input requests first;
  - "priority": retracts lower-priority requests first, and therefore
    requires --enable-priority-scheduling.

Scope (functional coverage):
  - both policies are accepted, reported via /server_info, and inference
    works with retraction wired to the chosen policy;
  - an invalid combination (priority without priority scheduling) is
    rejected at startup, which is what proves the parsed value reaches the
    validation pipeline;
  - a forced-retraction run (SGLANG_TEST_RETRACT, the same mechanism the
    retract decode suites use) drives the actual retraction code path under
    the "priority" policy with four requests whose input lengths grow with
    their priority value, and asserts the policy's core effect — WHICH
    request is retracted first. For these requests the two policies predict
    DIFFERENT first victims (priority policy: the priority-0 request;
    length policy: the longest-input priority-3 request — derivation in
    test_priority_policy_with_forced_retraction), so the first-victim
    assertion fails if --retraction-policy degrades into a no-op that
    always retracts by length. Retraction under real KV pressure is
    covered by the retract decode suites.

[Test Category] Parameter
[Test Target] --retraction-policy
"""

import os
import re
import subprocess
import sys
import tempfile
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
    auto_config_device,
    popen_launch_server,
)

register_npu_ci(est_time=500, suite="full-1-npu-a3", nightly=True)


class TestNpuRetractionPolicy(CustomTestCase):
    """Verify --retraction-policy values are accepted, do not break basic
    serving, and actually steer which request retraction picks.

    [Test Category] Parameter
    [Test Target] --retraction-policy
    """

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    # Request matrix for test_priority_policy_with_forced_retraction:
    # input length grows with the priority value (p0 shortest ... p3
    # longest) while max_new_tokens stays uniform — see that test's
    # docstring for why this makes the "length" and "priority" policies
    # predict different first victims. Every prompt ends with
    # "...the capital of France is" so greedy decoding continues with
    # "Paris".
    RETRACT_PROMPTS = {
        0: "The capital of France is",
        1: "Yes, the capital of France is",
        2: "As everyone knows, the capital of France is",
        3: "It is widely known and taught in schools that the capital of France is",
    }
    # Generous and uniform: no request can hit its cap (or finish early) in
    # the few decode steps before the first forced retraction event, and all
    # four keep decoding across many forced events.
    RETRACT_MAX_NEW_TOKENS = 64

    @staticmethod
    def _generate_one(
        priority=None, prompt="The capital of France is", max_new_tokens=16
    ):
        body = {
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
        }
        if priority is not None:
            body["priority"] = priority
        return requests.post(f"{DEFAULT_URL_FOR_TEST}/generate", json=body, timeout=120)

    @classmethod
    def _launch_server_and_check(cls, extra_args, expected_policy):
        process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "ascend"] + extra_args,
        )
        try:
            info = requests.get(f"{cls.base_url}/server_info").json()
            assert info["retraction_policy"] == expected_policy, (
                f"Expected retraction_policy={expected_policy}, "
                f"got {info['retraction_policy']}"
            )

            gen_resp = cls._generate_one()
            assert gen_resp.status_code == 200
            assert "Paris" in gen_resp.text

            assert process.poll() is None, "Server crashed during test"
        finally:
            kill_process_tree(process.pid)

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

    def test_length_policy(self):
        """--retraction-policy length → default policy, serving works."""
        self._launch_server_and_check(["--retraction-policy", "length"], "length")

    def test_priority_policy(self):
        """--retraction-policy priority → accepted together with
        --enable-priority-scheduling, serving works."""
        self._launch_server_and_check(
            ["--retraction-policy", "priority", "--enable-priority-scheduling"],
            "priority",
        )

    def test_priority_policy_without_priority_scheduling_rejected(self):
        """--retraction-policy priority without --enable-priority-scheduling →
        startup fails, proving the value reaches the cross-arg validation."""
        self._expect_startup_failure(
            ["--retraction-policy", "priority"],
            "--retraction-policy priority requires --enable-priority-scheduling",
        )

    def test_priority_policy_with_forced_retraction(self):
        """SGLANG_TEST_RETRACT drives real retractions through the policy's
        ordering branch; the policy's whole effect is WHICH request gets
        retracted, so the test asserts on the victim list the scheduler now
        logs in its retraction summary (rid=priority pairs).

        First-victim prediction for the four requests below (input length
        grows with priority value, max_new_tokens uniform):

          the prompts are tiny, so all four are admitted in one prefill
          batch; the waiting queue is sorted priority-descending
          (SchedulePolicy._sort_by_priority_and_fcfs), the running batch ends
          up ordered [p3, p2, p1, p0], and the requests decode in lockstep.
          The first forced event therefore sees four running requests with
          IDENTICAL output lengths, and:

          - priority policy: ScheduleBatch._get_decode_retraction_order
            sorts by (priority, len(output_ids), -len(origin_input_ids))
            descending and retract_decode pops the tail of that list, i.e.
            retracts the lowest numeric priority first — predicted first
            victim: p0;
          - length policy (what a no-op --retraction-policy degrades to):
            the key is (len(output_ids), -len(origin_input_ids)); with
            output lengths identical the tie falls to the LONGEST input —
            predicted first victim: p3.

          The predictions differ, so the assertion below (first victim is
          exactly the priority-0 request, checked by priority value and by
          rid) fails if the parameter stops steering the victim choice.
          A retracted request keeps its rid and is requeued into the
          priority-sorted waiting queue, so p0 is re-admitted last and every
          later forced event retracts it again — the divergence persists.

        Remaining assertions:
          - every victim is one of the four sent requests;
          - each summary line's #retracted_reqs matches its victim list;
          - within one event victims are listed least-preferred-first (see
            the scope note on that assertion below).

        Assumption: all four requests reach the running batch before the
        first non-empty forced event (tiny prompts, sent concurrently).
        Retraction under real KV pressure is covered by the retract decode
        suites.
        """
        # Priorities stay within a narrow band on purpose: a spread of more
        # than --priority-scheduling-preemption-threshold (10) would let the
        # high-priority request preempt a running one, which is a different
        # mechanism than the retraction ordering under test here.
        priorities = [0, 1, 2, 3]
        rid_by_priority = {}

        def send(priority):
            resp = self._generate_one(
                priority=priority,
                prompt=self.RETRACT_PROMPTS[priority],
                max_new_tokens=self.RETRACT_MAX_NEW_TOKENS,
            )
            rid_by_priority[priority] = resp.json()["meta_info"]["id"]
            return resp

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

                with ThreadPoolExecutor(max_workers=len(priorities)) as pool:
                    responses = list(pool.map(send, priorities))
                for resp in responses:
                    self.assertEqual(resp.status_code, 200, resp.text)
                    self.assertIn("Paris", resp.text)

                self.assertIsNone(process.poll(), "Server crashed during test")

                out_log.seek(0)
                err_log.seek(0)
                log_text = out_log.read() + err_log.read()
                self.assertIn(
                    "Testing retraction.",
                    log_text,
                    "SGLANG_TEST_RETRACT did not reach the scheduler, so the "
                    "retraction path was never forced (a 'KV cache pool is "
                    "full.' prefix instead would mean real KV pressure "
                    "retracted).",
                )
                # One record per retraction summary line, in log order.
                retract_events = []
                for line in log_text.splitlines():
                    if "retracted: [" not in line:
                        continue
                    if not (
                        "Testing retraction." in line or "Retract requests." in line
                    ):
                        continue
                    num_match = re.search(r"#retracted_reqs: (\d+)", line)
                    more_match = re.search(r"\(\+(\d+) more\)", line)
                    retract_events.append(
                        {
                            "num": int(num_match.group(1)) if num_match else None,
                            "pairs": re.findall(
                                r"rid=([^,\]]+),priority=(-?\d+)", line
                            ),
                            "truncated": int(more_match.group(1)) if more_match else 0,
                        }
                    )

                counts = [
                    event["num"] for event in retract_events if event["num"] is not None
                ]
                self.assertTrue(counts, "No retraction summary was logged")
                self.assertGreater(
                    max(counts),
                    0,
                    "Retraction was forced but reported 0 retracted requests",
                )

                # Cross-check: each summary's #retracted_reqs must equal the
                # number of rid=priority pairs it lists (the scheduler caps
                # the list at 8 and reports overflow as "... (+N more)",
                # unreachable with four requests).
                for event in retract_events:
                    self.assertIsNotNone(
                        event["num"],
                        f"Retraction summary without #retracted_reqs: {event}",
                    )
                    self.assertEqual(
                        event["num"],
                        len(event["pairs"]) + event["truncated"],
                        f"#retracted_reqs does not match the victim list: {event}",
                    )

                # Victim lists, one per retraction event, in log order.
                victim_events = [
                    [(rid, int(prio)) for rid, prio in event["pairs"]]
                    for event in retract_events
                    if event["pairs"]
                ]
                self.assertTrue(
                    victim_events,
                    "No retraction summary carried a victim list; the "
                    "rid/priority detail is missing from the retract log",
                )

                known_rids = set(rid_by_priority.values())
                for rid, _prio in (v for ev in victim_events for v in ev):
                    self.assertIn(
                        rid,
                        known_rids,
                        f"Retracted rid {rid} is not one of the requests sent",
                    )

                # The core policy assertion. Victim order comes from
                # ScheduleBatch._get_decode_retraction_order: the priority
                # policy pops the lowest numeric priority first, while a
                # no-op implementation degrades to the length sort, whose
                # first victim for these requests is the longest-input
                # priority-3 request instead — so this only holds when the
                # parameter actually steers the victim choice.
                first_event = victim_events[0]
                priority_by_rid = {v: k for k, v in rid_by_priority.items()}
                self.assertEqual(
                    priority_by_rid[first_event[0][0]],
                    0,
                    f"The first retracted victim should be the priority-0 "
                    f"request (lowest numeric priority retracts first), got "
                    f"priority {priority_by_rid[first_event[0][0]]}",
                )
                self.assertEqual(
                    first_event[0][0],
                    rid_by_priority[0],
                    "The first retracted victim's rid should be exactly the "
                    f"priority-0 request ({rid_by_priority[0]}), got "
                    f"{first_event[0][0]}",
                )
                # With ample KV memory each forced event retracts exactly one
                # request, so today this has effect only on multi-victim
                # events (real "KV cache pool is full." pressure, which this
                # test does not create); kept as a guard for those events.
                for ev in victim_events:
                    prios = [priority_by_rid[rid] for rid, _ in ev]
                    self.assertEqual(
                        prios,
                        sorted(prios),
                        f"Victims within one event must be listed ascending "
                        f"in priority (least-preferred first), got {prios}",
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
