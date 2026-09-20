"""
Test --enable-session-radix-cache on NPU.

The parameter tracks per-session references on the UnifiedRadixCache KV:
eviction consumes unreferenced entries before referenced ones, and closing
a session only dereferences its KV.

Scope (functional coverage):
  - a normal session (open → multi-turn generate → close) works end to end
    under the flag, with the second turn still reusing the KV cached by the
    first turn; closing the session is observed through the bookkeeping the
    flag itself adds — the tracker logs "release_session <id>: indexed <N>
    component leaves" only when the flag is on, and the test parses <N> and
    requires it to be > 0, proving the register path really tagged leaves;
  - the flag's eviction-ordering effect is exercised under real KV pressure
    (--max-total-tokens caps the pool): a session-referenced prefix and an
    unreferenced twin are cached back to back, then filler traffic several
    times the pool size forces eviction, and the two are re-probed: the
    referenced prefix must still hit (cached_tokens > 0) while the
    unreferenced twin must miss (cached_tokens == 0). The same-moment A/B
    contrast is what proves the ordering, not just that the cache works.

[Test Category] Parameter
[Test Target] --enable-session-radix-cache
"""

import os
import re
import tempfile
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="full-1-npu-a3", nightly=True)


class TestNpuEnableSessionRadixCache(CustomTestCase):
    """Verify --enable-session-radix-cache keeps session-based KV reuse
    working.

    [Test Category] Parameter
    [Test Target] --enable-session-radix-cache
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    # Small enough that ~2 filler rounds force eviction, large enough that
    # the session flow above never retraces.
    max_total_tokens = 30000
    probe_repeats = 60  # ~1.5k tokens per probe/filler text
    filler_rounds = 45  # ~67k filler tokens ≈ 2.2x the pool

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
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--attention-backend",
                    "ascend",
                    "--enable-session-radix-cache",
                    "--max-total-tokens",
                    str(cls.max_total_tokens),
                ],
                return_stdout_stderr=(cls.out_log, cls.err_log),
            )
        except Exception:
            # A failed launch leaves cls.process unset, so the teardown below
            # would abort before removing the files; clean up here instead.
            cls.out_log.close()
            cls.err_log.close()
            os.remove(cls.out_path)
            os.remove(cls.err_path)
            raise

    @classmethod
    def tearDownClass(cls):
        process = getattr(cls, "process", None)
        if process is not None:
            kill_process_tree(process.pid)
        for log, path in ((cls.out_log, cls.out_path), (cls.err_log, cls.err_path)):
            log.close()
            if os.path.exists(path):
                os.remove(path)

    @classmethod
    def _wait_for_release_log(cls, session_id, timeout=30):
        """The server output is teed into the log files by a background
        thread, so give the line a moment to land.

        Parses the tracker's release line ("release_session <id>: indexed
        <N> component leaves", session_ref_tracker.py) for this session and
        returns the indexed-leaf count, or None if the line never appeared
        within the timeout."""
        pattern = re.compile(
            rf"release_session {re.escape(session_id)}: indexed (\d+) "
            r"component leaves"
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            cls.out_log.seek(0)
            cls.err_log.seek(0)
            match = pattern.search(cls.out_log.read() + cls.err_log.read())
            if match is not None:
                return int(match.group(1))
            time.sleep(1)
        return None

    def test_flag_reported(self):
        info = requests.get(f"{self.base_url}/server_info").json()
        self.assertTrue(info["enable_session_radix_cache"])

    def test_session_kv_reuse_and_close(self):
        # Open a session and run two turns within it.
        open_resp = requests.post(
            f"{self.base_url}/open_session",
            json={"capacity_of_str_len": 1000},
            timeout=10,
        )
        self.assertEqual(open_resp.status_code, 200)
        session_id = open_resp.json()

        # Turn 1: builds the KV for the shared prefix.
        turn1 = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "Let me tell you something about France.",
                "session_params": {"id": session_id, "rid": None},
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
            timeout=30,
        )
        self.assertEqual(turn1.status_code, 200, turn1.text)
        self.assertEqual(
            turn1.json()["meta_info"]["cached_tokens"], 0, "Turn 1: clean start"
        )
        rid = turn1.json()["meta_info"]["id"]

        # Turn 2: extends the session via rid; must reuse turn 1's KV.
        turn2 = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "session_params": {"id": session_id, "rid": rid},
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
            timeout=30,
        )
        self.assertEqual(turn2.status_code, 200, turn2.text)
        self.assertIn("Paris", turn2.json()["text"])
        # cached_tokens > 0 is NOT, by itself, evidence that the flag works:
        # the plain radix cache produces the same prefix reuse with the flag
        # off. The flag's effect is proven by the eviction A/B contrast in
        # test_referenced_kv_survives_eviction_pressure and by the release
        # log below — keep both if this flow is ever simplified.
        self.assertGreater(
            turn2.json()["meta_info"]["cached_tokens"],
            0,
            "Turn 2 should reuse the KV cached by turn 1",
        )

        # Closing the session only dereferences its KV; it must succeed and
        # leave the server healthy.
        close_resp = requests.post(
            f"{self.base_url}/close_session",
            json={"session_id": session_id},
            timeout=10,
        )
        self.assertEqual(close_resp.status_code, 200)

        # The flag's own bookkeeping is observable here and only with the
        # flag on: the tracker logs the leaves it released for this session.
        # Parsing the count (not just the line) closes the "indexed 0" gap —
        # a release line with a zero count would mean the register path
        # never tagged this session's KV, so this test no longer relies on
        # the eviction-contrast test alone to catch a half-broken flag.
        indexed = self._wait_for_release_log(session_id)
        self.assertIsNotNone(
            indexed,
            "Closing the session did not produce the session-radix-cache "
            "release log, so the flag's session tracking never ran",
        )
        self.assertGreater(
            indexed,
            0,
            "The session-radix-cache release indexed 0 leaves, so the "
            "register path never tagged this session's KV",
        )

        health = requests.get(f"{self.base_url}/health", timeout=10)
        self.assertEqual(health.status_code, 200)
        self.assertIsNone(self.process.poll(), "Server crashed during test")

    def test_referenced_kv_survives_eviction_pressure(self):
        """Under KV pressure the eviction order must consume unreferenced
        leaves before session-referenced ones: a session-referenced prefix
        and an unreferenced twin are cached back to back, filler traffic
        ~2x the pool forces eviction, then both are re-probed — referenced
        hits, unreferenced misses."""
        sentence = (
            "The survey team logged the depth of every borehole along the "
            "northern ridge and filed the readings with the regional archive. "
        )

        def body(tag):
            # The unique tag must be the FIRST token: a shared opening like
            # "(record " would give every probe a few matching prefix tokens,
            # and the unreferenced twin's cached_tokens == 0 probe below
            # would spuriously hit them.
            return f"{tag}: " + sentence * self.probe_repeats

        # One referenced prefix (inside an open session) and one
        # unreferenced twin of the same size.
        open_resp = requests.post(
            f"{self.base_url}/open_session",
            json={"capacity_of_str_len": 100000},
            timeout=10,
        )
        self.assertEqual(open_resp.status_code, 200)
        session_id = open_resp.json()

        referenced_prompt = body("SessionA9")
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": referenced_prompt,
                "session_params": {"id": session_id, "rid": None},
                "sampling_params": {"temperature": 0, "max_new_tokens": 4},
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["meta_info"]["cached_tokens"], 0)

        unreferenced_prompt = body("PlainB7")
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": unreferenced_prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 4},
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["meta_info"]["cached_tokens"], 0)

        # Filler traffic several times the pool size, sequential so each
        # insert sees the pressure deterministically.
        for i in range(self.filler_rounds):
            resp = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": body(f"Filler{i}K2"),
                    "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                },
                timeout=120,
            )
            self.assertEqual(resp.status_code, 200, resp.text)

        # Probe the session-referenced prefix: its KV must have survived.
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": referenced_prompt,
                "session_params": {"id": session_id, "rid": None},
                "sampling_params": {"temperature": 0, "max_new_tokens": 4},
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertGreater(
            resp.json()["meta_info"]["cached_tokens"],
            0,
            "The session-referenced prefix was evicted under pressure — "
            "eviction did not prefer unreferenced entries",
        )

        # Probe the unreferenced twin: same size, same age, no session —
        # it must have been evicted to make room.
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": unreferenced_prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 4},
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(
            resp.json()["meta_info"]["cached_tokens"],
            0,
            "The unreferenced twin survived the same pressure that should "
            "have consumed it first",
        )

        close_resp = requests.post(
            f"{self.base_url}/close_session",
            json={"session_id": session_id},
            timeout=10,
        )
        self.assertEqual(close_resp.status_code, 200)
        self.assertIsNone(self.process.poll(), "Server crashed during test")


if __name__ == "__main__":
    unittest.main()
