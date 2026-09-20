"""
Test --moe-dp-size / --moe-data-parallel-size on NPU.

The two spellings are aliases of one field, the MoE data parallelism size
(``arg_groups/fields/parallel.py``, default 1). A value > 1 has one hard
topology prerequisite plus arithmetic gates, all in
``arg_groups/parallel_hook.py`` (``handle_context_parallelism``):

  - ``attn_cp_size == moe_dp_size``: outside the "both are 1" case the MoE-DP
    width is coupled to the attention-CP width, so a moe-dp launch must carry
    ``--attn-cp-size`` with it (the field defaults to 1, so moe-dp alone
    cannot launch). At ``tp=4`` the MoE-DP groups are the ranks {0,2} and
    {1,3} -- the same members the attention-CP groups get.
  - ``tp_size % moe_dp_size == 0``, ``ep_size * moe_dp_size <= tp_size``,
    ``pp_size == 1``.

Why the functional cases are skipped today: the MoE-DP token split is fed by
prefill context parallelism. Without ``--enable-prefill-cp`` every rank holds
the full sequence and the MoE-DP groups merely replicate work -- no value of
moe-dp-size then changes any observable output. Prefill CP was removed
upstream for non-DeepSeek-V4 NPU in the CP V1 deprecation (85d39401c8): the
CP KV path these backends call now raises (``layers/utils/cp_utils.py``), and
the officially registered same-topology case is skipped
(``test/registered/npu/llm_models/test_npu_qwen3_30b_attn_cp.py``). With the
token split gone there is no MoE-DP-specific functional observation left
(group construction logs nothing and ``/server_info`` reports only the
resolved field), so the launch cases keep their corrected arguments under
``@unittest.skip`` until the CP refactor lands, and only the in-process parse
coverage runs today.

[Test Category] Parameter
[Test Target] --moe-dp-size;--moe-data-parallel-size
"""

import os
import unittest

import requests

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="full-4-npu-a3", nightly=True)

TP_SIZE = 4
MOE_DP_SIZE = 2
ATTN_CP_SIZE = 2
CP_STRATEGY = "zigzag"

SKIP_REASON = (
    "NPU prefill CP removed upstream (CP V1 deprecation, 85d39401c8); "
    "moe-dp-size>1 on Qwen3 MoE requires CP "
    "(arg_groups/parallel_hook.py forces --attn-cp-size == moe-dp-size, and "
    "the MoE-DP token split comes only from prefill CP, whose KV path raises "
    "in layers/utils/cp_utils.py; see also the skipped "
    "test/registered/npu/llm_models/test_npu_qwen3_30b_attn_cp.py); "
    "re-enable after the CP refactor"
)

NEEDLE = "XQ7-4421"
_FILLER = (
    "The survey team logged the depth of every borehole along the northern "
    "ridge and filed the readings with the regional archive. "
)
_NEEDLE_REPEATS = 20


def _needle_prompt() -> str:
    """A ~1.4k-token prompt whose answer sits between two halves of filler, so
    it is only recoverable from the whole sequence."""
    body = _FILLER * _NEEDLE_REPEATS
    return (
        f"{body}The vault code word is {NEEDLE}. {body}"
        "Question: What is the vault code word?\nAnswer:"
    )


class TestNpuMoeDpSizeParsing(CustomTestCase):
    """Both spellings of the MoE data parallelism option parse to the same
    field, at a value the default (1) cannot produce. Runs without a server
    and without NPUs: the functional launch cases live in
    ``TestNpuMoeDpSizeFunctional`` below.

    [Test Category] Parameter
    [Test Target] --moe-dp-size;--moe-data-parallel-size
    """

    def test_both_spellings_parse_to_the_same_field(self):
        """--moe-data-parallel-size is declared as the alias of --moe-dp-size,
        so the two spellings are one field. Parsed in-process, which is all
        that is observable while the functional coverage is skipped: the
        parsed value 2 differs from the field default 1, so a pass cannot
        come from the alias silently falling back."""
        for spelling in ("--moe-dp-size", "--moe-data-parallel-size"):
            with self.subTest(spelling=spelling):
                args = prepare_server_args(
                    ["--model-path", "dummy", spelling, str(MOE_DP_SIZE)]
                )
                self.assertEqual(args.moe_dp_size, MOE_DP_SIZE)


@unittest.skip(SKIP_REASON)
class TestNpuMoeDpSizeFunctional(CustomTestCase):
    """Verify the coupled tp=4 / moe_dp=2 / attn_cp=2 topology serves correct
    answers end to end. Skipped until NPU prefill CP is restored (see
    SKIP_REASON); the arguments below carry the corrected spelling
    (--enable-prefill-cp, which replaced the removed
    --enable-prefill-context-parallel) so re-enabling needs only the
    decorator lifted.

    The MoE-DP group is the one CP hands the sequence chunks to: at tp=4 the
    MoE-DP groups are the ranks {0,2} and {1,3}. An answer that sits in the
    middle of a long prompt therefore only comes back right when the CP
    prefill and the MoE token split both hold across those ranks. For a GQA
    model on Ascend that prefill path is the FIA one, and the non-FIA path
    raises NotImplementedError at the first prefill: a server that merely
    starts proves nothing here, hence the request-level assertion and the
    explicit ASCEND_USE_FIA=1.

    [Test Category] Parameter
    [Test Target] --moe-dp-size
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--tp-size",
                str(TP_SIZE),
                "--moe-dp-size",
                str(MOE_DP_SIZE),
                "--attn-cp-size",
                str(ATTN_CP_SIZE),
                "--enable-prefill-cp",
                "--cp-strategy",
                CP_STRATEGY,
                "--mem-fraction-static",
                "0.7",
                # PCP is documented for batch_size == 1, and the requests below
                # are driven one at a time.
                "--max-running-requests",
                "1",
                "--disable-cuda-graph",
            ],
            env={**os.environ, "ASCEND_USE_FIA": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            kill_process_tree(cls.process.pid)

    @classmethod
    def _generate(cls, text, max_new_tokens, timeout):
        return requests.post(
            f"{cls.base_url}/generate",
            json={
                "text": text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=timeout,
        )

    def test_resolved_moe_dp_topology(self):
        """moe_dp_size is reported as the configured non-default value, next to
        the attn_cp_size it is required to match."""
        info = requests.get(f"{self.base_url}/server_info", timeout=60).json()
        self.assertEqual(info["moe_dp_size"], MOE_DP_SIZE)
        self.assertEqual(info["attn_cp_size"], ATTN_CP_SIZE)
        self.assertEqual(info["tp_size"], TP_SIZE)
        self.assertIn(f"--moe-dp-size {MOE_DP_SIZE}", info["launch_command"])

    def test_answer_from_the_middle_of_a_long_prompt(self):
        """The CP-split prefill plus the MoE token split must reconstruct the
        full sequence for the answer to come back."""
        resp = self._generate(_needle_prompt(), max_new_tokens=32, timeout=300)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(NEEDLE, resp.text)
        self.assertIsNone(self.process.poll(), "Server crashed during test")

    def test_repeated_short_requests(self):
        for _ in range(3):
            resp = self._generate("The capital of France is", 16, timeout=120)
            self.assertEqual(resp.status_code, 200)
            self.assertIn("Paris", resp.text)
        self.assertIsNone(self.process.poll(), "Server crashed during test")


if __name__ == "__main__":
    unittest.main()
