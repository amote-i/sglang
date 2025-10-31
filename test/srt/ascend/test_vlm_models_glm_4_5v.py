import argparse
import glob
import json
import os
import random
import subprocess
import sys
import unittest
from types import SimpleNamespace

from sglang.test.test_vlm_utils import TestVLMModels


class TestGLM4Models(TestVLMModels):
    models = [
        SimpleNamespace(
            model="/root/.cache/modelscope/hub/models/ZhipuAI/GLM-4.5V",
            mmmu_accuracy=0.2,
        ),
    ]
    tp_size = 8
    mem_fraction_static = 0.7

    def test_vlm_mmmu_benchmark(self):
        self.vlm_mmmu_benchmark()


if __name__ == "__main__":
    unittest.main()
