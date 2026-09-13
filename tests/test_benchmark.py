# -*- coding: utf-8 -*-
"""P0-10 性能必须绑定硬件指纹测试。"""
import unittest

import numpy as np

from xpu_converter.validator.benchmark import BenchmarkResult, BenchmarkRunner


class FakeRealSession:
    """模拟真实后端(非仿真/占位)。"""

    backend_name = "kunlun-xpu"
    input_names = ["images"]
    output_names = ["det"]

    def __init__(self, batch=1):
        self.batch = batch

    def run(self, feeds):
        return [np.zeros((self.batch, 6), np.float32)]

    def input_shapes(self):
        return {"images": [self.batch, 3, 640, 640]}


class FakeSimulatedSession(FakeRealSession):
    backend_name = "onnxruntime"


class BenchmarkResultTest(unittest.TestCase):
    def test_batch_serialized(self):
        result = BenchmarkResult(batch=4, chip="K200")
        self.assertEqual(result.to_dict()["batch"], 4)

    def test_credible_requires_chip(self):
        self.assertFalse(BenchmarkResult(available=True, chip="").credible)
        self.assertTrue(BenchmarkResult(available=True, chip="K200").credible)
        self.assertFalse(BenchmarkResult(available=False, chip="K200").credible)

    def test_fingerprint_stable_and_tracks_chip(self):
        a = BenchmarkResult(chip="K200", sdk_version="1.2", device="cuda:0")
        b = BenchmarkResult(chip="K200", sdk_version="1.2", device="cuda:0")
        c = BenchmarkResult(chip="K590", sdk_version="1.2", device="cuda:0")
        self.assertEqual(a.compute_fingerprint(), b.compute_fingerprint())
        self.assertNotEqual(a.compute_fingerprint(), c.compute_fingerprint())

    def test_to_dict_includes_required_fields(self):
        data = BenchmarkResult(batch=1, chip="K200").to_dict()
        for key in ("chip", "sdk_version", "driver_version", "firmware_version", "model",
                    "input_shape", "batch", "warmup", "iterations", "hardware_fingerprint"):
            self.assertIn(key, data)


class BenchmarkRunnerTest(unittest.TestCase):
    def test_simulated_backend_unavailable(self):
        result = BenchmarkRunner(iterations=2, warmup=0).run(FakeSimulatedSession())
        self.assertFalse(result.available)
        self.assertEqual(result.status, "NOT_AVAILABLE")

    def test_real_backend_without_chip_unavailable(self):
        # 真实后端但未提供硬件指纹(chip 为空)→ 性能无法溯源, 判 NOT_AVAILABLE(P0-10)
        result = BenchmarkRunner(iterations=2, warmup=0).run(FakeRealSession())
        self.assertFalse(result.available)
        self.assertEqual(result.status, "NOT_AVAILABLE")
        self.assertFalse(result.credible)

    def test_real_backend_with_chip_credible_and_batch(self):
        result = BenchmarkRunner(iterations=3, warmup=1).run(
            FakeRealSession(batch=2),
            hardware={"chip": "K200", "sdk_version": "1.2.0",
                      "driver_version": "3.4", "firmware_version": "a1"},
            input_shape=[2, 3, 640, 640],
            model="yolov8n.onnx",
            precision="fp16",
        )
        self.assertTrue(result.available)
        self.assertTrue(result.credible)
        self.assertEqual(result.batch, 2)
        self.assertEqual(result.chip, "K200")
        self.assertTrue(result.hardware_fingerprint)
        self.assertGreater(result.latency_ms.get("mean", 0.0), 0.0)

    def test_batch_from_sample_when_input_shape_empty(self):
        sample = {"images": np.zeros((7, 3, 32, 32), np.float32)}
        result = BenchmarkRunner(iterations=2, warmup=0).run(
            FakeRealSession(batch=7), sample=sample, hardware={"chip": "K200"}
        )
        self.assertEqual(result.batch, 7)


if __name__ == "__main__":
    unittest.main()