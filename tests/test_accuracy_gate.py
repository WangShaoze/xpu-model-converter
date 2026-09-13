# -*- coding: utf-8 -*-
"""P0-1(禁止 Validator 自动 CPU fallback)与 P1-4(执行溯源)测试。

核心断言: 目标会话只有实际运行在真实昆仑 XPU(actual_device == "xpu")上时,
精度结论才是 PASS; 一旦回退到 CPU, status 必须判 NOT_AVAILABLE, 而不是被当作
"数值一致即验收通过"。
"""
import unittest

import numpy as np

from xpu_converter.validator.accuracy import AccuracyValidator


class _FakeSession:
    """最小可用会话桩: 参考/目标都输出全零, 保证数值层 OK。"""

    backend_name = "fake"
    input_names = ["x"]
    output_names = ["y"]

    def __init__(self, actual_device="cpu", execution_mode="cpu", device_available=False,
                 backend_name="fake"):
        self.actual_device = actual_device
        self.execution_mode = execution_mode
        self.device_available = device_available
        self.backend_name = backend_name

    def run(self, feeds):
        return [np.zeros((1, 4), np.float32)]

    def input_shapes(self):
        return {"x": [1, 4]}


def _sample():
    return {"x": np.zeros((1, 4), np.float32)}


class AccuracyGpuGateTest(unittest.TestCase):
    def test_target_on_xpu_passes(self):
        ref = _FakeSession(actual_device="cpu", execution_mode="simulated", backend_name="onnxruntime")
        target = _FakeSession(actual_device="xpu", execution_mode="hardware", device_available=True)
        # 若 target 真实跑在 XPU 上, 数值一致才判 PASS
        report = AccuracyValidator(fail_on_mismatch=False).validate(ref, target, [_sample()])
        self.assertTrue(report.available)
        self.assertEqual(report.status, "PASS")

    def test_auto_fallback_to_cpu_is_not_available(self):
        # P0-1 反例: DEVICE=auto, 本机无卡回退 CPU → 绝不判 PASS
        ref = _FakeSession(actual_device="cpu", execution_mode="simulated", backend_name="onnxruntime")
        target = _FakeSession(actual_device="cpu", execution_mode="fallback_cpu",
                              device_available=False, backend_name="paddle-xpu")
        report = AccuracyValidator(fail_on_mismatch=False).validate(ref, target, [_sample()])
        self.assertFalse(report.available)
        self.assertEqual(report.status, "NOT_AVAILABLE")

    def test_cpu_fallback_does_not_raise_validation_error(self):
        # 不可用不是"未通过", 不应抛 ValidationError
        ref = _FakeSession(actual_device="cpu", execution_mode="simulated")
        target = _FakeSession(actual_device="cpu", execution_mode="fallback_cpu")
        report = AccuracyValidator(fail_on_mismatch=True).validate(ref, target, [_sample()])
        self.assertEqual(report.status, "NOT_AVAILABLE")

    def test_report_carries_execution_provenance(self):
        # P1-4: to_dict 必须带上 reference/target 执行溯源
        ref = _FakeSession(actual_device="cpu", execution_mode="simulated")
        target = _FakeSession(actual_device="xpu", execution_mode="hardware", device_available=True,
                              backend_name="paddle-xpu")
        report = AccuracyValidator(fail_on_mismatch=False).validate(ref, target, [_sample()])
        data = report.to_dict()
        self.assertIn("execution", data)
        self.assertEqual(data["execution"]["target_actual_device"], "xpu")
        self.assertEqual(data["execution"]["target"]["backend"], "paddle-xpu")
        self.assertEqual(data["status"], "PASS")


if __name__ == "__main__":
    unittest.main()