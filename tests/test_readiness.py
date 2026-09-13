# -*- coding: utf-8 -*-
"""P0-7 支持状态自动检查测试: readiness 检测 / stable 门禁 / 自动降级。"""
import unittest

from xpu_converter.registry.model_registry import STATUS_EXPERIMENTAL, STATUS_STABLE
from xpu_converter.registry.readiness import (
    ModelReadiness,
    ReadinessItem,
    assert_support_claims,
    check_model_readiness,
    effective_status,
)


class ReadinessTest(unittest.TestCase):
    def _unknown(self):
        return check_model_readiness("definitely-not-a-model", loaded_adapters=set())

    def test_structural_missing_for_unknown_model(self):
        readiness = self._unknown()
        self.assertFalse(readiness.structural_ok)
        self.assertIn("adapter", readiness.missing())
        self.assertIn("yaml", readiness.missing())

    def test_yolov10_stable_structural_ok(self):
        # yolov10 是 V1 唯一承诺交付模型: adapter+yaml+contract+runtime 应齐备
        readiness = check_model_readiness("yolov10", loaded_adapters={"yolov10"})
        self.assertTrue(readiness.structural_ok, readiness.missing())
        self.assertEqual(readiness.declared_status, STATUS_STABLE)

    def test_ocr_task_lacks_contract(self):
        # task=ocr 不在契约可确定性推断的任务集内 → contract 未就绪
        readiness = check_model_readiness("ppocr", loaded_adapters={"ppocr"})
        self.assertIn("contract", readiness.missing())

    def test_effective_status_downgrades_fake_stable(self):
        fake = ModelReadiness(
            model_type="fake",
            declared_status=STATUS_STABLE,
            items=[ReadinessItem("yaml", False, "缺 yaml")],
        )
        self.assertEqual(effective_status(fake), STATUS_EXPERIMENTAL)

    def test_effective_status_keeps_experimental_no_upgrade(self):
        # 结构就绪也不自动升 stable(stable 需人工+真机证据)
        ok = ModelReadiness(
            model_type="yolov8",
            declared_status=STATUS_EXPERIMENTAL,
            items=[ReadinessItem("adapter", True, ""),
                   ReadinessItem("yaml", True, ""),
                   ReadinessItem("contract", True, ""),
                   ReadinessItem("runtime", True, "")],
        )
        self.assertEqual(effective_status(ok), STATUS_EXPERIMENTAL)

    def test_assert_support_claims_passes_for_v1_stable(self):
        # 门禁: 声明 stable 的模型(当前仅 yolov10)必须结构就绪, 否则抛错
        summary = assert_support_claims(loaded_adapters={"yolov10"})
        self.assertTrue(summary)
        self.assertTrue(all(row["structural_ok"] for row in summary))


if __name__ == "__main__":
    unittest.main()