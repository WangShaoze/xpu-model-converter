# -*- coding: utf-8 -*-
"""P0-9 应用级精度指标测试: IoU / NMS / mAP / detection_evaluator。"""
import unittest

import numpy as np

from xpu_converter.validator.application import (
    decode_detections,
    detection_evaluator,
    eval_detections,
    iou_matrix,
    nms,
)


class IoUThresTest(unittest.TestCase):
    def test_iou_same_box_is_one(self):
        box = np.array([[10.0, 20.0, 110.0, 120.0]], np.float32)
        self.assertAlmostEqual(float(iou_matrix(box, box)[0, 0]), 1.0, places=5)

    def test_iou_no_overlap_zero(self):
        a = np.array([[0.0, 0.0, 10.0, 10.0]], np.float32)
        b = np.array([[20.0, 20.0, 30.0, 30.0]], np.float32)
        self.assertAlmostEqual(float(iou_matrix(a, b)[0, 0]), 0.0, places=5)

    def test_nms_keeps_non_overlapping(self):
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], np.float32)
        scores = np.array([0.9, 0.8], np.float32)
        keep = nms(boxes, scores, 0.5)
        self.assertEqual(sorted(keep.tolist()), [0, 1])

    def test_nms_drops_duplicate(self):
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], np.float32)
        scores = np.array([0.9, 0.8], np.float32)
        keep = nms(boxes, scores, 0.5)
        self.assertEqual(keep.tolist(), [0])


class MapTest(unittest.TestCase):
    def _rec(self):
        box = np.array([[10.0, 10.0, 40.0, 40.0],
                        [60.0, 60.0, 90.0, 90.0]], np.float32)
        return {
            "pred": box,
            "scores": np.array([0.9, 0.8], np.float32),
            "labels": np.array([0, 0], np.int64),
            "gt": box.copy(),
            "gt_labels": np.array([0, 0], np.int64),
        }

    def test_perfect_match_map1(self):
        records = [self._rec()]
        per_class, m = eval_detections(records, 0.5)
        self.assertAlmostEqual(m, 1.0, places=4)
        self.assertAlmostEqual(per_class[0]["ap"], 1.0, places=4)
        self.assertAlmostEqual(per_class[0]["precision"], 1.0, places=4)
        self.assertAlmostEqual(per_class[0]["recall"], 1.0, places=4)

    def test_empty_prediction_map0(self):
        records = [{
            "pred": np.zeros((0, 4), np.float32),
            "scores": np.zeros(0, np.float32),
            "labels": np.zeros(0, np.int64),
            "gt": np.array([[10, 10, 40, 40]], np.float32),
            "gt_labels": np.array([0], np.int64),
        }]
        _, m = eval_detections(records, 0.5)
        self.assertEqual(m, 0.0)

    def test_extra_false_positive_drops_map(self):
        # GT 2 个, 预测 1 个命中 + 2 个不匹配的 FP → recall=0.5, precision=1/3 → mAP 显著 < 0.9
        records = [{
            "pred": np.array([[10, 10, 40, 40],
                              [200, 200, 230, 230],
                              [300, 300, 330, 330]], np.float32),
            "scores": np.array([0.9, 0.8, 0.5], np.float32),
            "labels": np.array([0, 0, 0], np.int64),
            "gt": np.array([[10, 10, 40, 40], [60, 60, 90, 90]], np.float32),
            "gt_labels": np.array([0, 0], np.int64),
        }]
        _, m = eval_detections(records, 0.5)
        self.assertGreaterEqual(0.9, m)


class FakeSession:
    """返回固定原始输出的测试会话。"""

    def __init__(self, output):
        self.output = output
        self.input_names = ["images"]
        self.output_names = ["det"]
        self.backend_name = "fake"

    def run(self, _feeds):
        return [self.output]


class DetectionEvaluatorTest(unittest.TestCase):
    def test_exact_reconstruction_available_and_map1(self):
        # bnc6 原始输出: [x1,y1,x2,y2,conf,cls]
        box = np.array([[10.0, 10.0, 40.0, 40.0],
                        [60.0, 60.0, 90.0, 90.0]], np.float32)
        raw = np.concatenate([box, np.array([[0.9, 0.0], [0.8, 0.0]], np.float32)], axis=1)
        raw = raw[None, :, :]  # (1, N, 6)
        target = FakeSession(raw)
        gt = {"gt": box.copy(), "gt_labels": np.array([0, 0], np.int64)}

        evaluator = detection_evaluator(contract={"layout": "bnc6"}, ground_truth=[gt])
        report = evaluator(reference=None, target=target,
                           samples=[{"images": np.zeros((1, 3, 32, 32), np.float32)}],
                           num_classes_=1)
        self.assertTrue(report["available"])
        self.assertAlmostEqual(report["map50"], 1.0, places=3)
        self.assertLess(0.0, report["map"])

    def test_missing_gt_is_unavailable(self):
        evaluator = detection_evaluator(contract={"layout": "bnc6"}, ground_truth=[])
        report = evaluator(None, FakeSession(np.zeros((1, 0, 6), np.float32)), [{"images": None}], 1)
        self.assertFalse(report["available"])
        self.assertIn("GT", report["reason"])


class DecodeTest(unittest.TestCase):
    def test_decode_bnc6(self):
        raw = np.array([[[10, 10, 40, 40, 0.9, 2]]], np.float32)
        _, s, l = decode_detections([raw], "bnc6", 80, 0.25, 0.45)
        self.assertEqual(int(l[0]), 2)
        self.assertAlmostEqual(float(s[0]), 0.9, places=5)


if __name__ == "__main__":
    unittest.main()