# -*- coding: utf-8 -*-
"""P0-9 应用级精度指标测试: IoU / NMS / mAP / detection_evaluator。"""
import unittest

import numpy as np

from xpu_converter.validator.application import (
    build_evaluator,
    cls_evaluator,
    compute_oks,
    decode_cls,
    decode_detections,
    decode_pose,
    detection_evaluator,
    eval_detections,
    iou_matrix,
    nms,
    pose_evaluator,
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


class ClsEvaluatorTest(unittest.TestCase):
    def test_decode_cls_top1(self):
        raw = np.array([[[0.1, 0.2, 0.9, 0.3, 0.4]]], np.float32)  # argmax = 2
        label, topk, probs = decode_cls([raw], topk=3)
        self.assertEqual(label, 2)
        self.assertEqual(int(topk[0]), 2)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

    def test_cls_accuracy(self):
        # 3 样本, 前 2 命中 top-1, 第 3 命中 top-3 但非 top-1
        def session(i):
            out = np.zeros(5, np.float32)
            out[i % 5] = 1.0
            return FakeSession(out[None, :])

        gts = [{"gt_label": 0}, {"gt_label": 1}, {"gt_label": 2}]
        samples = [{"images": None}, {"images": None}, {"images": None}]
        evaluator = cls_evaluator(contract={"layout": "cls"}, ground_truth=gts, topk=3)
        report = evaluator(reference=None, target=session(0),
                           samples=samples, num_classes_=5)
        self.assertTrue(report["available"])
        self.assertEqual(report["samples"], 3)
        # target 每次返回 class 0/1/2 最高, 故 top-1 只中 1 个, top-3 中 3 个
        self.assertAlmostEqual(report["accuracy"], 1.0 / 3, places=4)
        self.assertAlmostEqual(report["top3_accuracy"], 1.0, places=4)

    def test_cls_missing_gt_unavailable(self):
        evaluator = cls_evaluator(contract={"layout": "cls"}, ground_truth=[])
        report = evaluator(None, FakeSession(np.zeros(3, np.float32)[None, :]),
                           [{"images": None}], 3)
        self.assertFalse(report["available"])
        self.assertIn("GT", report["reason"])


class PoseEvaluatorTest(unittest.TestCase):
    def _pose_output(self, kp_xy, nc=1, num_kpt=2, conf=0.9):
        # bnc 行: (N)=1 行, C 列 = [cx,cy,w,h, cls..., kpt_x,kpt_y,kpt_v,...]
        # 由于 decode_pose 的 bcn/bnc 用「行数<列数则转置」判别, 构造 N>C 的 (N,C) 数组
        box = np.array([50.0, 50.0, 100.0, 100.0], np.float32)  # cxcywh
        cls_part = np.full(nc, 0.0, np.float32)
        xs = np.asarray(kp_xy, np.float32)[:, 0]
        ys = np.asarray(kp_xy, np.float32)[:, 1]
        vis = np.ones(num_kpt, np.float32)
        kpt_part = np.stack([xs, ys, vis], axis=1).flatten()
        row = np.concatenate([box, cls_part, kpt_part]).astype(np.float32)
        row[4] = conf  # no-obj 布局下 cls 从 col4 开始, 置最高置信度
        # 复制到 N>C 行, 满足行优先判别
        return np.tile(row, (12, 1))

    def test_decode_pose(self):
        arr = self._pose_output([[50.0, 50.0], [60.0, 60.0]], nc=1, num_kpt=2)
        decoded = decode_pose([arr], num_classes=1)
        self.assertIsNotNone(decoded)
        box, score, kpts = decoded
        self.assertEqual(kpts.shape, (2, 3))
        self.assertEqual(int(box[0]), 0)  # cx-w/2 = 50 - 50 = 0
        self.assertGreater(score, 0.0)

    def test_oks_exact_is_one(self):
        gt = np.array([[50.0, 50.0, 1.0], [60.0, 60.0, 1.0]], np.float32)
        self.assertAlmostEqual(compute_oks(gt.copy(), gt, gt_scale=50.0), 1.0, places=5)

    def _oks(self, gt, num_kpt=2, nc=1):
        kp_xy = gt[:, :2]
        arr = self._pose_output(kp_xy, nc=nc, num_kpt=num_kpt)
        evaluator = pose_evaluator(contract={"layout": "pose", "num_classes": nc},
                                   ground_truth=[{"gt_keypoints": gt, "gt_scale": 50.0}])
        return evaluator(None, FakeSession([arr]), [{"images": None}], num_classes_=nc)

    def test_pose_perfect_match_oks1(self):
        gt = np.array([[50.0, 50.0, 1.0], [60.0, 60.0, 1.0]], np.float32)
        report = self._oks(gt)
        self.assertTrue(report["available"])
        self.assertAlmostEqual(report["oks"], 1.0, places=4)

    def test_pose_missing_gt_unavailable(self):
        evaluator = pose_evaluator(contract={"layout": "pose"}, ground_truth=[])
        report = evaluator(None, FakeSession([np.zeros((1, 11), np.float32)]),
                           [{"images": None}], 1)
        self.assertFalse(report["available"])
        self.assertIn("GT", report["reason"])


class BuildEvaluatorTest(unittest.TestCase):
    def test_build_cls(self):
        evaluator = build_evaluator("cls", contract={"layout": "cls"})
        self.assertTrue(callable(evaluator))

    def test_build_pose(self):
        evaluator = build_evaluator("pose", contract={"layout": "pose"})
        self.assertTrue(callable(evaluator))

    def test_build_seg_feels_unavailable_without_gt(self):
        evaluator = build_evaluator("segment", contract={"layout": "segment"}, ground_truth=[])
        report = evaluator(None, FakeSession(np.zeros((1, 0, 6), np.float32)),
                           [{"images": None}], 80)
        self.assertFalse(report["available"])


if __name__ == "__main__":
    unittest.main()