# -*- coding: utf-8 -*-
"""Rewrite 层测试。"""
import unittest

import numpy as np

from tests import _models
from tests import requires_onnx
from xpu_converter.ir import onnx as onnx_ir
from xpu_converter.rewrite.registry import default_registry


@requires_onnx
class TestSiluRewrite(unittest.TestCase):
    def test_silu_decomposed_and_equivalent(self):
        """Silu 被拆成 Sigmoid+Mul 后, 计算结果应与 x*sigmoid(x) 一致。"""
        graph = onnx_ir.from_model(_models.build_silu_graph())

        result = default_registry().apply(graph, only=["silu_to_sigmoid_mul"])

        self.assertTrue(result.changed)
        self.assertEqual(len(graph.find_nodes("Silu")), 0)
        self.assertEqual(len(graph.find_nodes("Sigmoid")), 1)
        self.assertEqual(len(graph.find_nodes("Mul")), 1)

        x = np.random.RandomState(2).randn(1, 3, 4, 4).astype(np.float32)
        actual = _models.run_onnx(graph.raw, {"images": x})[0]
        expected = x * (1.0 / (1.0 + np.exp(-x)))
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


class TestNMSRewrite(unittest.TestCase):
    def test_nms_stripped_to_raw_outputs(self):
        model = _models.build_nms_graph()
        onnx = onnx_ir.require_onnx()
        graph = onnx_ir.from_model(model)

        result = default_registry().apply(graph, only=["nms_to_cpu"])

        self.assertTrue(result.changed)
        self.assertEqual(len(graph.find_nodes("NonMaxSuppression")), 0)
        self.assertEqual(len(graph.find_nodes("Gather")), 0)
        self.assertEqual([out.name for out in graph.outputs], ["boxes", "scores"])
        onnx.checker.check_model(graph.raw)

    def test_skips_multi_head_model(self):
        """存在 NMS 之外的输出头时必须跳过, 避免误删。"""
        model = _models.build_nms_graph(extra_head=True)
        graph = onnx_ir.from_model(model)

        result = default_registry().apply(graph, only=["nms_to_cpu"])

        self.assertFalse(result.changed)
        self.assertEqual(len(graph.find_nodes("NonMaxSuppression")), 1)
        self.assertTrue(any("输出头" in note for note in result.notes))


class TestRegistry(unittest.TestCase):
    def test_default_rules_registered(self):
        names = default_registry().rule_names()
        self.assertIn("silu_to_sigmoid_mul", names)
        self.assertIn("mish_to_softplus_tanh_mul", names)
        self.assertIn("resize_normalize", names)
        self.assertIn("nms_to_cpu", names)

    def test_no_raw_graph_is_skipped(self):
        from xpu_converter.ir.graph import Graph

        result = default_registry().apply(Graph(name="empty"))
        self.assertFalse(result.changed)
        self.assertTrue(result.notes)


if __name__ == "__main__":
    unittest.main()
