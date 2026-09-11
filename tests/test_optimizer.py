# -*- coding: utf-8 -*-
"""Optimizer 层测试。"""
import unittest

import numpy as np

from tests import _models
from tests import requires_onnx
from xpu_converter.ir import onnx as onnx_ir
from xpu_converter.optimizer.constant_fold import ConstantFoldPass
from xpu_converter.optimizer.fusion import ConvBNFusionPass
from xpu_converter.optimizer.pipeline import OptimizationPipeline


@requires_onnx
class TestConstantFold(unittest.TestCase):
    def test_folds_shape_gather_chain(self):
        model = _models.build_shape_fold()
        feeds = {"images": np.zeros((1, 3, 8, 16), dtype=np.float32)}
        expected = _models.run_onnx(model, feeds)

        graph = onnx_ir.from_model(model)
        result = ConstantFoldPass().run(graph)

        self.assertTrue(result.changed)
        self.assertEqual(result.details["folded"], 2)
        self.assertEqual(len(graph.find_nodes("Shape")), 0)
        self.assertEqual(len(graph.find_nodes("Gather")), 0)
        self.assertIn("hw", graph.initializers)
        self.assertIn("shape_out", graph.initializers)

        actual = _models.run_onnx(graph.raw, feeds)
        np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=0)

    def test_does_not_fold_graph_output(self):
        model = _models.build_shape_fold()
        graph = onnx_ir.from_model(model)
        ConstantFoldPass().run(graph)
        # Cast 的输出是对外输出, 必须保留
        self.assertEqual(len(graph.find_nodes("Cast")), 1)


@requires_onnx
class TestConvBNFusion(unittest.TestCase):
    def test_fusion_is_numerically_equivalent(self):
        model = _models.build_conv_bn_relu()
        feeds = {"images": np.random.RandomState(1).randn(1, 3, 8, 8).astype(np.float32)}
        expected = _models.run_onnx(model, feeds)

        graph = onnx_ir.from_model(model)
        result = ConvBNFusionPass().run(graph)

        self.assertTrue(result.changed)
        self.assertEqual(result.details["fused"], 1)
        self.assertEqual(len(graph.find_nodes("BatchNormalization")), 0)
        actual = _models.run_onnx(graph.raw, feeds)
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-4, atol=1e-5)

    def test_skips_when_weight_shared(self):
        model = _models.build_conv_bn_relu()
        onnx = onnx_ir.require_onnx()
        extra = onnx.helper.make_node("Conv", ["images", "conv_w"], ["second"], name="conv2", pads=[1, 1, 1, 1])
        model.graph.node.append(extra)
        model.graph.output.append(
            onnx.helper.make_tensor_value_info("second", onnx.TensorProto.FLOAT, [1, 4, 8, 8])
        )
        graph = onnx_ir.from_model(model)
        result = ConvBNFusionPass().run(graph)
        self.assertEqual(result.details["fused"], 0)


@requires_onnx
class TestPipeline(unittest.TestCase):
    def test_default_pipeline_runs_all_passes(self):
        graph = onnx_ir.from_model(_models.build_conv_bn_relu())
        report = OptimizationPipeline.default(2).run(graph)
        self.assertEqual(len(report.passes), 4)
        self.assertTrue(report.changed_any)
        self.assertEqual(len(graph.find_nodes("BatchNormalization")), 0)

    def test_level_zero_is_noop(self):
        graph = onnx_ir.from_model(_models.build_conv_bn_relu())
        before = len(graph.nodes)
        report = OptimizationPipeline.default(0).run(graph)
        self.assertEqual(len(report.passes), 0)
        self.assertEqual(len(graph.nodes), before)


if __name__ == "__main__":
    unittest.main()
