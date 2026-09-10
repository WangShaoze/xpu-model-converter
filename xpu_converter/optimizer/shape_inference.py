# -*- coding: utf-8 -*-
"""Shape 推导 Pass。

对 ONNX 图执行 ``onnx.shape_inference``, 补齐 ``value_info`` —— 后续算子分析
需要准确的中间张量 shape。
"""
from xpu_converter.ir import onnx as onnx_ir
from xpu_converter.ir.graph import Graph
from xpu_converter.optimizer.base import GraphPass, PassResult


class ShapeInferencePass(GraphPass):
    name = "shape_inference"

    def applicable(self, graph: Graph) -> bool:
        return graph.has_raw and onnx_ir.available()

    def run(self, graph: Graph) -> PassResult:
        if not self.applicable(graph):
            return PassResult(self.name, skipped="需要 onnx 依赖与原始 ModelProto")
        before = len(graph.value_info)
        ok = graph.infer_shapes()
        if not ok:
            return PassResult(self.name, skipped="onnx.shape_inference 执行失败")
        after = len(graph.value_info)
        return PassResult(
            self.name,
            changed=after > before,
            details={"value_info": after, "new": max(0, after - before)},
        )
