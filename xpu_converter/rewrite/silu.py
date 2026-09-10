# -*- coding: utf-8 -*-
"""Silu 改写: ``Silu(x) = x * sigmoid(x)`` -> ``Sigmoid`` + ``Mul``。"""
from xpu_converter.ir.graph import Graph
from xpu_converter.rewrite.base import RewriteRule


class SiluRewrite(RewriteRule):
    name = "silu_to_sigmoid_mul"
    op_types = ("Silu",)
    priority = 20

    def rewrite(self, graph: Graph, node) -> bool:
        if len(node.input) != 1 or len(node.output) != 1:
            return False
        model = graph.raw
        x, y = node.input[0], node.output[0]
        base = node.name or "silu"
        sig_out = self.unique_name(model, "{}/sigmoid_out".format(base))
        sigmoid = self.make_node(
            "Sigmoid", [x], [sig_out], name=self.unique_name(model, "{}/Sigmoid".format(base))
        )
        mul = self.make_node("Mul", [x, sig_out], [y], name=self.unique_name(model, "{}/Mul".format(base)))
        self.replace_node(model, node, [sigmoid, mul])
        self.notes.append("Silu -> Sigmoid+Mul: {}".format(y))
        return True
