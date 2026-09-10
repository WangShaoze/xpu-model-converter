# -*- coding: utf-8 -*-
"""Mish 改写: ``Mish(x) = x * tanh(softplus(x))`` -> ``Softplus`` + ``Tanh`` + ``Mul``。"""
from xpu_converter.ir.graph import Graph
from xpu_converter.rewrite.base import RewriteRule


class MishRewrite(RewriteRule):
    name = "mish_to_softplus_tanh_mul"
    op_types = ("Mish",)
    priority = 21

    def rewrite(self, graph: Graph, node) -> bool:
        if len(node.input) != 1 or len(node.output) != 1:
            return False
        model = graph.raw
        x, y = node.input[0], node.output[0]
        base = node.name or "mish"
        softplus_out = self.unique_name(model, "{}/softplus_out".format(base))
        tanh_out = self.unique_name(model, "{}/tanh_out".format(base))
        softplus = self.make_node(
            "Softplus", [x], [softplus_out], name=self.unique_name(model, "{}/Softplus".format(base))
        )
        tanh = self.make_node(
            "Tanh", [softplus_out], [tanh_out], name=self.unique_name(model, "{}/Tanh".format(base))
        )
        mul = self.make_node("Mul", [x, tanh_out], [y], name=self.unique_name(model, "{}/Mul".format(base)))
        self.replace_node(model, node, [softplus, tanh, mul])
        self.notes.append("Mish -> Softplus+Tanh+Mul: {}".format(y))
        return True
