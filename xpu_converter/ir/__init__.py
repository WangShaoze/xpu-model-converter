# -*- coding: utf-8 -*-
"""IR 层: 与具体框架无关的中间表示 (Graph / Node / Tensor)。"""
from xpu_converter.ir.tensor import Tensor
from xpu_converter.ir.node import Node
from xpu_converter.ir.graph import Graph

__all__ = ["Tensor", "Node", "Graph"]
