# -*- coding: utf-8 -*-
"""优化流水线。"""
from dataclasses import dataclass, field
from typing import List, Optional

from xpu_converter.ir.graph import Graph
from xpu_converter.optimizer.base import GraphPass, PassResult
from xpu_converter.optimizer.constant_fold import ConstantFoldPass
from xpu_converter.optimizer.fusion import ConvBNFusionPass
from xpu_converter.optimizer.graph_simplify import GraphSimplifyPass
from xpu_converter.optimizer.shape_inference import ShapeInferencePass


@dataclass
class OptimizationReport:
    passes: List[PassResult] = field(default_factory=list)

    @property
    def changed_any(self) -> bool:
        return any(item.changed for item in self.passes)

    def summary(self) -> str:
        return "\n".join(item.summary() for item in self.passes)

    def to_dict(self) -> dict:
        return {
            "changed": self.changed_any,
            "passes": [
                {"name": p.name, "changed": p.changed, "details": p.details, "skipped": p.skipped}
                for p in self.passes
            ],
        }


class OptimizationPipeline:
    """按顺序执行一组 :class:`GraphPass`。"""

    def __init__(self, passes: Optional[List[GraphPass]] = None):
        self.passes: List[GraphPass] = list(passes or [])

    @classmethod
    def default(cls, level: int = 2) -> "OptimizationPipeline":
        level = int(level or 0)
        if level <= 0:
            return cls([])
        if level == 1:
            return cls([ShapeInferencePass(), GraphSimplifyPass()])
        return cls([ShapeInferencePass(), ConstantFoldPass(), ConvBNFusionPass(), GraphSimplifyPass()])

    def add(self, graph_pass: GraphPass) -> "OptimizationPipeline":
        self.passes.append(graph_pass)
        return self

    def run(self, graph: Graph) -> OptimizationReport:
        report = OptimizationReport()
        for graph_pass in self.passes:
            report.passes.append(graph_pass.run(graph))
        return report
