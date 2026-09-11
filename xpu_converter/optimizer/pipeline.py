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
    """按顺序执行一组 :class:`GraphPass`, 即 Pass Manager(ChatGPT 修改意见 §35)。

    每个 Pass 执行前对图做一次快照; 若 Pass 声称修改了图(``changed=True``), 则对
    结果跑一遍 ONNX 合法性检查——不合法就回滚到快照, 绝不让一个坏图流入下游。
    数值等价性由流水线 Step 08 的三层 Accuracy Validation 统一把关(§34)。
    """

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
        from xpu_converter.ir import onnx as onnx_ir

        report = OptimizationReport()
        for graph_pass in self.passes:
            snapshot = graph.copy() if graph.has_raw else None
            nodes_before = len(graph.nodes)
            result = graph_pass.run(graph)
            result.details.setdefault("nodes_before", nodes_before)
            result.details["nodes_after"] = len(graph.nodes)
            if result.changed and snapshot is not None:
                problem = onnx_ir.check_model(graph.raw)
                result.details["graph_check"] = problem is None
                if problem:
                    graph.restore(snapshot)
                    result.changed = False
                    result.details["nodes_after"] = nodes_before
                    result.details["rolled_back"] = problem
                    result.skipped = "已回滚(图非法): {}".format(problem)
            report.passes.append(result)
        return report
