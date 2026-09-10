# -*- coding: utf-8 -*-
"""IR 计算图。

``Graph`` 是算子分析与图改写的主要载体。当图来自 ONNX 时, 原始 ``ModelProto``
会挂在 :attr:`Graph.raw` 上——需要访问张量真实取值 (常量折叠 / Conv+BN 融合) 的
Pass 直接操作 ``raw``, 之后调用 :meth:`Graph.refresh` 重建轻量视图。
"""
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from xpu_converter.ir.node import Node
from xpu_converter.ir.tensor import Tensor


@dataclass
class Graph:
    """与框架无关的计算图。"""

    name: str = "graph"
    nodes: List[Node] = field(default_factory=list)
    inputs: List[Tensor] = field(default_factory=list)
    outputs: List[Tensor] = field(default_factory=list)
    initializers: Dict[str, Tensor] = field(default_factory=dict)
    value_info: Dict[str, Tensor] = field(default_factory=dict)
    opset: int = 13
    # 原始 ONNX ModelProto (来自前端导出时存在); 值级操作依赖它
    raw: Any = None

    # ------------------------------------------------------------------ 查询
    @property
    def has_raw(self) -> bool:
        return self.raw is not None

    @property
    def total_nodes(self) -> int:
        return len(self.nodes)

    def node(self, name: str) -> Optional[Node]:
        for item in self.nodes:
            if item.name == name:
                return item
        return None

    def find_nodes(self, op_type: Optional[str] = None, domain: Optional[str] = None) -> List[Node]:
        result = []
        for item in self.nodes:
            if op_type is not None and item.op_type != op_type:
                continue
            if domain is not None and item.domain != domain:
                continue
            result.append(item)
        return result

    def producers(self) -> Dict[str, Node]:
        """输出名 -> 生产该输出的节点。"""
        mapping: Dict[str, Node] = {}
        for item in self.nodes:
            for out in item.outputs:
                mapping[out] = item
        return mapping

    def consumers(self) -> Dict[str, List[Node]]:
        """输入名 -> 消费该输入的节点列表。"""
        mapping: Dict[str, List[Node]] = {}
        for item in self.nodes:
            for inp in item.inputs:
                mapping.setdefault(inp, []).append(item)
        return mapping

    def op_type_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in self.nodes:
            counts[item.key] = counts.get(item.key, 0) + 1
        return counts

    def tensor(self, name: str) -> Optional[Tensor]:
        for collection in (self.inputs, self.outputs):
            for item in collection:
                if item.name == name:
                    return item
        if name in self.initializers:
            return self.initializers[name]
        return self.value_info.get(name)

    def topological_order(self) -> List[Node]:
        """返回拓扑序节点 (输入先于消费者)。"""
        produced = set(t.name for t in self.inputs) | set(self.initializers.keys())
        remaining = list(self.nodes)
        ordered: List[Node] = []
        while remaining:
            progressed = False
            for item in list(remaining):
                if all((inp == "" or inp in produced) for inp in item.inputs):
                    ordered.append(item)
                    remaining.remove(item)
                    produced.update(item.outputs)
                    progressed = True
            if not progressed:  # 存在环或未声明输入, 保底返回剩余节点
                ordered.extend(remaining)
                break
        return ordered

    # ------------------------------------------------------------------ 修改
    def add_node(self, node: Node) -> Node:
        self.nodes.append(node)
        return node

    def remove_node(self, name: str) -> Optional[Node]:
        for index, item in enumerate(self.nodes):
            if item.name == name:
                return self.nodes.pop(index)
        return None

    def rename_io(self, old: str, new: str) -> int:
        """把所有节点中出现的张量名 old 替换为 new, 返回替换次数。"""
        count = 0
        for item in self.nodes:
            new_inputs = [new if x == old else x for x in item.inputs]
            new_outputs = [new if x == old else x for x in item.outputs]
            count += sum(1 for a, b in zip(item.inputs, new_inputs) if a != b)
            count += sum(1 for a, b in zip(item.outputs, new_outputs) if a != b)
            item.inputs = new_inputs
            item.outputs = new_outputs
        return count

    # ------------------------------------------------------------------ 序列化
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "opset": self.opset,
            "nodes": [n.to_dict() for n in self.nodes],
            "inputs": [t.to_dict() for t in self.inputs],
            "outputs": [t.to_dict() for t in self.outputs],
            "initializers": {k: v.to_dict() for k, v in self.initializers.items()},
            "value_info": {k: v.to_dict() for k, v in self.value_info.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Graph":
        return cls(
            name=str(data.get("name", "graph")),
            opset=int(data.get("opset", 13)),
            nodes=[Node.from_dict(n) for n in data.get("nodes") or []],
            inputs=[Tensor.from_dict(t) for t in data.get("inputs") or []],
            outputs=[Tensor.from_dict(t) for t in data.get("outputs") or []],
            initializers={k: Tensor.from_dict(v) for k, v in (data.get("initializers") or {}).items()},
            value_info={k: Tensor.from_dict(v) for k, v in (data.get("value_info") or {}).items()},
        )

    def copy(self) -> "Graph":
        cloned = Graph(
            name=self.name,
            nodes=[Node.from_dict(n.to_dict()) for n in self.nodes],
            inputs=[Tensor.from_dict(t.to_dict()) for t in self.inputs],
            outputs=[Tensor.from_dict(t.to_dict()) for t in self.outputs],
            initializers={k: Tensor.from_dict(v.to_dict()) for k, v in self.initializers.items()},
            value_info={k: Tensor.from_dict(v.to_dict()) for k, v in self.value_info.items()},
            opset=self.opset,
        )
        cloned.raw = copy.deepcopy(self.raw)
        return cloned

    # ------------------------------------------------------------------ 工具
    def infer_shapes(self) -> bool:
        """对 raw 执行 ONNX shape inference 并刷新视图。"""
        from xpu_converter.ir import onnx as onnx_ir

        return onnx_ir.infer_shapes(self)

    def refresh(self) -> "Graph":
        """依据 :attr:`raw` 重建轻量视图 (节点/输入/输出/初始化器)。"""
        from xpu_converter.ir import onnx as onnx_ir

        if self.raw is None:
            return self
        rebuilt = onnx_ir.from_model(self.raw, name=self.name)
        self.nodes = rebuilt.nodes
        self.inputs = rebuilt.inputs
        self.outputs = rebuilt.outputs
        self.initializers = rebuilt.initializers
        self.value_info = rebuilt.value_info
        self.opset = rebuilt.opset
        return self

    def summary(self) -> str:
        counts = self.op_type_counts()
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
        lines = [
            "graph     : {}".format(self.name),
            "opset     : {}".format(self.opset),
            "nodes     : {}".format(len(self.nodes)),
            "inputs    : {}".format(", ".join(str(t) for t in self.inputs) or "-"),
            "outputs   : {}".format(", ".join(str(t) for t in self.outputs) or "-"),
            "initializers: {}".format(len(self.initializers)),
            "operators : {}".format(len(counts)),
            "top ops   : {}".format(", ".join("{}x{}".format(op, num) for op, num in top) or "-"),
        ]
        return "\n".join(lines)
