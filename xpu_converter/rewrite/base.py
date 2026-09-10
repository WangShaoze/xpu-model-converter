# -*- coding: utf-8 -*-
"""改写规则公共接口。"""
from abc import ABC, abstractmethod
from typing import Any, List, Sequence, Tuple

from xpu_converter.ir import onnx as onnx_ir
from xpu_converter.ir.graph import Graph


class RewriteRule(ABC):
    """一条算子改写规则。

    子类通过 :attr:`op_types` 声明关注的算子, 在 :meth:`rewrite` 中直接修改
    ``graph.raw`` (保持输出名不变, 保证下游消费者无需改动)。
    """

    #: 规则名 (写入报告)
    name: str = "rule"
    #: 关注的 ONNX 算子类型
    op_types: Tuple[str, ...] = ()
    #: 执行优先级, 越小越先执行
    priority: int = 100

    def __init__(self):
        self.notes: List[str] = []

    # ------------------------------------------------------------- 匹配
    def applicable(self, graph: Graph) -> bool:
        return graph.has_raw and onnx_ir.available()

    def matches(self, node) -> bool:
        return node.op_type in self.op_types

    # ------------------------------------------------------------- 执行
    @abstractmethod
    def rewrite(self, graph: Graph, node) -> bool:
        """就地改写, 返回是否发生改写。"""

    # ------------------------------------------------------------- 工具
    @staticmethod
    def onnx():
        return onnx_ir.require_onnx()

    def make_node(self, op_type: str, inputs: Sequence[str], outputs: Sequence[str], name: str, **attrs):
        return self.onnx().helper.make_node(op_type, list(inputs), list(outputs), name=name, **attrs)

    @staticmethod
    def node_key(node) -> str:
        """节点稳定标识。

        protobuf 的 repeated field 每次访问都会产生新的 Python 包装对象,
        因此不能用 ``id()`` 做节点身份判断。
        """
        return "{}::{}".format(node.name or "", "|".join(node.output))

    @classmethod
    def node_index(cls, model, node) -> int:
        key = cls.node_key(node)
        for index, item in enumerate(model.graph.node):
            if cls.node_key(item) == key:
                return index
        raise KeyError("节点不存在于当前图中: {}".format(key))

    @classmethod
    def replace_node(cls, model, node, new_nodes: Sequence[Any]) -> None:
        """用 new_nodes 原位替换 node, 保持拓扑顺序。"""
        index = cls.node_index(model, node)
        del model.graph.node[index]
        for offset, item in enumerate(new_nodes):
            model.graph.node.insert(index + offset, item)

    @staticmethod
    def find_value_info(model, name: str):
        """在 input / value_info / output 中查找某个张量的类型信息。"""
        for collection in (model.graph.input, model.graph.value_info, model.graph.output):
            for vi in collection:
                if vi.name == name:
                    return vi
        return None

    @staticmethod
    def unique_name(model, prefix: str) -> str:
        existing = {node.name for node in model.graph.node}
        existing.update(init.name for init in model.graph.initializer)
        if prefix not in existing:
            return prefix
        index = 0
        while True:
            index += 1
            candidate = "{}_{}".format(prefix, index)
            if candidate not in existing:
                return candidate

    @staticmethod
    def consumers_of(model, tensor_name: str) -> List[Any]:
        return [node for node in model.graph.node if tensor_name in list(node.input)]
