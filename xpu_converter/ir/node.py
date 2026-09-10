# -*- coding: utf-8 -*-
"""IR 计算节点。"""
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Node:
    """一个算子实例。"""

    name: str
    op_type: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    domain: str = ""

    @property
    def key(self) -> str:
        """含 domain 的算子标识, 用于算子分析。"""
        return "{}{}".format(self.domain + "::" if self.domain else "", self.op_type)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "op_type": self.op_type,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attributes": dict(self.attributes),
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Node":
        return cls(
            name=str(data.get("name", "")),
            op_type=str(data.get("op_type", "")),
            inputs=list(data.get("inputs") or []),
            outputs=list(data.get("outputs") or []),
            attributes=dict(data.get("attributes") or {}),
            domain=str(data.get("domain") or ""),
        )

    def __str__(self) -> str:
        return "{}[{}] {} -> {}".format(self.name, self.op_type, self.inputs, self.outputs)
