# -*- coding: utf-8 -*-
"""优化 Pass 公共接口。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

from xpu_converter.ir.graph import Graph


@dataclass
class PassResult:
    """单个 Pass 的执行结果。"""

    name: str
    changed: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    skipped: str = ""

    def summary(self) -> str:
        if self.skipped:
            return "{}: skipped ({})".format(self.name, self.skipped)
        extra = ", ".join("{}={}".format(k, v) for k, v in self.details.items())
        return "{}: {}{}".format(self.name, "changed" if self.changed else "unchanged", " (" + extra + ")" if extra else "")


class GraphPass(ABC):
    """图优化 Pass 基类。"""

    name: str = "pass"

    def applicable(self, graph: Graph) -> bool:
        """当前图是否具备执行条件 (默认总是可以)。"""
        return True

    @abstractmethod
    def run(self, graph: Graph) -> PassResult:
        """就地修改 graph 并返回结果。"""
