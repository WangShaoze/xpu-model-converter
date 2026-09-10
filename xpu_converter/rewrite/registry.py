# -*- coding: utf-8 -*-
"""改写规则注册表。"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from xpu_converter.ir.graph import Graph
from xpu_converter.logging_utils import get_logger
from xpu_converter.rewrite.base import RewriteRule

logger = get_logger(__name__)


@dataclass
class RewriteResult:
    """改写执行结果。"""

    applied_rules: List[str] = field(default_factory=list)
    applied_count: int = 0
    notes: List[str] = field(default_factory=list)
    per_rule: Dict[str, int] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.applied_count > 0

    def to_dict(self) -> dict:
        return {
            "changed": self.changed,
            "applied_count": self.applied_count,
            "applied_rules": list(self.applied_rules),
            "per_rule": dict(self.per_rule),
            "notes": list(self.notes),
        }


class RewriteRegistry:
    """按优先级串行执行改写规则。"""

    def __init__(self, rules: Optional[Sequence[RewriteRule]] = None):
        self._rules: List[RewriteRule] = list(rules or [])

    # ------------------------------------------------------------- 注册
    def register(self, rule: RewriteRule) -> RewriteRule:
        self._rules.append(rule)
        return rule

    @property
    def rules(self) -> List[RewriteRule]:
        return sorted(self._rules, key=lambda item: (item.priority, item.name))

    def rule_names(self) -> List[str]:
        return [item.name for item in self.rules]

    def rule_for(self, op_type: str) -> Optional[str]:
        for item in self.rules:
            if op_type in item.op_types:
                return item.name
        return None

    # ------------------------------------------------------------- 执行
    def apply(self, graph: Graph, only: Optional[Sequence[str]] = None) -> RewriteResult:
        result = RewriteResult()
        if not graph.has_raw:
            result.notes.append("图无原始 ModelProto, 跳过改写")
            return result

        selected = set(only) if only else None
        for rule in self.rules:
            if selected is not None and rule.name not in selected:
                continue
            if not rule.applicable(graph):
                continue
            count = 0
            handled = set()
            # 每次重新从 raw 取候选节点 (规则会增删节点), 用稳定 key 防止重复处理
            for op_type in rule.op_types:
                while True:
                    target = None
                    for item in graph.raw.graph.node:
                        if item.op_type != op_type or not rule.matches(item):
                            continue
                        if rule.node_key(item) in handled:
                            continue
                        target = item
                        break
                    if target is None:
                        break
                    handled.add(rule.node_key(target))
                    try:
                        if rule.rewrite(graph, target):
                            count += 1
                    except Exception as err:
                        rule.notes.append("规则 {} 处理 {} 失败: {}".format(rule.name, target.name, err))
                        logger.warning("改写失败 rule=%s node=%s err=%s", rule.name, target.name, err)
            if count:
                result.applied_count += count
                result.applied_rules.append(rule.name)
                result.per_rule[rule.name] = count
            result.notes.extend(rule.notes)
            rule.notes = []

        if result.changed:
            graph.refresh()
        return result


def default_registry() -> RewriteRegistry:
    """构建内置改写规则集。"""
    from xpu_converter.rewrite.mish import MishRewrite
    from xpu_converter.rewrite.nms import NMSRewrite
    from xpu_converter.rewrite.resize import ResizeRewrite
    from xpu_converter.rewrite.silu import SiluRewrite

    registry = RewriteRegistry()
    for rule in (SiluRewrite(), MishRewrite(), ResizeRewrite(), NMSRewrite()):
        registry.register(rule)
    return registry
