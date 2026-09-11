# -*- coding: utf-8 -*-
"""Capability Matrix: 把"最终图 + 硬件能力 + 算子能力"合成一次编译前门禁。

这是 ChatGPT 修改意见 §9 / §10 / §32 / §39 中 ``Final Capability Check`` 的实现:
只有 PASS 才允许进入 compile, 否则直接 STOP。
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from xpu_converter.capability.hardware import HardwareCapability
from xpu_converter.capability.operators import (
    FLOAT_DTYPES,
    REWRITTEN,
    SUPPORTED,
    UNSUPPORTED,
    OperatorCapabilitySet,
)


@dataclass
class OperatorDecision:
    """单个算子的判定结果。"""

    op_type: str
    domain: str = ""
    status: str = SUPPORTED
    reason: str = ""
    dtype: str = ""
    count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_type": self.op_type,
            "domain": self.domain,
            "status": self.status,
            "reason": self.reason,
            "dtype": self.dtype,
            "count": self.count,
        }


@dataclass
class CapabilityReport:
    """编译前能力门禁结果。"""

    ok: bool = True
    hardware: Optional[HardwareCapability] = None
    precision: str = "fp16"
    op_counts: Dict[str, int] = field(default_factory=dict)
    supported: int = 0
    rewritten: int = 0
    unsupported: int = 0
    unsupported_ops: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    decisions: List[OperatorDecision] = field(default_factory=list)
    operator_capability_version: int = 1

    def summary(self) -> str:
        return "{} supported / {} rewritten / {} unsupported".format(
            self.supported, self.rewritten, self.unsupported
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "precision": self.precision,
            "op_counts": dict(self.op_counts),
            "supported": self.supported,
            "rewritten": self.rewritten,
            "unsupported": self.unsupported,
            "unsupported_ops": list(self.unsupported_ops),
            "reasons": list(self.reasons),
            "operator_capability_version": self.operator_capability_version,
            "hardware": self.hardware.to_dict() if self.hardware else {},
        }


def _build_dtype_map(graph) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for tensor in getattr(graph, "inputs", []) or []:
        mapping[tensor.name] = tensor.dtype
    for name, tensor in (getattr(graph, "initializers", {}) or {}).items():
        mapping.setdefault(name, tensor.dtype)
    for name, tensor in (getattr(graph, "value_info", {}) or {}).items():
        mapping.setdefault(name, tensor.dtype)
    for tensor in getattr(graph, "outputs", []) or []:
        mapping.setdefault(tensor.name, tensor.dtype)
    return mapping


def _resolve_dtype(node, dtype_map: Dict[str, str], precision_dtype: str) -> str:
    """推断节点参与计算的 dtype。

    图由 fp32 ONNX 导出、编译目标却可能是 fp16, 因此只要节点涉及浮点张量,
    就按编译精度(precision_dtype)判定; 纯整型算子(Shape/Gather...)则用实际 dtype。
    """
    observed = [dtype_map.get(name) for name in node.inputs]
    observed = [dtype for dtype in observed if dtype]
    floats = [dtype for dtype in observed if dtype in FLOAT_DTYPES]
    if floats:
        return precision_dtype
    return observed[0] if observed else ""


def build_report(
    graph,
    capability_set: OperatorCapabilitySet,
    precision: str = "fp16",
    hardware: Optional[HardwareCapability] = None,
    opset: Optional[int] = None,
) -> CapabilityReport:
    """遍历最终图, 逐算子做能力判定。"""
    if getattr(graph, "nodes", None) is None and getattr(graph, "graph", None) is not None:
        # 兼容直接传入 ONNX ModelProto 的调用方
        from xpu_converter.ir import onnx as onnx_ir

        graph = onnx_ir.from_model(graph)
    dtype_map = _build_dtype_map(graph)
    precision_dtype = OperatorCapabilitySet.precision_dtype(precision)
    if opset is None:
        opset = getattr(graph, "opset", None)

    report = CapabilityReport(
        precision=precision,
        hardware=hardware,
        operator_capability_version=capability_set.version,
    )
    seen: Dict[Tuple[str, str], OperatorDecision] = {}
    for node in getattr(graph, "nodes", []) or []:
        op_type = getattr(node, "op_type", "")
        domain = getattr(node, "domain", "") or ""
        dtype = _resolve_dtype(node, dtype_map, precision_dtype)
        status, reason = capability_set.classify(
            op_type, domain, dtype=dtype, attributes=getattr(node, "attributes", None), opset=opset
        )
        report.op_counts[op_type] = report.op_counts.get(op_type, 0) + 1
        key = (op_type, domain)
        if key in seen:
            seen[key].count += 1
            continue
        decision = OperatorDecision(op_type=op_type, domain=domain, status=status, reason=reason, dtype=dtype)
        seen[key] = decision
        report.decisions.append(decision)

    for decision in report.decisions:
        if decision.status == SUPPORTED:
            report.supported += decision.count
        elif decision.status == REWRITTEN:
            report.rewritten += decision.count
        else:
            report.unsupported += decision.count
            if decision.op_type not in report.unsupported_ops:
                report.unsupported_ops.append(decision.op_type)
            report.reasons.append(
                "{}(x{}) {} -> UNSUPPORTED: {}".format(
                    decision.op_type, decision.count, decision.dtype or "-", decision.reason
                )
            )
    report.ok = report.unsupported == 0
    return report
