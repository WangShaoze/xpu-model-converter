# -*- coding: utf-8 -*-
"""昆仑芯算子注册表(Capability-driven)。

历史实现把支持情况简化成 ``Set[str]``, 无法表达"``Resize`` 只支持 nearest"
这类约束(ChatGPT 修改意见 §7)。现在本模块只是
:class:`xpu_converter.capability.operators.OperatorCapabilitySet` 的薄封装:
能力数据来自 ``configs/hardware/<hardware>_capabilities.yaml``, 判定时带上
dtype / 属性 / opset。
"""
from typing import Any, Dict, Iterable, List, Optional, Set

from xpu_converter.backend.base import OperatorAnalysis, OperatorAnalysisResult
from xpu_converter.capability.operators import (
    REWRITTEN,
    SUPPORTED,
    UNSUPPORTED,
    OperatorCapability,
    OperatorCapabilitySet,
)
from xpu_converter.paths import hardware_capabilities_path

# 自定义域算子(如 com.microsoft / com.baidu)默认视为不支持, 需显式登记
DEFAULT_DOMAINS = ("", "ai.onnx", "ai.onnx.ml")

# 兼容旧接口: 若能力表文件缺失, 用这里的基线算子名构造能力集
FALLBACK_NATIVE_OPS: Set[str] = {
    "Conv", "ConvTranspose", "MaxPool", "AveragePool", "GlobalAveragePool",
    "GlobalMaxPool", "BatchNormalization", "InstanceNormalization", "GroupNormalization",
    "LayerNormalization", "LRN", "Relu", "LeakyRelu", "PRelu", "Sigmoid", "Tanh",
    "HardSigmoid", "HardSwish", "Elu", "Selu", "Softplus", "Softsign", "Gelu", "Erf",
    "Clip", "Gemm", "MatMul", "Einsum", "Add", "Sub", "Mul", "Div", "Pow", "Sqrt",
    "Exp", "Log", "Abs", "Neg", "Floor", "Ceil", "Round", "Reciprocal", "Sign",
    "Max", "Min", "Sum", "Mean", "Mod", "CumSum", "Equal", "Greater", "GreaterOrEqual",
    "Less", "LessOrEqual", "And", "Or", "Not", "Xor", "Reshape", "Squeeze", "Unsqueeze",
    "Transpose", "Flatten", "Expand", "Tile", "Pad", "Concat", "Split", "Slice",
    "DepthToSpace", "SpaceToDepth", "Identity", "Constant", "ConstantOfShape", "Cast",
    "CastLike", "Resize", "Upsample", "Where", "Range", "Compress", "Gather",
    "GatherElements", "GatherND", "ScatterElements", "ScatterND", "Shape", "Size",
    "NonZero", "TopK", "ArgMax", "ArgMin", "OneHot", "ReduceMax", "ReduceMin",
    "ReduceMean", "ReduceSum", "ReduceProd", "Softmax", "LogSoftmax", "Dropout",
    "LSTM", "GRU", "RNN",
}

FALLBACK_REWRITE_OPS: Dict[str, str] = {
    "Silu": "silu",
    "Mish": "mish",
    "Upsample": "resize",
    "NonMaxSuppression": "nms",
}

# 向后兼容别名(旧代码/测试引用过这两个名字)
NATIVE_OPS: Set[str] = FALLBACK_NATIVE_OPS
REWRITE_OPS: Dict[str, str] = FALLBACK_REWRITE_OPS


def load_capability_set(hardware: str = "kunlun") -> OperatorCapabilitySet:
    """读取硬件算子能力表; 文件缺失时回退到内置基线表。"""
    path = hardware_capabilities_path(hardware)
    if path.is_file():
        return OperatorCapabilitySet.from_yaml(path)
    return OperatorCapabilitySet(
        operators={name: OperatorCapability(name) for name in FALLBACK_NATIVE_OPS},
        rewrite_ops=FALLBACK_REWRITE_OPS,
        domains=DEFAULT_DOMAINS,
    )


class KunlunOperatorRegistry:
    """昆仑芯算子注册表(基于 OperatorCapabilitySet)。"""

    def __init__(
        self,
        capability_set: Optional[OperatorCapabilitySet] = None,
        operators: Optional[Iterable[str]] = None,
        rewrite_ops: Optional[Dict[str, str]] = None,
        domains: Optional[Iterable[str]] = None,
    ) -> None:
        caps = capability_set or OperatorCapabilitySet(
            operators={name: OperatorCapability(name) for name in FALLBACK_NATIVE_OPS},
            rewrite_ops=FALLBACK_REWRITE_OPS,
            domains=DEFAULT_DOMAINS,
        )
        if operators or rewrite_ops:
            caps = caps.with_extra(operators, rewrite_ops)
        self.capabilities = caps
        if domains:
            self.capabilities.domains = set(domains)

    # ---- 兼容属性 ----
    @property
    def native_ops(self) -> Set[str]:
        return set(self.capabilities.operators)

    @property
    def rewrite_ops(self) -> Dict[str, str]:
        return dict(self.capabilities.rewrite_ops)

    # ---- 注册 ----
    def with_extra(self, native_ops: Optional[Iterable[str]] = None,
                   rewrite_ops: Optional[Dict[str, str]] = None) -> "KunlunOperatorRegistry":
        return KunlunOperatorRegistry(self.capabilities.with_extra(native_ops, rewrite_ops))

    def register(self, op_type: str) -> None:
        self.capabilities.operators.setdefault(op_type, OperatorCapability(op_type))

    # ---- 分类 ----
    def classify(self, op_type: str, domain: str = "", dtype: Optional[str] = None,
                 attributes: Optional[Dict[str, Any]] = None,
                 opset: Optional[int] = None) -> str:
        """返回 ``supported`` / ``rewritten`` / ``unsupported``。"""
        status, _reason = self.capabilities.classify(
            op_type, domain, dtype=dtype, attributes=attributes, opset=opset
        )
        return status

    def classify_with_reason(self, op_type: str, domain: str = "", dtype: Optional[str] = None,
                             attributes: Optional[Dict[str, Any]] = None,
                             opset: Optional[int] = None):
        return self.capabilities.classify(
            op_type, domain, dtype=dtype, attributes=attributes, opset=opset
        )

    # ---- 分析 ----
    def analyze(self, graph, precision: str = "fp32") -> OperatorAnalysis:
        """统计图中算子的支持情况, ``graph`` 可为 IR Graph 或原始 ModelProto。"""
        from xpu_converter.capability.matrix import build_report

        report = build_report(graph, self.capabilities, precision=precision)
        result = OperatorAnalysis(
            supported=report.supported,
            rewritten=report.rewritten,
            unsupported=report.unsupported,
            op_counts=dict(report.op_counts),
            unsupported_ops=list(report.unsupported_ops),
            reasons=list(report.reasons),
        )
        for op_type in result.op_counts:
            if op_type in self.capabilities.rewrite_ops:
                result.rewrite_ops[op_type] = self.capabilities.rewrite_ops[op_type]
        for op_type, domain in self._iter_ops(graph):
            if domain and domain not in self.capabilities.domains and domain not in result.domain_ops:
                result.domain_ops.append(domain)
        return result

    def describe(self) -> Dict[str, Any]:
        return {
            "native_op_count": len(self.capabilities.operators),
            "rewrite_ops": dict(self.capabilities.rewrite_ops),
            "capability": self.capabilities.describe(),
        }

    @staticmethod
    def _iter_ops(graph):
        """兼容 IR Graph 与原始 ONNX ModelProto 的算子遍历。"""
        raw = getattr(graph, "graph", None)
        if raw is not None and hasattr(raw, "node"):  # 原始 ModelProto
            for node in raw.node:
                yield node.op_type, (node.domain or "")
            return
        for node in getattr(graph, "nodes", []) or []:
            yield getattr(node, "op_type", ""), getattr(node, "domain", "")


def default_registry(target_chip: str = "auto", hardware: str = "kunlun") -> KunlunOperatorRegistry:
    """按目标芯片返回算子注册表(ChatGPT 修改意见 §33 的 Capability Matrix 数据层)。"""
    registry = KunlunOperatorRegistry(load_capability_set(hardware))
    registry.capabilities.target_chip = target_chip
    return registry


def collect_unique_ops(graph) -> List[str]:
    """按首次出现顺序返回图中的去重算子名。"""
    seen: List[str] = []
    for op_type, _domain in KunlunOperatorRegistry._iter_ops(graph):
        if op_type not in seen:
            seen.append(op_type)
    return seen
