# -*- coding: utf-8 -*-
"""昆仑芯算子支持表与算子分析器。

本模块只维护"算子名 → 支持情况"这一层知识, 与后端 SDK 版本无关:

- ``NATIVE_OPS``   : 昆仑 XPU 原生支持的算子(可直接编译下发)
- ``REWRITE_OPS``  : 通过 :mod:`xpu_converter.rewrite` 改写后可支持的算子

真正的 SDK 相关差异(型号、XTCL 版本)通过 :meth:`KunlunOperatorRegistry.with_extra`
增量注入, 避免把支持表写死。
"""
from typing import Any, Dict, Iterable, List, Optional, Set

from xpu_converter.backend.base import OperatorAnalysis, OperatorAnalysisResult

# 自定义域算子(如 com.microsoft / com.baidu)默认视为不支持, 需显式登记
DEFAULT_DOMAINS = ("", "ai.onnx", "ai.onnx.ml")

# 昆仑 XPU 原生算子(覆盖 V1 检测/分类/分割常见结构)
NATIVE_OPS: Set[str] = {
    # ---- 卷积 / 池化 ----
    "Conv", "ConvTranspose", "MaxPool", "AveragePool", "GlobalAveragePool",
    "GlobalMaxPool", "LpPool", "MaxUnpool", "RoiAlign",
    # ---- 归一化 ----
    "BatchNormalization", "InstanceNormalization", "GroupNormalization",
    "LayerNormalization", "LRN", "MeanVarianceNormalization",
    # ---- 激活 ----
    "Relu", "LeakyRelu", "PRelu", "Sigmoid", "Tanh", "HardSigmoid", "HardSwish",
    "Elu", "Selu", "Softplus", "Softsign", "Gelu", "Erf", "Clip",
    "ThresholdedRelu", "Celu",
    # ---- 矩阵 / 全连接 ----
    "Gemm", "MatMul", "Einsum", "Trilu",
    # ---- 逐元素数学 ----
    "Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "Exp", "Log", "Abs", "Neg",
    "Floor", "Ceil", "Round", "Reciprocal", "Sign", "Max", "Min", "Sum",
    "Mean", "Mod", "CumSum",
    # ---- 逻辑 / 比较 ----
    "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "And", "Or",
    "Not", "Xor", "IsNaN", "IsInf",
    # ---- 张量变形 ----
    "Reshape", "Squeeze", "Unsqueeze", "Transpose", "Flatten", "Expand", "Tile",
    "Pad", "Concat", "Split", "Slice", "DepthToSpace", "SpaceToDepth",
    "Identity", "Constant", "ConstantOfShape", "Cast", "CastLike",
    "Resize", "Upsample", "Where", "Range", "Compress", "EyeLike",
    # ---- 索引 / 聚合 ----
    "Gather", "GatherElements", "GatherND", "ScatterElements", "ScatterND",
    "Shape", "Size", "NonZero", "TopK", "ArgMax", "ArgMin", "OneHot",
    "ReduceMax", "ReduceMin", "ReduceMean", "ReduceSum", "ReduceProd",
    "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp",
    "ReduceSumSquare", "ReduceSqrt",
    # ---- 概率 ----
    "Softmax", "LogSoftmax", "Dropout",
    # ---- 循环网络 ----
    "LSTM", "GRU", "RNN",
}

# 需经 :mod:`xpu_converter.rewrite` 改写后才能被昆仑 XPU 消费的算子
REWRITE_OPS: Dict[str, str] = {
    "Silu": "silu",
    "Mish": "mish",
    "Upsample": "resize",
    "NonMaxSuppression": "nms",
}


class KunlunOperatorRegistry:
    """昆仑芯算子注册表。"""

    def __init__(
        self,
        native_ops: Optional[Iterable[str]] = None,
        rewrite_ops: Optional[Dict[str, str]] = None,
        domains: Optional[Iterable[str]] = None,
    ) -> None:
        self.native_ops: Set[str] = set(native_ops if native_ops is not None else NATIVE_OPS)
        self.rewrite_ops: Dict[str, str] = dict(rewrite_ops if rewrite_ops is not None else REWRITE_OPS)
        self.domains: Set[str] = set(domains if domains is not None else DEFAULT_DOMAINS)

    # ---- 注册 ----
    def with_extra(self, native_ops: Optional[Iterable[str]] = None,
                   rewrite_ops: Optional[Dict[str, str]] = None) -> "KunlunOperatorRegistry":
        """返回注入增量后的新注册表(不修改自身)。"""
        merged_native = set(self.native_ops)
        merged_native.update(native_ops or [])
        merged_rewrite = dict(self.rewrite_ops)
        merged_rewrite.update(rewrite_ops or {})
        return KunlunOperatorRegistry(merged_native, merged_rewrite, self.domains)

    def register(self, op_type: str) -> None:
        self.native_ops.add(op_type)

    # ---- 分类 ----
    def classify(self, op_type: str, domain: str = "") -> str:
        """返回 ``supported`` / ``rewritten`` / ``unsupported``。"""
        if domain and domain not in self.domains:
            return OperatorAnalysisResult.UNSUPPORTED
        if op_type in self.rewrite_ops:
            return OperatorAnalysisResult.REWRITTEN
        if op_type in self.native_ops:
            return OperatorAnalysisResult.SUPPORTED
        return OperatorAnalysisResult.UNSUPPORTED

    # ---- 分析 ----
    def analyze(self, graph) -> OperatorAnalysis:
        """统计图中算子的支持情况, ``graph`` 可为 IR Graph 或原始 ModelProto。"""
        result = OperatorAnalysis()
        for op_type, domain in self._iter_ops(graph):
            result.op_counts[op_type] = result.op_counts.get(op_type, 0) + 1
            kind = self.classify(op_type, domain)
            if kind == OperatorAnalysisResult.SUPPORTED:
                result.supported += 1
            elif kind == OperatorAnalysisResult.REWRITTEN:
                result.rewritten += 1
            else:
                result.unsupported += 1
                if op_type not in result.unsupported_ops:
                    result.unsupported_ops.append(op_type)
                if domain and domain not in self.domains and domain not in result.domain_ops:
                    result.domain_ops.append(domain)
        for op_type in result.op_counts:
            if op_type in self.rewrite_ops:
                result.rewrite_ops[op_type] = self.rewrite_ops[op_type]
        return result

    def describe(self) -> Dict[str, Any]:
        return {
            "native_op_count": len(self.native_ops),
            "rewrite_ops": dict(self.rewrite_ops),
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


def default_registry(target_chip: str = "auto") -> KunlunOperatorRegistry:
    """按目标芯片返回算子注册表。

    ``target_chip`` 目前只做占位: 在拿到昆仑 SDK 后, 可在此按型号裁剪支持表,
    例如 R200 与 R300 对 ``Resize`` 的插值模式支持范围不同。
    """
    return KunlunOperatorRegistry()


def collect_unique_ops(graph) -> List[str]:
    """按首次出现顺序返回图中的去重算子名。"""
    seen: List[str] = []
    for op_type, _domain in KunlunOperatorRegistry._iter_ops(graph):
        if op_type not in seen:
            seen.append(op_type)
    return seen
