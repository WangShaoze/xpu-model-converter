# -*- coding: utf-8 -*-
"""算子能力(Capability)模型。

把过去扁平的 ``Set[str]`` 白名单升级为带约束的 :class:`OperatorCapability`,
使 ``Resize(mode=nearest)`` 与 ``Resize(mode=bilinear)``、``MatMul(fp16)`` 与
``MatMul(fp32)`` 可以被区分判定(ChatGPT 修改意见 §7 / §8)。
"""
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Set, Tuple

from xpu_converter.errors import ConfigError

# dtype 归一化: ONNX 侧 fp16 写作 float16, 精度配置写作 fp16, 二者需统一
PRECISION_TO_DTYPE = {"fp32": "float32", "fp16": "float16", "fp64": "float64", "int8": "int8"}
FLOAT_DTYPES = frozenset({"float32", "float16", "float64"})

SUPPORTED = "supported"
REWRITTEN = "rewritten"
UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class OperatorCapability:
    """单个算子的能力描述。

    空集合语义为「不做该维度限制」: ``dtypes`` 为空表示任意 dtype 均可,
    ``attributes`` 为空表示不校验属性。
    """

    op_type: str
    domain: str = ""
    dtypes: FrozenSet[str] = frozenset()
    attributes: Mapping[str, FrozenSet[Any]] = field(default_factory=dict)
    min_opset: int = 1
    max_opset: Optional[int] = None

    def check_dtype(self, dtype: Optional[str]) -> Optional[str]:
        """返回不满足原因, 满足则返回 ``None``。"""
        if not self.dtypes or not dtype:
            return None
        if dtype not in self.dtypes:
            return "dtype={} 不在支持列表 {}".format(dtype, sorted(self.dtypes))
        return None

    def check_attributes(self, attributes: Mapping[str, Any]) -> Optional[str]:
        """只校验白名单里显式列出的属性; 其余属性不干预。"""
        for name, allowed in (self.attributes or {}).items():
            if name not in (attributes or {}):
                continue
            value = attributes[name]
            if value not in allowed:
                return "属性 {}={} 不在支持列表 {}".format(name, value, sorted(allowed))
        return None

    def check_opset(self, opset: Optional[int]) -> Optional[str]:
        if opset is None:
            return None
        if opset < self.min_opset:
            return "opset={} 低于最小支持版本 {}".format(opset, self.min_opset)
        if self.max_opset is not None and opset > self.max_opset:
            return "opset={} 高于最大支持版本 {}".format(opset, self.max_opset)
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_type": self.op_type,
            "domain": self.domain,
            "dtypes": sorted(self.dtypes),
            "attributes": {k: sorted(v, key=str) for k, v in (self.attributes or {}).items()},
            "min_opset": self.min_opset,
            "max_opset": self.max_opset,
        }


def _to_frozenset(values: Any) -> FrozenSet[Any]:
    if values is None:
        return frozenset()
    if isinstance(values, (str, bytes)):
        return frozenset([values])
    return frozenset(values)


class OperatorCapabilitySet:
    """一套算子能力表(通常对应一个硬件 + SDK 版本)。"""

    def __init__(
        self,
        operators: Optional[Mapping[str, OperatorCapability]] = None,
        rewrite_ops: Optional[Mapping[str, str]] = None,
        domains: Optional[Iterable[str]] = None,
        version: int = 1,
        target_chip: str = "auto",
    ) -> None:
        self.operators: Dict[str, OperatorCapability] = dict(operators or {})
        self.rewrite_ops: Dict[str, str] = dict(rewrite_ops or {})
        self.domains: Set[str] = set(domains or ("", "ai.onnx", "ai.onnx.ml"))
        self.version = int(version)
        self.target_chip = target_chip

    # ------------------------------------------------------------------ 构造
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "OperatorCapabilitySet":
        data = dict(data or {})
        operators: Dict[str, OperatorCapability] = {}
        for op_type, spec in (data.get("operators") or {}).items():
            spec = dict(spec or {})
            operators[op_type] = OperatorCapability(
                op_type=op_type,
                domain=str(spec.get("domain", "") or ""),
                dtypes=_to_frozenset(spec.get("dtypes")),
                attributes={
                    str(k): _to_frozenset(v) for k, v in (spec.get("attributes") or {}).items()
                },
                min_opset=int(spec.get("min_opset", 1) or 1),
                max_opset=(None if spec.get("max_opset") is None else int(spec["max_opset"])),
            )
        return cls(
            operators=operators,
            rewrite_ops={str(k): str(v) for k, v in (data.get("rewrite") or {}).items()},
            domains=data.get("domains"),
            version=int(data.get("version", 1) or 1),
            target_chip=str(data.get("target_chip", "auto") or "auto"),
        )

    @classmethod
    def from_yaml(cls, path) -> "OperatorCapabilitySet":
        from xpu_converter.config import load_yaml

        return cls.from_dict(load_yaml(path))

    def with_extra(self, operators: Optional[Iterable[str]] = None,
                   rewrite_ops: Optional[Mapping[str, str]] = None) -> "OperatorCapabilitySet":
        """在副本上追加算子/改写项, 不修改自身。"""
        merged = dict(self.operators)
        for op_type in operators or []:
            merged.setdefault(str(op_type), OperatorCapability(op_type=str(op_type)))
        merged_rewrite = dict(self.rewrite_ops)
        merged_rewrite.update(rewrite_ops or {})
        return OperatorCapabilitySet(merged, merged_rewrite, self.domains, self.version, self.target_chip)

    # ------------------------------------------------------------------ 判定
    def classify(
        self,
        op_type: str,
        domain: str = "",
        dtype: Optional[str] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        opset: Optional[int] = None,
    ) -> Tuple[str, str]:
        """返回 ``(状态, 原因)``, 状态为 supported / rewritten / unsupported。"""
        if domain and domain not in self.domains:
            return UNSUPPORTED, "自定义域 {} 未登记为可支持".format(domain)
        if op_type in self.rewrite_ops:
            return REWRITTEN, "由 rewrite 规则 {} 改写".format(self.rewrite_ops[op_type])
        capability = self.operators.get(op_type)
        if capability is None:
            return UNSUPPORTED, "能力表中不存在算子 {}".format(op_type)
        for reason in (
            capability.check_opset(opset),
            capability.check_dtype(dtype),
            capability.check_attributes(attributes or {}),
        ):
            if reason:
                return UNSUPPORTED, "{}: {}".format(op_type, reason)
        return SUPPORTED, ""

    def describe(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "target_chip": self.target_chip,
            "operator_count": len(self.operators),
            "rewrite_ops": dict(self.rewrite_ops),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "target_chip": self.target_chip,
            "rewrite": dict(self.rewrite_ops),
            "operators": {k: v.to_dict() for k, v in self.operators.items()},
        }

    @staticmethod
    def precision_dtype(precision: str) -> str:
        return PRECISION_TO_DTYPE.get(str(precision or "").lower(), "float32")


def deep_copy_capability_set(caps: OperatorCapabilitySet) -> OperatorCapabilitySet:
    return copy.deepcopy(caps)
