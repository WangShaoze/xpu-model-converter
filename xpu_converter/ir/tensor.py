# -*- coding: utf-8 -*-
"""IR 张量。"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ONNX TensorProto.DataType -> IR dtype 名称
ONNX_DTYPE_TO_NAME: Dict[int, str] = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    8: "string",
    9: "bool",
    10: "float16",
    11: "float64",
    12: "uint32",
    13: "uint64",
    14: "complex64",
    15: "complex128",
    16: "bfloat16",
}

NAME_TO_ONNX_DTYPE: Dict[str, int] = {v: k for k, v in ONNX_DTYPE_TO_NAME.items()}


@dataclass
class Tensor:
    """一个具名张量（图输入/输出、中间值或常量）。"""

    name: str
    shape: List[Optional[int]] = field(default_factory=list)
    dtype: str = "float32"
    is_initializer: bool = False

    def has_dynamic_dim(self) -> bool:
        return any(d is None or int(d) < 0 for d in self.shape)

    def is_static(self) -> bool:
        return len(self.shape) > 0 and not self.has_dynamic_dim()

    def numel(self) -> Optional[int]:
        if not self.shape or self.has_dynamic_dim():
            return None
        total = 1
        for dim in self.shape:
            total *= int(dim)
        return total

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "is_initializer": self.is_initializer,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Tensor":
        return cls(
            name=str(data.get("name", "")),
            shape=list(data.get("shape") or []),
            dtype=str(data.get("dtype") or "float32"),
            is_initializer=bool(data.get("is_initializer", False)),
        )

    def __str__(self) -> str:
        shape = "x".join("?" if d is None else str(d) for d in self.shape) or "scalar"
        return "{}:{}[{}]".format(self.name, self.dtype, shape)
