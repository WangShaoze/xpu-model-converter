# -*- coding: utf-8 -*-
"""后端抽象层。

设计约束(建设目标 §17 末尾):
    昆仑芯 Backend 必须刻意做成 **SDK 解耦接口**。在 XPU 型号、SDK/XTCL 版本
    确定之前, 本层不假设任何具体厂商 API, 只固定"输入优化后的 IR, 输出编译
    产物"这一契约。后续拿到昆仑 SDK 信息后, 只需在
    :class:`xpu_converter.backend.kunlun.compiler.KunlunSdkAdapter` 的子类里
    落地真实调用, 整个项目架构不受影响。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class OperatorAnalysisResult:
    """算子分析结果的取值常量。"""

    SUPPORTED = "supported"
    REWRITTEN = "rewritten"
    UNSUPPORTED = "unsupported"


@dataclass
class OperatorAnalysis:
    """算子分析结果 (CLI 步骤 05)。"""

    supported: int = 0
    rewritten: int = 0
    unsupported: int = 0
    op_counts: Dict[str, int] = field(default_factory=dict)
    rewrite_ops: Dict[str, str] = field(default_factory=dict)
    unsupported_ops: List[str] = field(default_factory=list)
    domain_ops: List[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.supported + self.rewritten + self.unsupported

    @property
    def ok(self) -> bool:
        return self.unsupported == 0

    def summary(self) -> str:
        """形如 ``143 supported / 2 rewritten / 0 unsupported``。"""
        return "{} supported / {} rewritten / {} unsupported".format(
            self.supported, self.rewritten, self.unsupported
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "supported": self.supported,
            "rewritten": self.rewritten,
            "unsupported": self.unsupported,
            "op_counts": dict(self.op_counts),
            "rewrite_ops": dict(self.rewrite_ops),
            "unsupported_ops": list(self.unsupported_ops),
            "domain_ops": list(self.domain_ops),
        }


@dataclass
class BackendArtifact:
    """编译产物。

    ``model.xpu`` 是交付包内约定的编译模型文件名(建设目标 §6), 其余字段用于
    回写 ``manifest.json`` / ``model.yaml`` / ``metadata.json``。
    """

    model_path: str
    precision: str = "fp16"
    sdk_adapter: str = "unknown"
    target_chip: str = "auto"
    artifact_format: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        """是否为降级产物(SDK 缺失时生成的占位件)。"""
        return self.artifact_format in ("stub", "placeholder")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "precision": self.precision,
            "sdk_adapter": self.sdk_adapter,
            "target_chip": self.target_chip,
            "artifact_format": self.artifact_format,
            "degraded": self.degraded,
            "metadata": dict(self.metadata),
            "notes": list(self.notes),
        }


class BaseRuntimeSession(ABC):
    """编译产物的一次推理会话。"""

    backend_name = "base"

    @abstractmethod
    def run(self, inputs: Dict[str, Any]) -> List[Any]:
        """执行一次推理, 输入/输出张量均为 numpy 数组。"""

    @property
    @abstractmethod
    def input_names(self) -> List[str]:
        """模型输入名列表。"""

    @property
    @abstractmethod
    def output_names(self) -> List[str]:
        """模型输出名列表。"""

    def input_shapes(self) -> Optional[Dict[str, Tuple]]:
        return None

    def close(self) -> None:
        """释放会话资源, 默认无操作。"""

    def __enter__(self) -> "BaseRuntimeSession":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False


class BaseBackend(ABC):
    """硬件后端抽象。"""

    name = "base"

    @abstractmethod
    def describe(self) -> Dict[str, Any]:
        """返回后端与 SDK 可用性描述, 供 ``inspect`` / ``analyze`` 输出。"""

    @abstractmethod
    def analyze(self, graph) -> OperatorAnalysis:
        """分析图内算子在目标后端上的支持情况。"""

    @abstractmethod
    def compile(self, xpu_graph, output_path, config) -> BackendArtifact:
        """把后端无关图编译为交付产物。"""

    @abstractmethod
    def create_runtime(self, artifact: BackendArtifact, config) -> BaseRuntimeSession:
        """为编译产物创建推理会话。"""
