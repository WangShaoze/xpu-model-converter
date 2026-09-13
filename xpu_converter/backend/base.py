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
from pathlib import Path
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
    # 每个不支持算子的具体原因(如 Resize 的 mode/dtype 不满足), 便于定位问题
    reasons: List[str] = field(default_factory=list)

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
            "reasons": list(self.reasons),
        }


@dataclass
class BackendArtifact:
    """编译产物。

    产物可能是**单个文件**(如 ``model.xpu`` / stub 的 ``model.onnx``), 也可能是
    **一组文件**(Paddle 静态图为 ``model.pdmodel`` + ``model.pdiparams``)。
    因此除 ``model_path``(主文件)外, 增加 ``files`` 记录全部交付文件。
    """

    model_path: str
    precision: str = "fp16"
    sdk_adapter: str = "unknown"
    target_chip: str = "auto"
    artifact_format: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        """是否为降级产物(SDK 缺失时生成的占位件)。"""
        return self.artifact_format in ("stub", "placeholder")

    @property
    def artifact_files(self) -> List[str]:
        """产物全部文件(主文件在前); ``files`` 为空时回退到 ``model_path``。"""
        if self.files:
            return list(self.files)
        return [self.model_path] if self.model_path else []

    @property
    def filenames(self) -> List[str]:
        return [Path(path).name for path in self.artifact_files]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "model_file": Path(self.model_path).name if self.model_path else "",
            "files": self.artifact_files,
            "precision": self.precision,
            "sdk_adapter": self.sdk_adapter,
            "target_chip": self.target_chip,
            "artifact_format": self.artifact_format,
            "degraded": self.degraded,
            "metadata": dict(self.metadata),
            "notes": list(self.notes),
        }


class BaseRuntimeSession(ABC):
    """编译产物的一次推理会话。

    从 P0-1/P0-2(禁止把 CPU 执行误当成目标 XPU 执行)起, 所有会话语义上都必须
    携带**执行溯源(execution provenance)**: 它到底跑在哪个真实设备上, 而不是
    只凭 ``backend_name = "paddle-xpu"`` 就默认等于跑在 XPU 上。Validator 与
    Benchmark 依赖这里判定 ``actual_device == "xpu"`` 才会产出可信的
    PASS / OK 结论。
    """

    backend_name = "base"
    # ---- 执行溯源(每个会话必须如实填充, 默认置空表示"未声明", 便于门禁识别)
    requested_device: str = "auto"      # 用户请求的运行设备
    actual_device: str = ""             # 实际运行设备: xpu / cpu / cuda:0 ...
    execution_mode: str = ""            # hardware / fallback_cpu / cpu / simulated
    device_available: bool = False      # 目标加速设备在本次运行是否真实可用
    device_id: int = 0                  # 实际使用的设备编号

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

    def synchronize(self) -> None:
        """阻塞直到异步加速器执行完成(P1-2)。

        默认实现为无操作——CPU / 同步式后端天然同步, 无需等待; 真正异步的
        XPU/GPU 后端必须在子类覆盖, 否则 benchmark 计时会只测到 enqueue 时间。
        """

    def execution_provenance(self) -> Dict[str, Any]:
        """把执行溯源序列化为可落盘报告(P1-4 / 状态机 Gate G6)。"""
        return {
            "backend": self.backend_name,
            "requested_device": self.requested_device,
            "actual_device": self.actual_device,
            "execution_mode": self.execution_mode,
            "device_available": bool(self.device_available),
            "device_id": int(self.device_id or 0),
        }

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
    def analyze(self, graph, precision: Optional[str] = None) -> OperatorAnalysis:
        """分析图内算子在目标后端上的支持情况。"""

    @abstractmethod
    def compile(self, xpu_graph, output_path, config) -> BackendArtifact:
        """把后端无关图编译为交付产物。"""

    @abstractmethod
    def create_runtime(self, artifact: BackendArtifact, config) -> BaseRuntimeSession:
        """为编译产物创建推理会话。"""
