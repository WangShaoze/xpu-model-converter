# -*- coding: utf-8 -*-
"""昆仑芯编译器(SDK 解耦)。

设计约束(建设目标 §17 末尾):
    在 XPU 型号、SDK/XTCL 版本确定之前, 本模块**不假设任何具体厂商 API**。
    具体调用被隔离在 :class:`KunlunSdkAdapter` 的各个子类中:

    ============  ==================================================
    sdk_adapter   说明
    ============  ==================================================
    ``auto``      按优先级自动探测可用适配器
    ``paddle``    复用镜像内 Paddle Inference 的 XPU 能力(加载即编译)
    ``xpuctl``    调用昆仑 XPU Toolkit / XTCL 编译 ONNX 为 ``model.xpu``
    ``stub``      SDK 未就绪时的离线占位产物, 标记 degraded, 禁止对外交付
    ============  ==================================================

拿到昆仑 SDK 信息后, 只需补全对应适配器里的 ``compile`` / ``open_session``,
:meth:`KunlunBackend.compile` 这一层无需改动。
"""
import importlib
import importlib.util
import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from xpu_converter.backend.base import BackendArtifact, BaseBackend, BaseRuntimeSession, OperatorAnalysis
from xpu_converter.backend.kunlun.config import KunlunConfig
from xpu_converter.backend.kunlun.graph_builder import XpuGraph, write_compile_report
from xpu_converter.backend.kunlun.operator_registry import KunlunOperatorRegistry, default_registry
from xpu_converter.errors import BackendNotAvailableError, CompileError
from xpu_converter.version import CONVERTER_VERSION

# 交付包内约定的编译模型文件名(建设目标 §6)
ARTIFACT_FILENAME = "model.xpu"
METADATA_FILENAME = "metadata.json"

# 候选 SDK 模块名: 昆仑侧命名可能变化, 因此做多候选探测
XPU_TOOLKIT_MODULES = ("xpu_toolkit", "xtcl", "xpu_inference")


class KunlunSdkAdapter(ABC):
    """昆仑 SDK 适配器。

    刻意保持极薄接口: 只要求 ``available`` / ``compile`` / ``open_session``,
    使整个项目不绑死在某个 SDK 版本上。
    """

    name = "base"
    priority = 100
    accepts_onnx = True

    @classmethod
    def available(cls) -> bool:
        """当前环境是否具备该 SDK。"""
        return False

    @classmethod
    def describe(cls) -> Dict[str, Any]:
        return {"adapter": cls.name, "available": cls.available()}

    @abstractmethod
    def compile(self, xpu_graph: XpuGraph, output_path: str, config: KunlunConfig) -> BackendArtifact:
        """把待编译图落成 ``model.xpu``。"""

    @abstractmethod
    def open_session(self, model_path: str, config: KunlunConfig) -> BaseRuntimeSession:
        """为 ``model.xpu`` 创建推理会话。"""


class StubSdkAdapter(KunlunSdkAdapter):
    """离线占位适配器。

    昆仑 SDK 未就绪时, 把规范化后的 ONNX 直接作为 ``model.xpu`` 落盘, 使
    转换 → 校验 → 打包 → Runtime 的**全链路可以先打通并验证**。
    产物在 :class:`xpu_converter.backend.base.BackendArtifact` 中标记为
    ``degraded=True``, 交付层模板会同时写入醒目告警, 避免误交付。
    """

    name = "stub"
    priority = 999

    @classmethod
    def available(cls) -> bool:
        return True

    def compile(self, xpu_graph: XpuGraph, output_path: str, config: KunlunConfig) -> BackendArtifact:
        source = xpu_graph.onnx_path
        if not source or not Path(source).is_file():
            raise CompileError("stub 适配器需要待编译图已落盘, 请先构建 XpuGraph 时传入 workdir")
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, str(target))
        return BackendArtifact(
            model_path=str(target),
            precision=config.precision,
            sdk_adapter=self.name,
            target_chip=config.target_chip,
            artifact_format="stub",
            notes=[
                "未检测到昆仑 SDK, 已生成占位产物(内容为规范化 ONNX), 仅供链路联调, 不可对外交付",
            ],
            metadata={"onnx_source": source},
        )

    def open_session(self, model_path: str, config: KunlunConfig) -> BaseRuntimeSession:
        from xpu_converter.backend.kunlun.runtime import OnnxRuntimeSession

        return OnnxRuntimeSession(model_path, device=config.device)


class XpuToolkitSdkAdapter(KunlunSdkAdapter):
    """昆仑 XPU Toolkit / XTCL 适配器(ONNX → ``model.xpu``)。

    TODO(SDK): 拿到昆仑 SDK 版本后, 在 :meth:`compile` 中落地真实编译参数
    (芯片型号、输入 layout、量化配置等)。当前实现保留了完整的探测与报错路径,
    确保调用方代码无需改动。
    """

    name = "xpuctl"
    priority = 10
    accepts_onnx = True

    @classmethod
    def module_name(cls, config: Optional[KunlunConfig] = None) -> Optional[str]:
        if config is not None and config.sdk_module:
            return config.sdk_module if importlib.util.find_spec(config.sdk_module) else None
        for candidate in XPU_TOOLKIT_MODULES:
            if importlib.util.find_spec(candidate) is not None:
                return candidate
        return None

    @classmethod
    def available(cls) -> bool:
        return cls.module_name() is not None

    def compile(self, xpu_graph: XpuGraph, output_path: str, config: KunlunConfig) -> BackendArtifact:
        module_name = self.module_name(config)
        if not module_name:
            raise BackendNotAvailableError(
                "未找到昆仑 XPU Toolkit 模块(候选: {}), 可用 sdk_adapter=stub 先行联调".format(
                    ", ".join(XPU_TOOLKIT_MODULES)
                )
            )
        if not xpu_graph.onnx_path or not Path(xpu_graph.onnx_path).is_file():
            raise CompileError("待编译图未落盘, 无法调用 XPU Toolkit")

        module = importlib.import_module(module_name)
        compiler_cls = _first_attr(module, ("XpuCompiler", "Compiler", "XPUCompiler"))
        if compiler_cls is None:
            raise BackendNotAvailableError(
                "模块 {} 中未找到编译入口(期望 XpuCompiler/Compiler)。"
                "请在 KunlunSdkAdapter 子类中按实际 SDK 调整调用方式".format(module_name)
            )
        # TODO(SDK): 依据实际 SDK 签名补全参数(芯片型号 / 精度 / 输入规格)
        compiler = compiler_cls(xpu_graph.onnx_path)
        compile_method = _callable_attr(compiler, ("compile", "build", "convert"))
        if compile_method is None:
            raise BackendNotAvailableError(
                "{} 实例上未找到 compile/build/convert 方法, 待按实际 SDK 落地".format(module_name)
            )
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        result = compile_method(output=str(target), precision=xpu_graph.precision)
        produced = _resolve_output_path(result, target)
        if not produced.is_file():
            raise CompileError("XPU Toolkit 未生成编译产物: {}".format(produced))
        return BackendArtifact(
            model_path=str(produced),
            precision=config.precision,
            sdk_adapter=self.name,
            target_chip=config.target_chip,
            artifact_format="xpu",
            metadata={"sdk_module": module_name, "onnx_source": xpu_graph.onnx_path},
        )

    def open_session(self, model_path: str, config: KunlunConfig) -> BaseRuntimeSession:
        from xpu_converter.backend.kunlun.runtime import XpuToolkitRuntimeSession

        module_name = self.module_name(config)
        if not module_name:
            raise BackendNotAvailableError("未找到昆仑 XPU Toolkit 模块, 无法创建推理会话")
        return XpuToolkitRuntimeSession(model_path, module_name=module_name, device=config.device)


class PaddleXpuSdkAdapter(KunlunSdkAdapter):
    """Paddle Inference + XPU 适配器。

    昆仑 XPU 的 kernel 由上游 Paddle 在**加载时编译**, 因此本适配器只接受
    Paddle 推理 program(``.pdmodel``/``.pdiparams``), 不消费 ONNX。
    用于第三阶段 PaddleOCR / PaddleDetection 线路。
    """

    name = "paddle"
    priority = 20
    accepts_onnx = False

    @classmethod
    def available(cls) -> bool:
        if importlib.util.find_spec("paddle") is None:
            return False
        try:
            inference = importlib.import_module("paddle.inference")
        except ImportError:  # pragma: no cover - 依赖镜像内 Paddle 版本
            return False
        return hasattr(inference, "Config")

    def compile(self, xpu_graph: XpuGraph, output_path: str, config: KunlunConfig) -> BackendArtifact:
        raise BackendNotAvailableError(
            "paddle 适配器仅支持 Paddle 推理模型, 当前流水线输入为 ONNX; "
            "如需走 Paddle 线路请使用 frontend=paddle 并在拿到昆仑 SDK 后补全本适配器"
        )

    def open_session(self, model_path: str, config: KunlunConfig) -> BaseRuntimeSession:
        from xpu_converter.backend.kunlun.runtime import PaddleXpuRuntimeSession

        return PaddleXpuRuntimeSession(model_path, device=config.device)


def _first_attr(obj, names) -> Any:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _callable_attr(obj, names):
    for name in names:
        value = getattr(obj, name, None)
        if callable(value):
            return value
    return None


def _resolve_output_path(result: Any, default: Path) -> Path:
    """SDK 返回值可能是路径字符串、``Path`` 或 dict, 统一解析为产物路径。"""
    if result is None:
        return default
    if isinstance(result, (str, Path)):
        return Path(result)
    if isinstance(result, dict):
        for key in ("output", "model_path", "path", "model"):
            if result.get(key):
                return Path(result[key])
    return default


# 自动探测顺序: 优先能直接消费 ONNX 的 SDK
ADAPTER_REGISTRY: List[type] = [XpuToolkitSdkAdapter, PaddleXpuSdkAdapter]


class KunlunBackend(BaseBackend):
    """昆仑芯后端实现。"""

    name = "kunlun"

    def __init__(self, config: Optional[KunlunConfig] = None,
                 registry: Optional[KunlunOperatorRegistry] = None) -> None:
        self.config = config or KunlunConfig()
        self.registry = registry or default_registry(self.config.target_chip)

    # ------------------------------------------------------------------ 描述
    def describe(self) -> Dict[str, Any]:
        adapter, degraded = self._resolve_adapter()
        return {
            "backend": self.name,
            "target_chip": self.config.target_chip,
            "precision": self.config.precision,
            "device": self.config.device,
            "sdk_adapter": adapter.name if adapter else None,
            "degraded": degraded,
            "adapters": [cls.describe() for cls in ADAPTER_REGISTRY + [StubSdkAdapter]],
            "operator_registry": self.registry.describe(),
        }

    def sdk_ready(self) -> bool:
        """是否存在可直接产出真实 XPU 产物的 SDK。"""
        adapter, degraded = self._resolve_adapter()
        return adapter is not None and not degraded

    # ------------------------------------------------------------------ 分析
    def analyze(self, graph) -> OperatorAnalysis:
        return self.registry.analyze(graph)

    # ------------------------------------------------------------------ 编译
    def compile(self, xpu_graph: XpuGraph, output_path: str, config: Optional[KunlunConfig] = None) -> BackendArtifact:
        config = config or self.config
        adapter, degraded = self._resolve_adapter(config)
        if adapter is None:
            raise BackendNotAvailableError(
                "无可用的昆仑 SDK 适配器, 且未允许降级生成占位产物"
            )
        if degraded and not config.allow_degraded:
            raise BackendNotAvailableError(
                "未检测到昆仑 SDK(候选: {}), 且 allow_degraded=False, 拒绝生成占位产物。"
                "请安装昆仑 SDK 或改用 sdk_adapter=xpuctl".format(", ".join(XPU_TOOLKIT_MODULES))
            )

        artifact = adapter.compile(xpu_graph, output_path, config)
        artifact.metadata.setdefault("converter_version", CONVERTER_VERSION)
        artifact.metadata.setdefault("target_chip", config.target_chip)
        artifact.metadata.setdefault("graph", xpu_graph.to_dict())
        if degraded:
            artifact.notes.append("sdk_adapter=stub: 编译层尚未落到真实昆仑 SDK API(建设目标 §17)")
        self._write_metadata(artifact, output_path)
        if xpu_graph.onnx_path:
            write_compile_report(xpu_graph, str(Path(output_path).parent), {"sdk_adapter": artifact.sdk_adapter})
        return artifact

    # ------------------------------------------------------------------ 运行
    def create_runtime(self, artifact: BackendArtifact, config: Optional[KunlunConfig] = None) -> BaseRuntimeSession:
        config = config or self.config
        adapter_cls = self._adapter_class(artifact.sdk_adapter)
        if adapter_cls is None:
            raise BackendNotAvailableError("未知的 sdk_adapter: {}".format(artifact.sdk_adapter))
        return adapter_cls().open_session(artifact.model_path, config)

    # ------------------------------------------------------------------ 内部
    def _resolve_adapter(self, config: Optional[KunlunConfig] = None) -> Tuple[Optional[KunlunSdkAdapter], bool]:
        """返回 ``(适配器实例, 是否降级)``。"""
        config = config or self.config
        requested = config.sdk_adapter
        if requested == "auto":
            for adapter_cls in ADAPTER_REGISTRY:
                if adapter_cls.accepts_onnx and adapter_cls.available():
                    return adapter_cls(), False
            if config.allow_degraded:
                return StubSdkAdapter(), True
            return None, False
        if requested == "stub":
            return StubSdkAdapter(), True
        adapter_cls = self._adapter_class(requested)
        if adapter_cls is None:
            raise BackendNotAvailableError("不支持的 sdk_adapter: {}".format(requested))
        if not adapter_cls.available():
            raise BackendNotAvailableError(
                "sdk_adapter={} 在当前环境不可用(缺少对应 SDK 模块)".format(requested)
            )
        return adapter_cls(), False

    @staticmethod
    def _adapter_class(name: str) -> Optional[type]:
        for adapter_cls in ADAPTER_REGISTRY + [StubSdkAdapter]:
            if adapter_cls.name == name:
                return adapter_cls
        return None

    @staticmethod
    def _write_metadata(artifact: BackendArtifact, output_path: str) -> str:
        path = Path(output_path).parent / METADATA_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fw:
            json.dump(artifact.to_dict(), fw, ensure_ascii=False, indent=2)
        return str(path)
