# -*- coding: utf-8 -*-
"""昆仑芯编译器(以 Paddle Inference 为唯一真实 Python 入口)。

环境事实(经实测确认):

* 昆仑 XPU SDK 是 C/C++ 库(``libxpurt.so`` / ``libxpuml.so``), **没有 Python API**;
* Python 侧唯一入口是 ``paddlepaddle-xpu`` 的 Paddle Inference
  (``Config`` / ``enable_xpu`` / ``set_xpu_device_id``);
* 因此"编译"= 把 ONNX 转成 Paddle 静态图(``model.pdmodel`` + ``model.pdiparams``),
  真正的 XPU kernel 由 Paddle 在**加载时**按设备编译。

适配器:

    ============  ==================================================
    sdk_adapter   说明
    ============  ==================================================
    ``auto``      按优先级自动探测(优先 paddle)
    ``paddle``    ONNX --x2paddle--> Paddle 静态图(设备无关, 加载时 enable_xpu)
    ``xpuctl``    预留: 若存在真实 XPU Toolkit Python 模块(由 sdk_module 指定)
    ``stub``      无后端时的占位产物, 标记 degraded, 禁止对外交付
    ============  ==================================================
"""
import importlib
import importlib.util
import json
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from xpu_converter.backend.base import BackendArtifact, BaseBackend, BaseRuntimeSession, OperatorAnalysis
from xpu_converter.backend.kunlun.config import KunlunConfig
from xpu_converter.backend.kunlun.graph_builder import XpuGraph, write_compile_report
from xpu_converter.backend.kunlun.operator_registry import KunlunOperatorRegistry, default_registry
from xpu_converter.errors import BackendNotAvailableError, CompileError
from xpu_converter.version import CONVERTER_VERSION

# 兼容旧引用: 交付包内约定的默认模型文件名
ARTIFACT_FILENAME = "model.xpu"
METADATA_FILENAME = "metadata.json"

# 仅供 ``sdk_adapter=xpuctl`` 使用: 真实 XPU Toolkit Python 模块名由配置
# (``sdk_module``)或环境变量指定。历史实现硬编码了三个并不存在的模块名, 导致
# 永远探测失败并静默降级, 这里改为"无配置即不可用"。
XPU_TOOLKIT_MODULE_ENV = "XPU_TOOLKIT_MODULE"


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
        """把待编译图落成交付产物。"""

    @abstractmethod
    def open_session(self, model_path: str, config: KunlunConfig) -> BaseRuntimeSession:
        """为产物创建推理会话。"""


class StubSdkAdapter(KunlunSdkAdapter):
    """离线占位适配器(仅供开发联调, 禁止对外交付)。

    把规范化后的 ONNX 落盘为 ``<stem>.onnx``(不再伪装成 ``model.xpu``), 并在
    :class:`BackendArtifact` 中标记 ``degraded``; 默认配置下不会走到这里。
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
        target = Path(output_path).with_suffix(".onnx")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, str(target))
        return BackendArtifact(
            model_path=str(target),
            precision=config.precision,
            sdk_adapter=self.name,
            target_chip=config.target_chip,
            artifact_format="stub",
            notes=[
                "未检测到真实昆仑后端, 已生成占位产物(内容为规范化 ONNX), 仅供链路联调, 不可对外交付",
            ],
            metadata={"onnx_source": source},
            files=[str(target)],
        )

    def open_session(self, model_path: str, config: KunlunConfig) -> BaseRuntimeSession:
        from xpu_converter.backend.kunlun.runtime import OnnxRuntimeSession

        return OnnxRuntimeSession(model_path, device=config.device)


class PaddleXpuSdkAdapter(KunlunSdkAdapter):
    """Paddle Inference + XPU 适配器(真实路径)。

    昆仑 XPU 的 kernel 由 Paddle 在**加载时**编译, 因此"编译"产物是一份
    **设备无关**的 Paddle 静态图: 有卡时 ``enable_xpu()``, 无卡时退 CPU。
    ONNX -> Paddle 静态图由 X2Paddle 完成。
    """

    name = "paddle"
    priority = 10
    accepts_onnx = True

    @classmethod
    def available(cls) -> bool:
        if importlib.util.find_spec("paddle") is None:
            return False
        if importlib.util.find_spec("x2paddle") is None:
            return False
        try:
            inference = importlib.import_module("paddle.inference")
        except ImportError:  # pragma: no cover - 依赖镜像内 Paddle 版本
            return False
        return hasattr(inference, "Config")

    def compile(self, xpu_graph: XpuGraph, output_path: str, config: KunlunConfig) -> BackendArtifact:
        if not xpu_graph.onnx_path or not Path(xpu_graph.onnx_path).is_file():
            raise CompileError("待编译图未落盘, 无法执行 ONNX -> Paddle 转换")
        if not self.available():
            raise BackendNotAvailableError(
                "缺少 Paddle Inference 或 X2Paddle, 无法执行 ONNX -> Paddle 转换"
            )

        target = Path(output_path)
        stem = target.stem or "model"
        workdir = target.parent
        workdir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="xpu_paddle_") as tmp:
            sanitized = _sanitize_onnx_for_x2paddle(xpu_graph.onnx_path, Path(tmp))
            _onnx_to_paddle(sanitized, tmp)
            prog, params = _locate_paddle_pair(Path(tmp))
            if prog is None or params is None:
                raise CompileError("X2Paddle 未产出 Paddle 静态图(.pdmodel/.pdiparams)")
            prog_dst = workdir / (stem + prog.suffix)
            params_dst = workdir / (stem + ".pdiparams")
            shutil.copyfile(str(prog), str(prog_dst))
            shutil.copyfile(str(params), str(params_dst))

        return BackendArtifact(
            model_path=str(prog_dst),
            precision=config.precision,
            sdk_adapter=self.name,
            target_chip=config.target_chip,
            artifact_format="paddle",
            notes=[
                "昆仑 XPU kernel 由 Paddle Inference 在加载时编译; 交付产物为设备无关静态图",
            ],
            metadata={
                "onnx_source": xpu_graph.onnx_path,
                "program_file": prog_dst.name,
                "params_file": params_dst.name,
                "input_names": list(xpu_graph.input_names),
                "output_names": list(xpu_graph.output_names),
                "x2paddle_version": _module_version("x2paddle"),
                "paddle_version": _module_version("paddle"),
            },
            files=[str(prog_dst), str(params_dst)],
        )

    def open_session(self, model_path: str, config: KunlunConfig) -> BaseRuntimeSession:
        from xpu_converter.backend.kunlun.runtime import PaddleXpuRuntimeSession

        return PaddleXpuRuntimeSession(model_path, device=config.device)


class XpuToolkitSdkAdapter(KunlunSdkAdapter):
    """昆仑 XPU Toolkit 适配器(预留)。

    昆仑 SDK 为 C/C++ 库, 无官方 Python API; 本适配器仅在显式配置了真实
    ``sdk_module``(配置项或环境变量 ``XPU_TOOLKIT_MODULE``)时才可用, 否则
    视为不可用, 不做任何虚构模块探测。
    """

    name = "xpuctl"
    priority = 20
    accepts_onnx = True

    @classmethod
    def module_name(cls, config: Optional[KunlunConfig] = None) -> Optional[str]:
        candidate = None
        if config is not None and config.sdk_module:
            candidate = config.sdk_module
        candidate = candidate or os.environ.get(XPU_TOOLKIT_MODULE_ENV) or ""
        if candidate and importlib.util.find_spec(candidate) is not None:
            return candidate
        return None

    @classmethod
    def available(cls) -> bool:
        return cls.module_name() is not None

    def compile(self, xpu_graph: XpuGraph, output_path: str, config: KunlunConfig) -> BackendArtifact:
        module_name = self.module_name(config)
        if not module_name:
            raise BackendNotAvailableError(
                "未配置真实昆仑 XPU Toolkit 模块(设置 sdk_module 或环境变量 {}), "
                "如需真实产物请使用 sdk_adapter=paddle".format(XPU_TOOLKIT_MODULE_ENV)
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
        target = Path(output_path).with_suffix(".xpu")
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
            files=[str(produced)],
        )

    def open_session(self, model_path: str, config: KunlunConfig) -> BaseRuntimeSession:
        from xpu_converter.backend.kunlun.runtime import XpuToolkitRuntimeSession

        module_name = self.module_name(config)
        if not module_name:
            raise BackendNotAvailableError("未配置昆仑 XPU Toolkit 模块, 无法创建推理会话")
        return XpuToolkitRuntimeSession(model_path, module_name=module_name, device=config.device)


# ---------------------------------------------------------------------- 工具
def _module_version(name: str) -> str:
    try:
        if importlib.util.find_spec(name) is None:
            return ""
        return str(getattr(importlib.import_module(name), "__version__", "") or "")
    except Exception:
        return ""


def _sanitize_onnx_for_x2paddle(onnx_path: str, workdir: Path) -> str:
    """修补 X2Paddle 1.6.0 无法处理的 ONNX 写法, 返回可转换的 ONNX 路径。

    处理的几类问题(任一按需应用, 无改动时返回原路径, 避免无谓复制):

    0. 常数折叠(onnx-simplifier): thuyngch 系(xolov9/7)的 ``chunk/split`` 会被
       torch 导出成 Slice, 其 start/end 由 ``Shape->Gather`` 等地步推导; X2Paddle
       无法折叠, 认为通道维是 -1 并报
       ``The number of input's channels should be ..., input's channels is -1``。
       先用 onnxsim 常数折叠把这些动态 Slice 边界固化为字面量, 得到全静态图;
    1. MaxPool 的默认 ``dilations=[1,1]`` 会被当作不支持属性并报错;
    2. Conv 的 ``kernel_shape`` 在 ONNX 中是**可选**属性(可从权重推导), 但
       X2Paddle 直接 ``len(kernel_shape)`` 会因 ``None`` 崩溃, 需按权重补齐。
    3. Resize 的 ``[X, "", "", sizes]``(opset>=11 提供 sizes 的标准写法)会让
       X2Paddle 1.6.0 在 ``node.inputs`` 里跳过空占位 scale, 取 sizes 时下标
       越界。把 scale 固化为常量并收敛为两输入 ``[X, scales]`` 规避。
    """
    current = str(onnx_path)
    simplified = _simplify_onnx_for_x2paddle(current, workdir)
    current = simplified

    import onnx

    model = onnx.load(current)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    changed = 0

    # 2. ``Resize`` 的 [X, "", "", sizes] 写法会让 X2Paddle 1.6.0 在构造
    #    ``node.inputs`` 时跳过空占位 scale, 随后 ``get_input_node(idx=3)``
    #    越界报 ``IndexError``。这里统一重写为 opset-10 的两输入形式
    #    ``[X, scales]``, 常量 scale 由静态 sizes / 输入形状推导。
    for node in model.graph.node:
        if node.op_type == "Resize" and any(i == "" for i in node.input):
            _rewrite_resize_in_place(model, node, initializers)
            changed += 1

    for node in model.graph.node:
        if node.op_type == "MaxPool" and _has_attribute(node, "dilations"):
            kept = [attr for attr in node.attribute if attr.name != "dilations"]
            del node.attribute[:]
            node.attribute.extend(kept)
            changed += 1
        if node.op_type in ("Conv", "ConvTranspose", "MaxPool", "AveragePool") \
                and not _has_attribute(node, "kernel_shape") and len(node.input) >= 2:
            weight = initializers.get(node.input[1])
            kernel = list(weight.dims)[2:] if weight is not None else []
            if kernel:
                node.attribute.extend([onnx.helper.make_attribute("kernel_shape", kernel)])
                changed += 1
    if changed == 0:
        return current
    sanitized = workdir / (Path(current).stem + "_x2paddle.onnx")
    onnx.save(model, str(sanitized))
    return str(sanitized)


def _simplify_onnx_for_x2paddle(onnx_path: str, workdir: Path) -> str:
    """用 onnx-simplifier 做常数折叠, 把数据相关的 Slice 边界固化为字面量。

    thuyngch 系(yolov9/7)的 ``chunk/split`` 经 torch 导出为 Slice, 其 start/end 常
    由 ``Shape -> Gather -> ... -> Mul`` 推导; X2Paddle 不折叠, 通道维被判为 -1。
    onnxsim 的常数折叠可消解这些动态边界, 得到 X2Paddle 能处理的静态图。

    依赖缺失或简化失败时**不阻断**, 退回原始路径; 真正的转换失败会在 onnx2paddle
    阶段报出可读错误。
    """
    if importlib.util.find_spec("onnxsim") is None:
        return onnx_path
    try:
        from onnxsim import simplify

        import onnx

        model = onnx.load(onnx_path)
        simplified_model, _ok = simplify(model)
        # 结果仍是个静态可转换图即可; ok 标志在 onnxruntime 校验缺失时可能为 False
        if simplified_model is None:
            return onnx_path
        target = workdir / (Path(onnx_path).stem + "_simplified.onnx")
        onnx.save(simplified_model, str(target))
        return str(target)
    except Exception:
        return onnx_path


def _static_shape(model: Any, name: str) -> List[int]:
    """返回静态图里某张量名字的已知 shape, 查不到或含动态维时返回 -1。"""
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if vi.name == name:
            dims = vi.type.tensor_type.shape.dim
            return [d.dim_value if d.HasField("dim_value") else -1 for d in dims]
    return []


def _rewrite_resize_in_place(model: Any, node: Any, initializers: Dict[str, Any]) -> None:
    """把 [X, "", "", sizes] 形式的 ``Resize`` 重写为 opset-10 两输入 [X, scales]。

    X2Paddle 1.6.0 解析 4 输入 Resize 时会跳过空占位 scale(``build_connection``
    里 ``in_node == ''`` 直接 ``continue``), 导致 ``node.inputs`` 里实际下标右移,
    随后 ``_interpolate`` 用 ``get_input_node(idx=3)`` 取 sizes 时 ``IndexError``。
    这里把 scale 固化为常量并收敛为两输入, 走 x2paddle 的 ``len(input) == 2`` 分支。
    """
    import onnx

    if len(node.input) not in (3, 4):
        return
    x_name = node.input[0]
    sizes_name = node.input[3] if len(node.input) == 4 else None
    scales_name = next(
        (i for i in (node.input[2] if len(node.input) >= 3 else None, node.input[1]) if i), None
    )

    scale_val: Optional[List[float]] = None
    if sizes_name and sizes_name in initializers:
        sizes = [float(v) for v in onnx.numpy_helper.to_array(initializers[sizes_name]).tolist()]
        x_shape = _static_shape(model, x_name)
        if len(x_shape) == 4 and x_shape[2] > 0 and x_shape[3] > 0:
            scale_val = [1.0, 1.0, sizes[2] / x_shape[2], sizes[3] / x_shape[3]]
    if scale_val is None and scales_name and scales_name in initializers:
        arr = [float(v) for v in onnx.numpy_helper.to_array(initializers[scales_name]).tolist()]
        if len(arr) == 4:
            scale_val = arr
    if scale_val is None:
        return

    scale_name = (node.name + "_scales").replace("/", "_")
    const = onnx.helper.make_tensor(
        scale_name, onnx.TensorProto.FLOAT, [4], scale_val
    )
    model.graph.initializer.append(const)
    initializers[scale_name] = const
    del node.input[:]
    node.input.extend([x_name, scale_name])


def _has_attribute(node, name: str) -> bool:
    return any(attr.name == name for attr in node.attribute)


def _onnx_to_paddle(onnx_path: str, save_dir: str) -> None:
    try:
        from x2paddle.convert import onnx2paddle
    except ImportError as err:
        raise BackendNotAvailableError("缺少 X2Paddle 依赖: {}".format(err))
    try:
        onnx2paddle(onnx_path, save_dir)
    except Exception as err:
        raise CompileError("X2Paddle ONNX -> Paddle 转换失败: {}".format(err))


def _locate_paddle_pair(root: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """在 X2Paddle 输出目录中定位 (program, params) 文件对。"""
    for prog in sorted(root.rglob("*.pdmodel")):
        params = prog.with_suffix(".pdiparams")
        if params.is_file():
            return prog, params
    for prog in sorted(root.rglob("*.json")):
        params = prog.with_suffix(".pdiparams")
        if params.is_file():
            return prog, params
    return None, None


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


# 自动探测顺序: 优先能产出真实设备无关静态图的 Paddle 路径
ADAPTER_REGISTRY: List[type] = [PaddleXpuSdkAdapter, XpuToolkitSdkAdapter]


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
        """是否存在可直接产出真实产物的 SDK。"""
        adapter, degraded = self._resolve_adapter()
        return adapter is not None and not degraded

    # ------------------------------------------------------------------ 分析
    def analyze(self, graph, precision: Optional[str] = None) -> OperatorAnalysis:
        return self.registry.analyze(graph, precision=precision or self.config.precision)

    # ------------------------------------------------------------------ 编译
    def compile(self, xpu_graph: XpuGraph, output_path: str, config: Optional[KunlunConfig] = None) -> BackendArtifact:
        config = config or self.config
        adapter, degraded = self._resolve_adapter(config)
        if adapter is None:
            raise BackendNotAvailableError(
                "未检测到可用的昆仑 SDK 后端适配器(Paddle Inference / XPU Toolkit), "
                "且未允许降级生成占位产物"
            )
        if degraded and not config.allow_degraded:
            raise BackendNotAvailableError(
                "未检测到真实昆仑后端(Paddle Inference / XPU Toolkit), 且 allow_degraded=False, "
                "拒绝生成占位产物。请安装 paddlepaddle-xpu + x2paddle, 或显式使用 --allow-degraded 联调"
            )

        artifact = adapter.compile(xpu_graph, output_path, config)
        artifact.metadata.setdefault("converter_version", CONVERTER_VERSION)
        artifact.metadata.setdefault("target_chip", config.target_chip)
        artifact.metadata.setdefault("graph", xpu_graph.to_dict())
        if degraded:
            artifact.notes.append("sdk_adapter=stub: 未接入真实昆仑后端, 产物仅供链路联调")
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
                "sdk_adapter={} 在当前环境不可用(缺少对应 SDK/依赖)".format(requested)
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
