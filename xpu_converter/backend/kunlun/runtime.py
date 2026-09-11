# -*- coding: utf-8 -*-
"""昆仑芯推理会话(SDK 解耦)。

三类会话:

- :class:`XpuToolkitRuntimeSession` : 真实昆仑 XPU 运行时(需 SDK, 待落地)
- :class:`PaddleXpuRuntimeSession`  : Paddle Inference + XPU
- :class:`OnnxRuntimeSession`       : 无 XPU 环境下的仿真/校验会话

三者都实现 :class:`xpu_converter.backend.base.BaseRuntimeSession`, 因此
Validator 与 Runtime 服务不需要区分后端类型。
"""
import importlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from xpu_converter.backend.base import BaseRuntimeSession
from xpu_converter.errors import BackendNotAvailableError


class OnnxRuntimeSession(BaseRuntimeSession):
    """基于 onnxruntime 的会话。

    在 SDK 未就绪时承担两个职责:
    1. 作为占位产物(stub)的推理后端, 打通 Runtime 交付链路;
    2. 作为 Validator 的 ONNX 侧基准实现。
    """

    backend_name = "onnxruntime"

    def __init__(self, model_path: str, device: str = "auto", providers: Optional[List[str]] = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as err:  # pragma: no cover - 依赖缺失
            raise BackendNotAvailableError("缺少 onnxruntime 依赖: {}".format(err))
        try:
            import numpy  # noqa: F401  确保 numpy 可用, ORT 输出为 ndarray
        except ImportError as err:  # pragma: no cover
            raise BackendNotAvailableError("缺少 numpy 依赖: {}".format(err))

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=self._resolve_providers(ort, device, providers),
        )

    @staticmethod
    def _resolve_providers(ort, device: str, providers: Optional[List[str]]) -> List[str]:
        if providers:
            return list(providers)
        if str(device).lower() in ("cpu",):
            return ["CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        for candidate in ("CUDAExecutionProvider", "CPUExecutionProvider"):
            if candidate in available:
                return [candidate]
        return ["CPUExecutionProvider"]

    @property
    def input_names(self) -> List[str]:
        return [item.name for item in self._session.get_inputs()]

    @property
    def output_names(self) -> List[str]:
        return [item.name for item in self._session.get_outputs()]

    def input_shapes(self) -> Dict[str, tuple]:
        return {item.name: tuple(item.shape) for item in self._session.get_inputs()}

    def run(self, inputs: Dict[str, Any]) -> List[Any]:
        return list(self._session.run(None, dict(inputs)))

    def close(self) -> None:
        self._session = None


class XpuToolkitRuntimeSession(BaseRuntimeSession):
    """昆仑 XPU Toolkit 运行时。

    TODO(SDK): 拿到昆仑 SDK 后补全真实运行时调用。当前实现保留探测路径,
    找不到入口时给出明确错误, 避免静默产生错误结果。
    """

    backend_name = "kunlun-xpu"

    def __init__(self, model_path: str, module_name: str, device: str = "auto") -> None:
        self._model_path = model_path
        self._module = importlib.import_module(module_name)
        runtime_cls = None
        for name in ("XpuRuntime", "Runtime", "XpuInference", "XNNToolkitRuntime"):
            runtime_cls = getattr(self._module, name, None)
            if runtime_cls is not None:
                break
        if runtime_cls is None:
            raise BackendNotAvailableError(
                "模块 {} 中未找到 XPU 运行时入口(期望 XpuRuntime/Runtime), "
                "请在 XpuToolkitRuntimeSession 中按实际 SDK 落地".format(module_name)
            )
        # TODO(SDK): 依据实际 SDK 签名补全设备/精度参数
        self._runtime = runtime_cls(model_path)

    @property
    def input_names(self) -> List[str]:
        return list(getattr(self._runtime, "input_names", []) or [])

    @property
    def output_names(self) -> List[str]:
        return list(getattr(self._runtime, "output_names", []) or [])

    def run(self, inputs: Dict[str, Any]) -> List[Any]:
        runner = getattr(self._runtime, "run", None) or getattr(self._runtime, "infer", None)
        if runner is None:
            raise BackendNotAvailableError("XPU 运行时未提供 run/infer 方法, 待按实际 SDK 落地")
        return list(runner(inputs))

    def close(self) -> None:
        closer = getattr(self._runtime, "close", None) or getattr(self._runtime, "destroy", None)
        if callable(closer):
            closer()
        self._runtime = None


class PaddleXpuRuntimeSession(BaseRuntimeSession):
    """Paddle Inference 会话(XPU / CPU)。

    交付产物是**设备无关**的 Paddle 静态图(``model.pdmodel`` + ``model.pdiparams``),
    由 Paddle 在**加载时**按设备编译 XPU kernel:

    * ``DEVICE=xpu``  : ``enable_xpu()`` + ``set_xpu_device_id()``
    * ``DEVICE=cpu``  : 关闭 GPU/XPU, 走 CPU 数学库
    * ``DEVICE=auto`` : 仅当本机确实有可用 XPU 卡时才用 XPU, 否则退 CPU
    """

    backend_name = "paddle-xpu"

    def __init__(self, model_path: str, device: str = "auto", params_path: Optional[str] = None) -> None:
        try:
            from paddle import inference
        except ImportError as err:  # pragma: no cover - 依赖缺失
            raise BackendNotAvailableError("缺少 paddle 依赖: {}".format(err))
        if not hasattr(inference, "Config"):  # pragma: no cover
            raise BackendNotAvailableError("当前 paddle 版本不支持 inference.Config")
        self._inference = inference
        self._model_path = model_path
        self._params_path = params_path or self._resolve_params(model_path)
        self._device = self._resolve_device(device)
        self._predictor = self._create_predictor()

    @staticmethod
    def _resolve_params(model_path: str) -> Optional[str]:
        """Paddle 静态图参数文件与 program 同名同目录。"""
        candidate = Path(model_path).with_suffix(".pdiparams")
        return str(candidate) if candidate.is_file() else None

    @staticmethod
    def _xpu_available() -> bool:
        """本机是否存在可用的昆仑 XPU 卡。"""
        try:
            import paddle

            if not bool(getattr(paddle.device, "is_compiled_with_xpu", lambda: False)()):
                return False
            xpu = getattr(paddle.device, "xpu", None)
            counter = getattr(xpu, "device_count", None)
            if callable(counter):
                return int(counter()) > 0
            return os.path.exists("/dev/xpuctrl")
        except Exception:
            return os.path.exists("/dev/xpuctrl")

    def _resolve_device(self, device: str) -> str:
        value = str(device or "auto").lower()
        if value in ("xpu", "kunlun"):
            return "xpu"
        if value == "cpu":
            return "cpu"
        return "xpu" if self._xpu_available() else "cpu"

    @staticmethod
    def _device_id() -> int:
        value = os.environ.get("GPU_ID") or os.environ.get("XPU_DEVICE_ID") or ""
        return int(value) if str(value).strip().isdigit() else 0

    def _create_predictor(self):
        if self._params_path:
            config = self._inference.Config(self._model_path, self._params_path)
        else:
            config = self._inference.Config(self._model_path)
        config.disable_glog_info()
        if self._device == "xpu":
            if not hasattr(config, "enable_xpu"):
                raise BackendNotAvailableError(
                    "当前 Paddle 未编译 XPU 支持, 无法加载 XPU 产物: {}".format(self._model_path)
                )
            config.disable_mkldnn()
            config.enable_xpu()
            if hasattr(config, "set_xpu_device_id"):
                config.set_xpu_device_id(self._device_id())
        else:
            config.disable_gpu()
            if hasattr(config, "disable_xpu"):
                config.disable_xpu()
            config.set_cpu_math_library_num_threads(4)
        return self._inference.create_predictor(config)

    @property
    def input_names(self) -> List[str]:
        return list(self._predictor.get_input_names())

    @property
    def output_names(self) -> List[str]:
        return list(self._predictor.get_output_names())

    def input_shapes(self) -> Dict[str, tuple]:
        shapes: Dict[str, tuple] = {}
        for name in self.input_names:
            try:
                shapes[name] = tuple(self._predictor.get_input_handle(name).shape())
            except Exception:
                continue
        return shapes

    def run(self, inputs: Dict[str, Any]) -> List[Any]:
        import numpy as np

        for name, value in inputs.items():
            handle = self._predictor.get_input_handle(name)
            handle.copy_from_cpu(np.asarray(value))
        self._predictor.run()
        return [self._predictor.get_output_handle(name).copy_to_cpu() for name in self.output_names]

    def close(self) -> None:
        self._predictor = None
