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
    """Paddle Inference + XPU 会话。

    TODO(SDK): Paddle XPU 的 device 与 precision 开关在不同昆仑镜像版本中存在
    差异, 待确认镜像版本后在此补全。
    """

    backend_name = "paddle-xpu"

    def __init__(self, model_path: str, device: str = "auto") -> None:
        try:
            from paddle import inference
        except ImportError as err:  # pragma: no cover - 依赖缺失
            raise BackendNotAvailableError("缺少 paddle 依赖: {}".format(err))
        if not hasattr(inference, "Config"):  # pragma: no cover
            raise BackendNotAvailableError("当前 paddle 版本不支持 inference.Config")
        self._inference = inference
        self._model_path = model_path
        self._device = device
        self._predictor = self._create_predictor()

    def _create_predictor(self):
        config = self._inference.Config(self._model_path)
        config.disable_glog_info()
        if str(self._device).lower() != "cpu" and hasattr(config, "enable_xpu"):
            # TODO(SDK): 依据实际昆仑镜像版本确认 xpu 设备号获取方式
            config.enable_xpu(0)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(4)
        return self._inference.create_predictor(config)

    @property
    def input_names(self) -> List[str]:
        return list(self._predictor.get_input_names())

    @property
    def output_names(self) -> List[str]:
        return list(self._predictor.get_output_names())

    def run(self, inputs: Dict[str, Any]) -> List[Any]:
        import numpy as np

        for name, value in inputs.items():
            handle = self._predictor.get_input_handle(name)
            handle.copy_from_cpu(np.asarray(value))
        self._predictor.run()
        return [self._predictor.get_output_handle(name).copy_to_cpu() for name in self.output_names]

    def close(self) -> None:
        self._predictor = None
