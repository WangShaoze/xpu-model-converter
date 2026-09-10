# -*- coding: utf-8 -*-
"""运行时推理引擎。

按交付包内 ``model/metadata.json`` 记录的 ``sdk_adapter`` 选择加载方式:

======================  ==========================================
sdk_adapter             加载方式
======================  ==========================================
``stub``                占位产物(实为 ONNX), 用 onnxruntime 加载
``xpuctl``              昆仑 XPU Toolkit 运行时
``paddle``              Paddle Inference(XPU)
======================  ==========================================

``DEVICE=cpu`` 或未检测到加速卡时自动降级 CPU(可用但较慢), 与客户现场行为一致。
"""
import importlib
import os
from typing import Any, Dict, List, Optional

import numpy as np

from runtime_config import load_model_metadata, load_runtime_config, model_path

XPU_TOOLKIT_MODULES = ("xpu_toolkit", "xtcl", "xpu_inference")


def xpu_device_count() -> int:
    """探测昆仑 XPU 卡数量(无 SDK 时返回 0)。"""
    for name in XPU_TOOLKIT_MODULES:
        try:
            module = importlib.import_module(name)
        except Exception:
            continue
        for attr in ("device_count", "get_device_count", "xpu_device_count"):
            counter = getattr(module, attr, None)
            if callable(counter):
                try:
                    return int(counter())
                except Exception:
                    continue
    if os.path.exists("/dev/xpuctrl"):
        return 1
    return 0


class RuntimeEngine:
    """统一推理引擎。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None, device: Optional[str] = None) -> None:
        self.config = config or load_runtime_config()
        self.metadata = load_model_metadata(self.config)
        self.model_file = model_path(self.config)
        self.device = (device or (self.config.get("runtime") or {}).get("device") or "auto").lower()
        self.backend = ""
        self.degraded = False
        self._session = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._input_shapes: Dict[str, tuple] = {}

    # ------------------------------------------------------------------ 加载
    def load(self) -> "RuntimeEngine":
        if not os.path.isfile(self.model_file):
            raise RuntimeError("编译模型不存在: {}，请检查交付包 model/ 目录".format(self.model_file))

        adapter = str(self.metadata.get("sdk_adapter") or "").lower()
        artifact_format = str(self.metadata.get("artifact_format") or "").lower()

        if self.device == "cpu" or xpu_device_count() == 0:
            if adapter and adapter != "stub" and artifact_format == "xpu":
                raise RuntimeError(
                    "未检测到昆仑 XPU 加速卡或被显式指定 DEVICE=cpu, "
                    "而交付模型为 XPU 专用产物({}), 无法在 CPU 上加载。".format(self.model_file)
                )
            self._load_onnx_runtime()
            self.degraded = xpu_device_count() == 0
            return self

        if adapter == "stub" or artifact_format == "stub":
            self._load_onnx_runtime()
            self.degraded = True
            return self

        self._load_xpu()
        return self

    def _load_onnx_runtime(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as err:
            raise RuntimeError("缺少 onnxruntime 依赖, 无法以仿真模式加载模型: {}".format(err))
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if self.device != "cpu" and "CUDAExecutionProvider" in set(ort.get_available_providers()):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(self.model_file, sess_options=options, providers=providers)
        self._input_names = [item.name for item in self._session.get_inputs()]
        self._output_names = [item.name for item in self._session.get_outputs()]
        self._input_shapes = {item.name: tuple(item.shape) for item in self._session.get_inputs()}
        self.backend = "onnxruntime"

    def _load_xpu(self) -> None:
        module = None
        for name in XPU_TOOLKIT_MODULES:
            try:
                module = importlib.import_module(name)
                break
            except Exception:
                continue
        if module is None:
            # 交付包里是真实 XPU 产物但容器内没有 SDK: 宁可启动失败也不要给出错误结果
            raise RuntimeError(
                "未找到昆仑 XPU 运行时模块(候选: {}), 无法加载 {}".format(
                    ", ".join(XPU_TOOLKIT_MODULES), self.model_file
                )
            )
        runtime_cls = None
        for name in ("XpuRuntime", "Runtime", "XpuInference"):
            runtime_cls = getattr(module, name, None)
            if runtime_cls is not None:
                break
        if runtime_cls is None:
            raise RuntimeError("昆仑 XPU 运行时模块中未找到可用入口, 请核对 SDK 版本")
        self._session = runtime_cls(self.model_file)
        self._input_names = list(getattr(self._session, "input_names", []) or [])
        self._output_names = list(getattr(self._session, "output_names", []) or [])
        self.backend = "kunlun-xpu"

    # ------------------------------------------------------------------ 推理
    @property
    def input_names(self) -> List[str]:
        return list(self._input_names)

    @property
    def output_names(self) -> List[str]:
        return list(self._output_names)

    def input_shapes(self) -> Dict[str, tuple]:
        return dict(self._input_shapes)

    def run(self, inputs: Dict[str, Any]) -> List[np.ndarray]:
        if self._session is None:
            raise RuntimeError("推理引擎尚未加载模型")
        if self.backend == "onnxruntime":
            return list(self._session.run(None, dict(inputs)))
        runner = getattr(self._session, "run", None) or getattr(self._session, "infer", None)
        if runner is None:
            raise RuntimeError("XPU 运行时未提供 run/infer 方法")
        return [np.asarray(item) for item in runner(dict(inputs))]


_ENGINE: Optional[RuntimeEngine] = None


def get_engine(device: Optional[str] = None) -> RuntimeEngine:
    """进程内单例推理引擎(供 gunicorn 各 worker 独立持有)。"""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RuntimeEngine(device=device).load()
    return _ENGINE
