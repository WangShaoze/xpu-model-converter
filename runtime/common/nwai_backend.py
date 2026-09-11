# -*- coding: utf-8 -*-
"""运行时推理引擎。

按交付包算法目录下 ``metadata.json`` 记录的 ``sdk_adapter`` / ``artifact_format``
选择加载方式:

======================  ==========================================
artifact_format         加载方式
======================  ==========================================
``paddle``              Paddle Inference(XPU / CPU), 静态图 .pdmodel + .pdiparams
``xpu``                 昆仑 XPU Toolkit 运行时(需 XPU_TOOLKIT_MODULE, 预留)
``stub`` / ``onnx``     onnxruntime(仿真/联调产物)
======================  ==========================================

CPU 回退策略(ChatGPT 修改意见 §43/§44): **不做静默降级**。真实 XPU 产物在
``DEVICE=auto`` 且无加速卡时直接启动失败, 避免"部署失败却悄悄用 CPU 跑"的性能悬崖;
仅当产物本身是 ONNX/stub, 或操作者显式 ``DEVICE=cpu`` 时才在 CPU 上运行。
"""
import importlib
import os
from typing import Any, Dict, List, Optional

import numpy as np

from nwai_config import load_model_metadata, load_runtime_config, model_params_path, model_path


def xpu_device_count() -> int:
    """探测昆仑 XPU 卡数量(基于 paddle-xpu 与设备节点, 无则返回 0)。"""
    try:
        import paddle

        if not bool(getattr(paddle.device, "is_compiled_with_xpu", lambda: False)()):
            return 0
        xpu = getattr(paddle.device, "xpu", None)
        counter = getattr(xpu, "device_count", None)
        if callable(counter):
            return max(0, int(counter()))
        return 1 if os.path.exists("/dev/xpuctrl") else 0
    except Exception:
        return 1 if os.path.exists("/dev/xpuctrl") else 0


def _xpu_device_id() -> int:
    """本进程使用的卡号(GPU_ID 由 gunicorn 设备池注入)。"""
    value = os.environ.get("GPU_ID") or os.environ.get("XPU_DEVICE_ID") or ""
    return int(value) if str(value).strip().isdigit() else 0


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
        if not artifact_format and adapter == "stub":
            artifact_format = "stub"
        device_count = xpu_device_count()
        # 真实 XPU 交付产物: xpu(专用) 与 paddle(设备无关静态图, 现场按卡加载)
        requires_xpu = artifact_format in ("xpu", "paddle")

        if requires_xpu:
            if self.device == "cpu":
                if artifact_format == "xpu":
                    raise RuntimeError(
                        "DEVICE=cpu, 而交付模型为 XPU 专用产物({}), 无法在 CPU 上加载; "
                        "如需 CPU 联调请改用 stub/ONNX 产物。".format(self.model_file)
                    )
                self._load_paddle(use_xpu=False)
                return self
            if device_count == 0:
                raise RuntimeError(
                    "未检测到昆仑 XPU 加速卡(DEVICE={}), 拒绝在 CPU 上静默运行 XPU 产物 {}; "
                    "请以 --device=/dev/xpuctrl 透传加速卡, 或显式设置 DEVICE=cpu 进行联调。".format(
                        self.device, self.model_file
                    )
                )
            if artifact_format == "paddle":
                self._load_paddle(use_xpu=True)
            else:
                self._load_xpu_toolkit()
            return self

        # ONNX / stub 仿真产物: 可在 CPU 上运行
        if self.device == "xpu" and device_count == 0:
            raise RuntimeError(
                "DEVICE=xpu 但未检测到昆仑 XPU 加速卡, 无法加载 {}".format(self.model_file)
            )
        self._load_onnx_runtime()
        self.degraded = artifact_format == "stub" or device_count == 0
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

    def _load_paddle(self, use_xpu: bool) -> None:
        """用 Paddle Inference 加载静态图(``.pdmodel`` + ``.pdiparams``)。"""
        try:
            from paddle import inference
        except ImportError as err:
            raise RuntimeError("缺少 paddle 依赖, 无法加载 Paddle 交付产物: {}".format(err))
        params = model_params_path(self.config)
        if params:
            config = inference.Config(self.model_file, params)
        else:
            config = inference.Config(self.model_file)
        config.disable_glog_info()
        if use_xpu:
            if not hasattr(config, "enable_xpu"):
                raise RuntimeError(
                    "当前 Paddle 未编译 XPU 支持, 无法加载 XPU 产物: {}".format(self.model_file)
                )
            config.disable_mkldnn()
            config.enable_xpu()
            if hasattr(config, "set_xpu_device_id"):
                config.set_xpu_device_id(_xpu_device_id())
        else:
            config.disable_gpu()
            if hasattr(config, "disable_xpu"):
                config.disable_xpu()
            config.set_cpu_math_library_num_threads(4)
        self._session = inference.create_predictor(config)
        self._input_names = list(self._session.get_input_names())
        self._output_names = list(self._session.get_output_names())
        self._input_shapes = {}
        for name in self._input_names:
            try:
                self._input_shapes[name] = tuple(self._session.get_input_handle(name).shape())
            except Exception:
                continue
        self.backend = "paddle-xpu" if use_xpu else "paddle"

    def _load_xpu_toolkit(self) -> None:
        """加载昆仑 XPU Toolkit 专用产物(需环境变量 XPU_TOOLKIT_MODULE)。"""
        module_name = (os.environ.get("XPU_TOOLKIT_MODULE") or "").strip()
        module = None
        if module_name:
            try:
                module = importlib.import_module(module_name)
            except Exception as err:
                raise RuntimeError(
                    "无法导入昆仑 XPU Toolkit 模块 {}: {}".format(module_name, err)
                )
        if module is None:
            raise RuntimeError(
                "交付模型为 XPU 专用产物({}), 但容器内未配置昆仑 XPU Toolkit Python 模块; "
                "请设置环境变量 XPU_TOOLKIT_MODULE, 或改用 paddle 适配器产物。".format(self.model_file)
            )
        runtime_cls = None
        for name in ("XpuRuntime", "Runtime", "XpuInference"):
            runtime_cls = getattr(module, name, None)
            if runtime_cls is not None:
                break
        if runtime_cls is None:
            raise RuntimeError("昆仑 XPU 运行时模块 {} 中未找到可用入口, 请核对 SDK 版本".format(module_name))
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
        if self.backend.startswith("paddle"):
            for name, value in inputs.items():
                self._session.get_input_handle(name).copy_from_cpu(np.asarray(value))
            self._session.run()
            return [self._session.get_output_handle(name).copy_to_cpu() for name in self._output_names]
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
