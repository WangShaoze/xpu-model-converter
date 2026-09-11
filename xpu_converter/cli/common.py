# -*- coding: utf-8 -*-
"""CLI 公共参数与辅助函数。

命令签名对齐建设目标 §13; 其中 ``--hardware`` 为 §2 的正式写法, ``--device``
作为等价别名保留, 以免与"执行设备"(``--exec-device auto|xpu|cpu``)混淆。
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from xpu_converter.backend.base import BackendArtifact, BaseBackend, BaseRuntimeSession
from xpu_converter.backend.kunlun.runtime import OnnxRuntimeSession
from xpu_converter.config import HardwareConfig, normalize_shape
from xpu_converter.errors import ConfigError
from xpu_converter.registry.backend_registry import (
    available_backends,
    create_backend,
    resolve_hardware_config,
)
from xpu_converter.registry.model_registry import available_model_types, create_adapter, detect_model_type

DEFAULT_HARDWARE = "kunlun"
METADATA_FILENAME = "metadata.json"


# --------------------------------------------------------------------- 参数
def add_hardware_args(parser: argparse.ArgumentParser, with_precision: bool = True) -> argparse.ArgumentParser:
    """添加硬件/后端相关参数。

    默认值留空, 由 :func:`hardware_config_from` 决定回退顺序(manifest > YAML > kunlun)。
    """
    parser.add_argument(
        "--hardware", "--device", dest="hardware", default=None, metavar="HARDWARE",
        help="目标硬件后端, 当前支持: {}; --device 为等价写法".format(", ".join(available_backends())),
    )
    if with_precision:
        parser.add_argument("--precision", choices=("fp32", "fp16"), default=None, help="编译精度")
    parser.add_argument(
        "--exec-device", dest="exec_device", default=None, choices=("auto", "xpu", "cpu"),
        help="执行设备(与目标硬件区分): auto|xpu|cpu",
    )
    parser.add_argument(
        "--sdk-adapter", dest="sdk_adapter", default=None, choices=("auto", "paddle", "xpuctl", "stub"),
        help="昆仑 SDK 适配器; 缺省按硬件配置(auto)",
    )
    parser.add_argument("--target-chip", dest="target_chip", default=None, help="昆仑芯型号, 如 R200/R300")
    return parser


def add_model_args(parser: argparse.ArgumentParser, required: bool = True,
                   positional: bool = False) -> argparse.ArgumentParser:
    """添加模型输入相关参数。

    ``inspect`` / ``analyze`` 按建设目标 §13 用位置参数(``inspect best.pt``),
    其余子命令用 ``--model``。
    """
    help_text = "输入模型: .pt/.pth 框架模型, 或 .onnx 中间模型"
    if positional:
        parser.add_argument("model", help=help_text)
    else:
        parser.add_argument("--model", required=required, help=help_text)
    parser.add_argument(
        "--model-type", dest="model_type", default=None,
        help="模型类型, 缺省按文件名推断; 支持: {}".format(", ".join(available_model_types())),
    )
    return parser


def input_shape_from(text: Optional[str]) -> Optional[List[int]]:
    return normalize_shape(text) if text else None


def resolve_model_type(model_path: str, explicit: Optional[str]) -> str:
    return detect_model_type(model_path, explicit)


def adapter_for(model_path: str, model_type: Optional[str], overrides: Optional[Dict[str, Any]] = None):
    overrides = dict(overrides or {})
    overrides.setdefault("model_type", resolve_model_type(model_path, model_type))
    return create_adapter(overrides["model_type"], overrides)


# --------------------------------------------------------------------- 硬件
def hardware_config_from(args: argparse.Namespace, hardware: Optional[str] = None) -> HardwareConfig:
    """由命令行参数派生硬件配置(命令行覆盖 YAML 默认值)。"""
    overrides: Dict[str, Any] = {}
    for key, attr in (("precision", "precision"), ("device", "exec_device"),
                      ("sdk_adapter", "sdk_adapter"), ("target_chip", "target_chip")):
        value = getattr(args, attr, None)
        if value:
            overrides[key] = value
    name = hardware or getattr(args, "hardware", None) or DEFAULT_HARDWARE
    return resolve_hardware_config(name, overrides)


def backend_for(args: argparse.Namespace, hardware_config: Optional[HardwareConfig] = None) -> BaseBackend:
    hardware_config = hardware_config or hardware_config_from(args)
    return create_backend(hardware_config.name, hardware_config)


# ------------------------------------------------------------------ 产物元数据
def load_sidecar_metadata(model_path: str) -> Dict[str, Any]:
    """读取编译产物旁的 ``metadata.json``(优先同名 json, 其次同目录 metadata.json)。"""
    path = Path(model_path)
    candidates = [path.with_suffix(".json"), path.parent / METADATA_FILENAME]
    for candidate in candidates:
        if candidate.is_file():
            try:
                with open(candidate, "r", encoding="utf-8") as fr:
                    data = json.load(fr)
                if isinstance(data, dict):
                    return data
            except (OSError, ValueError):
                continue
    return {}


def artifact_from_model(model_path: str, precision: str = "fp16", target_chip: str = "auto") -> BackendArtifact:
    """由已有产物文件还原 :class:`BackendArtifact`。

    没有编译元数据(``metadata.json``)时一律按**占位产物**处理, 避免把 ONNX
    中间模型当作真实 XPU 产物打包交付(建设目标 §11 / §15)。
    """
    path = Path(model_path)
    metadata = load_sidecar_metadata(str(path))
    notes = list(metadata.get("notes") or [])
    artifact_format = str(metadata.get("artifact_format") or "")
    if not metadata:
        notes.append("未找到编译元数据 metadata.json, 按占位产物处理, 请勿直接对外交付")
    elif path.suffix.lower() == ".onnx":
        notes.append("输入为 ONNX 中间模型而非 XPU 编译产物, 已按占位产物处理")
    if path.suffix.lower() == ".onnx":
        artifact_format = "stub"
    return BackendArtifact(
        model_path=str(path),
        precision=str(metadata.get("precision") or precision),
        sdk_adapter=str(metadata.get("sdk_adapter") or "stub"),
        target_chip=str(metadata.get("target_chip") or target_chip),
        artifact_format=artifact_format or "placeholder",
        metadata=dict(metadata.get("metadata") or {}),
        notes=notes,
    )


def target_session(backend: BaseBackend, artifact: BackendArtifact,
                   hardware_config: HardwareConfig) -> BaseRuntimeSession:
    """为待验证产物创建推理会话。"""
    if Path(artifact.model_path).suffix.lower() == ".onnx":
        return OnnxRuntimeSession(artifact.model_path, device=hardware_config.device)
    return backend.create_runtime(artifact)


def reference_session(source: str, model_type: Optional[str], input_shape: Optional[List[int]],
                      workdir: Path) -> BaseRuntimeSession:
    """准备基准会话: ``.onnx`` 直接加载, 框架模型先导出为 ONNX。"""
    path = Path(source)
    if not path.is_file():
        raise ConfigError("基准模型不存在: {}".format(source))
    if path.suffix.lower() == ".onnx":
        return OnnxRuntimeSession(str(path), device="cpu")
    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / "{}_reference.onnx".format(resolve_model_type(str(path), model_type))
    adapter = adapter_for(str(path), model_type, {"input_shape": list(input_shape)} if input_shape else None)
    exported = adapter.export_onnx(
        str(path), str(target), input_shape=input_shape or adapter.input_shape(), opset=adapter.opset(), dynamic=False
    )
    return OnnxRuntimeSession(str(exported), device="cpu")


# --------------------------------------------------------------------- 输出
def print_kv(title: str, data: Dict[str, Any]) -> None:
    print(title)
    width = max((len(str(key)) for key in data), default=0)
    for key, value in data.items():
        print("  {:<{width}} : {}".format(str(key), _render(value), width=width))


def _render(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) if value else "-"
    if isinstance(value, dict):
        return ", ".join("{}={}".format(k, v) for k, v in value.items()) if value else "-"
    return "-" if value in (None, "") else str(value)


def save_json(data: Dict[str, Any], path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fw:
        json.dump(data, fw, ensure_ascii=False, indent=2)
    return str(target)
