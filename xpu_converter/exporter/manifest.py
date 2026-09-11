# -*- coding: utf-8 -*-
"""交付包元信息模型。

交付包采用"算法同名目录平铺"布局(对齐客户标准包), 包名、镜像名、容器名、端口
等派生字段由 :class:`PackageManifest` 统一持有, 供 Dockerfile / build.sh /
readme.txt 模板渲染使用。
"""
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from xpu_converter.version import CONVERTER_VERSION, RUNTIME_API_VERSION

ARTIFACT_MANIFEST_FILENAME = "artifact.json"

# 产物文件名约定
ARTIFACT_FILENAME = "model.xpu"
MODEL_YAML_FILENAME = "model.yaml"
METADATA_FILENAME = "metadata.json"
CONFIDENCE_FILENAME = "confidence.json"
RUNTIME_YAML_FILENAME = "runtime.yaml"


def sha256_file(path) -> str:
    """计算文件 SHA256。"""
    digest = hashlib.sha256()
    with open(path, "rb") as fr:
        for chunk in iter(lambda: fr.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class PackageManifest:
    """交付包清单。"""

    package_name: str = "model-dockerimg_v1.0"
    package_version: str = "v1.0"
    model_name: str = "model"
    model_version: str = "custom"
    framework: str = "pytorch"
    task: str = "detection"
    hardware: str = "kunlun"
    precision: str = "fp16"
    input: Dict[str, Any] = field(default_factory=dict)
    runtime: Dict[str, Any] = field(default_factory=dict)
    conversion: Dict[str, Any] = field(default_factory=dict)
    source: Dict[str, Any] = field(default_factory=dict)
    target: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    benchmark: Dict[str, Any] = field(default_factory=dict)
    image: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ 派生字段
    @property
    def degraded(self) -> bool:
        return bool((self.conversion or {}).get("degraded"))

    @property
    def image_name(self) -> str:
        return str((self.image or {}).get("name") or self.model_name)

    @property
    def image_tag(self) -> str:
        return str((self.image or {}).get("tag") or self.package_version)

    @property
    def image_full(self) -> str:
        return "{}:{}".format(self.image_name, self.image_tag)

    @property
    def container_name(self) -> str:
        return "{}_{}".format(self.image_name, self.image_tag)

    @property
    def image_tar(self) -> str:
        """包内实际存在的镜像 tar 文件名(不存在时返回空串, README 据此决定是否描述)。"""
        return str((self.image or {}).get("tar") or "")

    @property
    def web_port(self) -> int:
        return int((self.runtime or {}).get("port") or 58025)

    @property
    def api_version(self) -> str:
        return str((self.runtime or {}).get("api_version") or RUNTIME_API_VERSION)

    @property
    def zip_name(self) -> str:
        return "{}.zip".format(self.package_name)


def build_manifest(
    name: str,
    version: str = "v1.0",
    model_type: str = "yolov10",
    framework: str = "pytorch",
    task: str = "detection",
    hardware: str = "kunlun",
    precision: str = "fp16",
    input_spec: Optional[Dict[str, Any]] = None,
    runtime_spec: Optional[Dict[str, Any]] = None,
    artifact: Optional[Any] = None,
    model_file: Optional[str] = None,
    source: Optional[Dict[str, Any]] = None,
    validation: Optional[Dict[str, Any]] = None,
    benchmark: Optional[Dict[str, Any]] = None,
    image: Optional[Dict[str, Any]] = None,
    notes: Optional[List[str]] = None,
) -> PackageManifest:
    """构造交付包元信息(包名遵循客户标准 ``<算法名>-dockerimg_<版本>``)。"""
    conversion = {
        "converter_version": CONVERTER_VERSION,
        "sdk_adapter": getattr(artifact, "sdk_adapter", "unknown"),
        "artifact_format": getattr(artifact, "artifact_format", ""),
        "degraded": bool(getattr(artifact, "degraded", False)),
        "notes": list(getattr(artifact, "notes", []) or []),
    }
    return PackageManifest(
        package_name="{}-dockerimg_{}".format(name, version),
        package_version=version,
        model_name=name,
        model_version=version,
        framework=framework,
        task=task,
        hardware=hardware,
        precision=precision,
        input=dict(input_spec or {}),
        runtime=dict(runtime_spec or {}),
        conversion=conversion,
        source=dict(source or {}),
        target={"hardware": hardware, "precision": precision,
                "target_chip": getattr(artifact, "target_chip", "auto"),
                "model_file": model_file or getattr(artifact, "model_file", "") or ARTIFACT_FILENAME},
        validation=dict(validation or {}),
        benchmark=dict(benchmark or {}),
        image=dict(image or {"name": name, "tag": version, "tar": ""}),
        notes=list(notes or []),
    )


def build_artifact_manifest(
    artifact: Optional[Any],
    model_files: Optional[List[Any]] = None,
    model_config: Optional[Any] = None,
    hardware_config: Optional[Any] = None,
    source: Optional[Dict[str, Any]] = None,
    optimization: Optional[Dict[str, Any]] = None,
    hardware: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构造 ``artifact.json``(ChatGPT 修改意见 §29 / §30)。

    ``artifact.json`` 只描述**产物自身的来源与编译信息**: 从哪个 checkpoint
    (含 SHA256)、用什么导出/优化/后端编译出来的, 以及最终产物的文件与 SHA256。
    关系::

        build.yaml(model.yaml) -> artifact.json(随算法目录一并交付)
    """
    files = [Path(item) for item in (model_files or []) if item]
    primary = files[0] if files else None
    source_spec = dict(source or {})
    if hardware is None:
        hardware_spec: Dict[str, Any] = {}
    elif hasattr(hardware, "to_dict"):
        hardware_spec = dict(hardware.to_dict())
    else:
        hardware_spec = dict(hardware)

    def _value(*candidates):
        for candidate in candidates:
            if candidate:
                return candidate
        return ""

    artifact_files = [
        {"file": path.name, "sha256": sha256_file(path)}
        for path in files if path.is_file()
    ]
    return {
        "artifact_format": getattr(artifact, "artifact_format", "") or "",
        "converter": {"name": "xpu-model-converter", "version": CONVERTER_VERSION},
        "source": {
            "framework": _value(source_spec.get("framework"),
                                 getattr(model_config, "framework", "")),
            "framework_version": source_spec.get("source_framework_version", ""),
            "model_type": _value(source_spec.get("model_type"),
                                 getattr(model_config, "model_type", "")),
            "checkpoint": source_spec.get("checkpoint", ""),
            "model_sha256": source_spec.get("model_sha256", ""),
        },
        "export": {
            "onnx_opset": int(getattr(model_config, "opset", 0) or 0),
            "input_shape": list(getattr(model_config, "input_shape", []) or []),
        },
        "optimization": dict(optimization or {}),
        "backend": {
            "name": getattr(hardware_config, "name", "") or "",
            "chip": _value(hardware_spec.get("chip"),
                           getattr(hardware_config, "target_chip", "")),
            "sdk_version": hardware_spec.get("sdk_version", ""),
            "compiler_version": hardware_spec.get("compiler_version", ""),
        },
        "precision": _value(getattr(artifact, "precision", ""),
                            getattr(hardware_config, "precision", "")),
        "artifact": {
            "file": primary.name if primary else "",
            "sha256": artifact_files[0]["sha256"] if artifact_files else "",
            "files": artifact_files,
        },
    }
