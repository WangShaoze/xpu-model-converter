# -*- coding: utf-8 -*-
"""``manifest.json`` 规范(建设目标 §6)。

manifest.json 是交付包的**唯一事实来源**: 包名、镜像名、容器名、端口、文件清单
与校验和全部由它派生, README 与 Dockerfile 也由同一份 metadata 渲染。
"""
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from xpu_converter.version import CONVERTER_VERSION, RUNTIME_API_VERSION

MANIFEST_FILENAME = "manifest.json"

# 产物文件名约定
ARTIFACT_FILENAME = "model.xpu"
MODEL_YAML_FILENAME = "model.yaml"
METADATA_FILENAME = "metadata.json"
CONFIDENCE_FILENAME = "confidence.json"
RUNTIME_YAML_FILENAME = "runtime.yaml"
RUNTIME_TGZ_FILENAME = "runtime.tgz"


def sha256_file(path) -> str:
    """计算文件 SHA256。"""
    digest = hashlib.sha256()
    with open(path, "rb") as fr:
        for chunk in iter(lambda: fr.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(root, exclude: Optional[List[str]] = None) -> List[str]:
    """收集目录下所有文件(相对路径, POSIX 分隔符)。"""
    root_path = Path(root)
    excluded = set(exclude or [])
    files: List[str] = []
    for path in sorted(root_path.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root_path).as_posix()
        if relative in excluded or relative.endswith(".pyc"):
            continue
        files.append(relative)
    return files


def build_checksums(root, files: List[str]) -> Dict[str, str]:
    root_path = Path(root)
    checksums: Dict[str, str] = {}
    for relative in files:
        path = root_path / relative
        if path.is_file():
            checksums[relative] = sha256_file(path)
    return checksums


@dataclass
class PackageManifest:
    """交付包清单。"""

    package_name: str = "model_dockerimg_v1.0"
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
    runtime_package: Dict[str, Any] = field(default_factory=dict)
    files: List[str] = field(default_factory=list)
    checksums: Dict[str, str] = field(default_factory=dict)
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

    # ------------------------------------------------------------------ 序列化
    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "package_name": self.package_name,
            "package_version": self.package_version,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "framework": self.framework,
            "task": self.task,
            "hardware": self.hardware,
            "precision": self.precision,
            "input": dict(self.input),
            "runtime": dict(self.runtime),
            "conversion": dict(self.conversion),
            "source": dict(self.source),
            "target": dict(self.target),
            "validation": dict(self.validation),
            "benchmark": dict(self.benchmark),
            "image": dict(self.image),
        }
        if self.runtime_package:
            payload["runtime_package"] = dict(self.runtime_package)
        payload["degraded"] = self.degraded
        payload["files"] = list(self.files)
        payload["checksums"] = dict(self.checksums)
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def save(self, path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # newline="\n": manifest.json 随交付包分发到 Linux, 统一 LF
        target.write_text(self.to_json(), encoding="utf-8", newline="\n")
        return str(target)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PackageManifest":
        data = dict(data or {})
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)

    @classmethod
    def load(cls, path) -> "PackageManifest":
        with open(path, "r", encoding="utf-8") as fr:
            return cls.from_dict(json.load(fr) or {})


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
    source: Optional[Dict[str, Any]] = None,
    validation: Optional[Dict[str, Any]] = None,
    benchmark: Optional[Dict[str, Any]] = None,
    runtime_package: Optional[Dict[str, Any]] = None,
    image: Optional[Dict[str, Any]] = None,
    notes: Optional[List[str]] = None,
) -> PackageManifest:
    """构造交付包清单, 字段顺序与建设目标 §6 的示例保持一致。"""
    conversion = {
        "converter_version": CONVERTER_VERSION,
        "sdk_adapter": getattr(artifact, "sdk_adapter", "unknown"),
        "artifact_format": getattr(artifact, "artifact_format", ""),
        "degraded": bool(getattr(artifact, "degraded", False)),
        "notes": list(getattr(artifact, "notes", []) or []),
    }
    return PackageManifest(
        package_name="{}_dockerimg_{}".format(name, version),
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
                "model_file": ARTIFACT_FILENAME},
        validation=dict(validation or {}),
        benchmark=dict(benchmark or {}),
        image=dict(image or {"name": name, "tag": version, "tar": ""}),
        runtime_package=dict(runtime_package or {}),
        notes=list(notes or []),
    )
