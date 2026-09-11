# -*- coding: utf-8 -*-
"""Docker 交付包导出(建设目标 §6 / §8)。

产出的包结构与 ``yolov9t_dockerimg_v1.0.zip`` 保持交付层兼容, 并补齐
``manifest.json`` / ``runtime.tgz`` / ``model.yaml`` / ``config/runtime.yaml``:

    <package_name>/
    ├── Dockerfile / build.sh / install.conf / readme.txt / start.sh
    ├── packages/
    ├── runtime.tgz
    ├── model/{<产物文件>, model.yaml, metadata.json}
    ├── config/{confidence.json, runtime.yaml}
    └── manifest.json

``model/`` 下的产物文件由编译产物决定(单文件如 ``model.onnx``, Paddle 静态图为
``model.pdmodel`` + ``model.pdiparams``), 全部复制并由 manifest.json 记录。
"""
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from xpu_converter.classes import coco_classes
from xpu_converter.config import BuildManifest, HardwareConfig, ModelConfig, dump_yaml
from xpu_converter.environment import SOURCE_FINGERPRINT_KEYS
from xpu_converter.errors import PackageError
from xpu_converter.exporter.manifest import (
    ARTIFACT_FILENAME,
    ARTIFACT_MANIFEST_FILENAME,
    CONFIDENCE_FILENAME,
    MANIFEST_FILENAME,
    METADATA_FILENAME,
    MODEL_YAML_FILENAME,
    RUNTIME_TGZ_FILENAME,
    RUNTIME_YAML_FILENAME,
    PackageManifest,
    build_artifact_manifest,
    build_checksums,
    build_manifest,
    collect_files,
)
from xpu_converter.exporter.template import TemplateRenderer
from xpu_converter.paths import home_dir
from xpu_converter.paths import runtime_dir
from xpu_converter.version import CONVERTER_VERSION, RUNTIME_API_VERSION

RUNTIME_COMMON_DIR = "common"
DEFAULT_RUNTIME_TASK = "detection"


def artifact_files(model_path: str, artifact: Optional[Any]) -> List[Path]:
    """产物文件清单(主文件在前)。

    Paddle 静态图为 ``model.pdmodel`` + ``model.pdiparams`` 两个文件, 因此交付包
    必须把它们**一起**复制; ``artifact.artifact_files`` 为空时回退到单文件。
    """
    files = list(getattr(artifact, "artifact_files", None) or []) if artifact is not None else []
    paths = [Path(item) for item in files if item]
    if not paths and model_path:
        paths = [Path(model_path)]
    return paths


def artifact_model_filename(model_files: List[Path]) -> str:
    """交付包 ``model/`` 内的主产物文件名(runtime.yaml / manifest 引用它)。"""
    return model_files[0].name if model_files else ARTIFACT_FILENAME


class RuntimePackager:
    """交付产物 ②: 公共 Runtime 打包。

    把 ``runtime/common`` 与 ``runtime/<task>`` 合并为**一套与模型无关的 Runtime**,
    所有 YOLO 模型共用同一份 ``runtime.tgz``(建设目标 §5/§10)。
    """

    def __init__(self, task: str = DEFAULT_RUNTIME_TASK, version: str = "v1.0") -> None:
        self.task = task or DEFAULT_RUNTIME_TASK
        self.version = version

    @property
    def name(self) -> str:
        return "runtime-{}-{}".format(self.task, self.version)

    def source_dirs(self) -> List[Path]:
        root = runtime_dir()
        dirs = [root / RUNTIME_COMMON_DIR, root / self.task]
        for path in dirs:
            if not path.is_dir():
                raise PackageError("Runtime 目录不存在: {}".format(path))
        return dirs

    def stage(self, target_dir) -> List[str]:
        """把 Runtime 源码平铺到 ``target_dir``, 返回相对文件列表。"""
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        for source in self.source_dirs():
            for path in sorted(source.rglob("*")):
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                relative = path.relative_to(source)
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
        return sorted(path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file())

    def build(self, output_path) -> str:
        """生成 ``runtime.tgz``(tar.gz)。"""
        import tarfile
        import tempfile

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            stage_dir = Path(tmp) / self.name
            self.stage(stage_dir)
            with tarfile.open(str(output), "w:gz") as tar:
                tar.add(str(stage_dir), arcname=".")
        return str(output)


class DockerExporter:
    """把编译产物封装成可直接交付的 Docker ZIP。"""

    def __init__(
        self,
        manifest: Optional[BuildManifest] = None,
        model_config: Optional[ModelConfig] = None,
        hardware_config: Optional[HardwareConfig] = None,
        runtime: str = DEFAULT_RUNTIME_TASK,
        packages_dir: Optional[str] = None,
        converter_version: str = CONVERTER_VERSION,
        allow_degraded: bool = False,
    ) -> None:
        self.manifest = manifest or BuildManifest()
        self.model_config = model_config or ModelConfig.from_dict(self.manifest.to_model_override())
        self.hardware_config = hardware_config or HardwareConfig()
        self.runtime = runtime or DEFAULT_RUNTIME_TASK
        if packages_dir is None:
            # 未显式指定时回退到 configs/hardware/*.yaml 的 docker.packages_dir
            packages_dir = (self.hardware_config.docker or {}).get("packages_dir") or None
        self.packages_dir = self._resolve_packages_dir(packages_dir)
        self.converter_version = converter_version
        # 是否允许把 degraded(占位)产物封装成正式交付包。默认禁止, 仅 --dev-package 打开。
        self.allow_degraded = bool(allow_degraded)

    @staticmethod
    def _resolve_packages_dir(value: Optional[str]) -> Optional[str]:
        """规范化为绝对路径: 相对路径以项目根为基准, 目录不存在则视为未配置。"""
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = home_dir() / path
        return str(path) if path.is_dir() else None

    # ------------------------------------------------------------------ 主流程
    def export(
        self,
        model_path: str,
        output_dir: str,
        artifact: Optional[Any] = None,
        onnx_path: Optional[str] = None,
        accuracy: Optional[Any] = None,
        benchmark: Optional[Any] = None,
        image_tar: Optional[str] = None,
        optimization_report: Optional[Any] = None,
        hardware_capability: Optional[Any] = None,
    ) -> str:
        """生成交付包目录与 ZIP, 返回 ZIP 路径。"""
        if not model_path or not Path(model_path).is_file():
            raise PackageError("编译产物不存在, 无法导出交付包: {}".format(model_path))
        if artifact is not None and getattr(artifact, "degraded", False) and not self.allow_degraded:
            raise PackageError(
                "当前为 degraded/占位产物(artifact_format={}), 禁止生成正式 Docker 交付包。"
                "请在拿到真实昆仑后端后重新编译; 若仅用于链路联调, 请显式使用 --dev-package。".format(
                    getattr(artifact, "artifact_format", "") or "unknown"
                )
            )
        model_files = artifact_files(model_path, artifact)
        missing = [str(path) for path in model_files if not path.is_file()]
        if missing:
            raise PackageError("编译产物文件缺失, 无法导出交付包: {}".format(", ".join(missing)))
        model_filename = artifact_model_filename(model_files)

        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        package_name = self.manifest.package_name
        package_dir = root / package_name
        if package_dir.exists():
            shutil.rmtree(package_dir)
        package_dir.mkdir(parents=True)

        runtime_packager = RuntimePackager(task=self.runtime, version=self.manifest.version)
        runtime_tgz = package_dir / RUNTIME_TGZ_FILENAME
        runtime_packager.build(runtime_tgz)

        self._write_model_dir(package_dir, model_files, artifact, onnx_path,
                              optimization_report, hardware_capability)
        self._write_config_dir(package_dir, model_filename)
        self._write_packages_dir(package_dir)

        package_manifest = self._build_package_manifest(package_dir, artifact, accuracy, benchmark,
                                                        runtime_packager, image_tar, model_filename)
        context = self._template_context(package_manifest, model_filename)
        TemplateRenderer(self.hardware_config.name).render_all(package_dir, context)
        self._make_executable(package_dir)

        # files/checksums 覆盖除 manifest.json 之外的全部交付内容
        files = collect_files(package_dir, exclude=[MANIFEST_FILENAME])
        package_manifest.files = files
        package_manifest.checksums = build_checksums(package_dir, files)
        package_manifest.save(package_dir / MANIFEST_FILENAME)

        zip_path = self._zip(package_dir, root)
        return zip_path

    # ------------------------------------------------------------------ 子步骤
    def _write_model_dir(self, package_dir: Path, model_files: List[Path],
                         artifact: Optional[Any], onnx_path: Optional[str],
                         optimization_report: Optional[Any] = None,
                         hardware_capability: Optional[Any] = None) -> None:
        model_dir = package_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        # Paddle 静态图由 .pdmodel + .pdiparams 两个文件组成, 必须一起复制
        for path in model_files:
            shutil.copyfile(str(path), model_dir / path.name)

        metadata: Dict[str, Any] = {
            "model_name": self.manifest.name,
            "model_version": self.manifest.version,
            "framework": self.manifest.framework,
            "task": self.model_config.task,
            "hardware": self.manifest.hardware,
            "precision": self.manifest.precision,
            "converter_version": self.converter_version,
        }
        if artifact is not None and hasattr(artifact, "to_dict"):
            metadata.update(artifact.to_dict())
        # 来源环境指纹(ChatGPT 修改意见 §13 / §31): 现场排障需要, 平铺到 metadata.json 顶层
        source = dict(self.manifest.source or {})
        for key in SOURCE_FINGERPRINT_KEYS:
            if source.get(key):
                metadata[key] = source[key]
        metadata["metadata"] = {
            "input": {"shape": self.model_config.input_shape,
                      "dtype": self.model_config.input_dtype,
                      "layout": self.model_config.input_layout},
            "output_layout": self.model_config.output_layout,
            "output": dict(self.model_config.output),
            "end2end": bool(self.model_config.end2end),
            "num_classes": int(self.model_config.num_classes),
        }
        if onnx_path:
            metadata["metadata"]["onnx_source"] = Path(onnx_path).name
        with open(model_dir / METADATA_FILENAME, "w", encoding="utf-8", newline="\n") as fw:
            json.dump(metadata, fw, ensure_ascii=False, indent=2)

        # artifact.json: 产物自身的来源与编译信息(ChatGPT 修改意见 §29/§30)
        artifact_manifest = build_artifact_manifest(
            artifact=artifact,
            model_files=model_files,
            model_config=self.model_config,
            hardware_config=self.hardware_config,
            source=dict(self.manifest.source or {}),
            optimization=self._optimization_spec(optimization_report),
            hardware=hardware_capability,
        )
        with open(model_dir / ARTIFACT_MANIFEST_FILENAME, "w", encoding="utf-8", newline="\n") as fw:
            json.dump(artifact_manifest, fw, ensure_ascii=False, indent=2)

        # model.yaml: 只描述"如何产生这个模型"(§28), 包内容由 manifest.json 负责
        dump_yaml(self.manifest.to_model_yaml(), model_dir / MODEL_YAML_FILENAME)

    def _optimization_spec(self, optimization_report: Optional[Any]) -> Dict[str, Any]:
        spec: Dict[str, Any] = {"level": int(self.hardware_config.optimization_level)}
        if optimization_report is not None and hasattr(optimization_report, "to_dict"):
            spec.update(optimization_report.to_dict())
        return spec

    def _write_config_dir(self, package_dir: Path, model_filename: str) -> None:
        config_dir = package_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        class_names = list(self.model_config.class_names) or coco_classes()
        conf_thres = float(self.model_config.postprocess.get("conf_thres", 0.25))
        confidence = [
            {
                "model_id": index + 1,
                "export_id": "{:08d}".format(index + 1),
                "name": name,
                "confidence": conf_thres,
                "is_export": 1,
            }
            for index, name in enumerate(class_names)
        ]
        with open(config_dir / CONFIDENCE_FILENAME, "w", encoding="utf-8", newline="\n") as fw:
            json.dump(confidence, fw, ensure_ascii=False, indent=2)

        dump_yaml(self._runtime_yaml(class_names, model_filename), config_dir / RUNTIME_YAML_FILENAME)

    def _write_packages_dir(self, package_dir: Path) -> None:
        packages = package_dir / "packages"
        packages.mkdir(parents=True, exist_ok=True)
        copied = 0
        if self.packages_dir and Path(self.packages_dir).is_dir():
            for path in sorted(Path(self.packages_dir).iterdir()):
                if path.is_file() and path.suffix in (".whl", ".gz", ".zip"):
                    shutil.copyfile(path, packages / path.name)
                    copied += 1
        if copied == 0:
            # Dockerfile 使用 `COPY packages/`, 目录必须非空
            (packages / "README.txt").write_text(
                "本目录用于放置算法镜像所需的额外依赖轮子(.whl/.tar.gz)。\n"
                "当前交付包未包含额外依赖, 基础镜像内 Python 环境已满足运行需要。\n",
                encoding="utf-8",
                newline="\n",
            )

    def _runtime_yaml(self, class_names: List[str], model_filename: str) -> Dict[str, Any]:
        runtime_conf = dict(self.hardware_config.runtime or {})
        return {
            "name": self.manifest.name,
            "version": self.manifest.version,
            "task": self.model_config.task,
            "framework": self.manifest.framework,
            "algorithm": {
                "name": self.manifest.name,
                "version": self.manifest.version,
                "type": self.model_config.task,
                "app_type": "",
                "component_code": "",
                "template_version": self.manifest.version,
            },
            "model": {"file": model_filename},
            "input": {
                "shape": list(self.model_config.input_shape),
                "dtype": self.model_config.input_dtype,
                "layout": self.model_config.input_layout,
            },
            "preprocess": dict(self.model_config.preprocess),
            "postprocess": dict(self.model_config.postprocess),
            "num_classes": int(self.model_config.num_classes),
            "class_names": list(class_names),
            "runtime": {
                "api_version": runtime_conf.get("api_version", RUNTIME_API_VERSION),
                "port": int(runtime_conf.get("port", 58025)),
                "workers": int(runtime_conf.get("workers", 1)),
                "device": self.hardware_config.device,
                "host_ip": "127.0.0.1",
                "result_dir": "/tmp/nwai_result",
            },
            "logging": {"process_log": "/tmp/nwai_log/process.log"},
        }

    def _build_package_manifest(self, package_dir: Path, artifact: Optional[Any],
                                accuracy: Optional[Any], benchmark: Optional[Any],
                                runtime_packager: RuntimePackager,
                                image_tar: Optional[str],
                                model_filename: str) -> PackageManifest:
        runtime_conf = dict(self.hardware_config.runtime or {})
        validation_spec = self.manifest.validation or {}
        validation: Dict[str, Any] = {
            "enabled": bool(validation_spec.get("enabled", False)),
            "dataset": validation_spec.get("dataset", ""),
        }
        if accuracy is not None and hasattr(accuracy, "to_dict"):
            validation.update(accuracy.to_dict())
        benchmark_payload = benchmark.to_dict() if (benchmark is not None and hasattr(benchmark, "to_dict")) else {}

        return build_manifest(
            name=self.manifest.name,
            version=self.manifest.version,
            model_type=self.manifest.model_type,
            framework=self.manifest.framework,
            task=self.model_config.task,
            hardware=self.hardware_config.name,
            precision=self.hardware_config.precision,
            input_spec={
                "shape": list(self.model_config.input_shape),
                "dtype": self.model_config.input_dtype,
                "layout": self.model_config.input_layout,
            },
            runtime_spec={
                "type": self.manifest.runtime.get("type") or self.runtime,
                "api_version": runtime_conf.get("api_version", RUNTIME_API_VERSION),
                "port": int(runtime_conf.get("port", 58025)),
                "workers": int(runtime_conf.get("workers", 1)),
                "device": self.hardware_config.device,
                "endpoints": ["/predict", "/predict_image", "/health", "/setflag"],
            },
            artifact=artifact,
            model_file=model_filename,
            source=dict({
                "framework": self.manifest.framework,
                "model_type": self.manifest.model_type,
                "file": self.manifest.model_file,
            }, **{k: self.manifest.source.get(k) for k in SOURCE_FINGERPRINT_KEYS if self.manifest.source.get(k)}),
            validation=validation,
            benchmark=benchmark_payload,
            runtime_package={
                "name": runtime_packager.name,
                "file": RUNTIME_TGZ_FILENAME,
                "api_version": RUNTIME_API_VERSION,
                "shared": True,
            },
            image={
                "name": self.manifest.name,
                "tag": self.manifest.version,
                "tar": image_tar or "",
            },
            notes=list(getattr(artifact, "notes", []) or []),
        )

    def _template_context(self, package_manifest: PackageManifest, model_filename: str) -> Dict[str, Any]:
        runtime_conf = dict(self.hardware_config.runtime or {})
        docker_conf = dict(self.hardware_config.docker or {})
        log_conf = dict(docker_conf.get("log") or {})
        return {
            "package_name": package_manifest.package_name,
            "package_version": package_manifest.package_version,
            "model_name": package_manifest.model_name,
            "model_version": package_manifest.model_version,
            "model_file": model_filename,
            "framework": package_manifest.framework,
            "task": package_manifest.task,
            "hardware": package_manifest.hardware,
            "precision": package_manifest.precision,
            "api_version": package_manifest.api_version,
            "degraded": package_manifest.degraded,
            "converter_version": self.converter_version,
            "base_image": self.hardware_config.base_image,
            "gpu_id": 0,
            "workers": int(runtime_conf.get("workers", 1)),
            "device": self.hardware_config.device,
            "web_port": package_manifest.web_port,
            "host_ip": "127.0.0.1",
            "conf_thres": float(runtime_conf.get("conf_thres", 0.25)),
            "iou_thres": float(runtime_conf.get("iou_thres", 0.45)),
            "algorithm_name": package_manifest.model_name,
            "algorithm_version": package_manifest.model_version,
            "image_name": package_manifest.image_name,
            "image_tag": package_manifest.image_tag,
            "image_full": package_manifest.image_full,
            "container_name": package_manifest.container_name,
            "image_tar": package_manifest.image_tar,
            "class_count": int(self.model_config.num_classes),
            # 日志上报地址留空: 现场通过 docker run -e 或 install.conf 注入, 不落默认口令
            "minio_host": log_conf.get("minio_host", ""),
            "minio_port": log_conf.get("minio_port", ""),
            "minio_bucket": log_conf.get("minio_bucket", ""),
            "minio_access_key": log_conf.get("minio_access_key", ""),
            "minio_secret_key": log_conf.get("minio_secret_key", ""),
            "kafka_host": log_conf.get("kafka_host", ""),
            "kafka_port": log_conf.get("kafka_port", ""),
            "kafka_topic": log_conf.get("kafka_topic", ""),
        }

    @staticmethod
    def _make_executable(package_dir: Path) -> None:
        import stat

        for name in ("build.sh", "start.sh"):
            path = package_dir / name
            if not path.is_file():
                continue
            try:
                path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            except Exception:
                # Windows 等不支持 POSIX 权限的文件系统, 忽略即可
                pass

    @staticmethod
    def _zip(package_dir: Path, output_root: Path) -> str:
        archive_base = output_root / package_dir.name
        try:
            return shutil.make_archive(str(archive_base), "zip",
                                       root_dir=str(package_dir.parent),
                                       base_dir=package_dir.name)
        except Exception as err:
            raise PackageError("交付包压缩失败: {}".format(err))
