# -*- coding: utf-8 -*-
"""Exporter 与公共 Runtime 测试(建设目标 §5 / §6 / §15)。

覆盖三件事:

1. ``RuntimePackager``: ``runtime.tgz`` 由 ``runtime/common`` 与 ``runtime/<task>``
   合并平铺而成, 所有模型共用, 且不含 ``__pycache__``;
2. ``packages/`` 目录: 配置 ``docker.packages_dir`` 时复制本地依赖轮子, 并纳入
   ``manifest.json`` 的 files/checksums; 未配置时写入占位 README.txt 保证
   ``COPY packages/`` 不失败;
3. 交付层一致性(§15): readme.txt 描述的镜像加载方式必须与包内实际内容一致。
"""
import json
import shutil
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests import _models
from xpu_converter.backend.kunlun import KunlunBackend, KunlunConfig, XpuGraphBuilder
from xpu_converter.config import BuildManifest, HardwareConfig, ModelConfig
from xpu_converter.exporter.docker_exporter import DockerExporter, RuntimePackager
from xpu_converter.exporter.manifest import MANIFEST_FILENAME, sha256_file
from xpu_converter.ir import onnx as onnx_ir

INPUT_SHAPE = [1, 3, 32, 32]


def build_stub_artifact(workdir: Path):
    """用 stub 适配器生成占位产物, 等价于本机无昆仑 SDK 的场景。"""
    source = Path(workdir) / "model.onnx"
    onnx_ir.save_model(
        _models.build_conv_bn_relu(batch=1, channels=4, height=32, width=32), str(source)
    )
    graph = onnx_ir.load(str(source))
    graph.infer_shapes()
    xpu_dir = Path(workdir) / "xpu"
    xpu_graph = XpuGraphBuilder("kunlun").build(
        graph,
        precision="fp16",
        workdir=str(xpu_dir),
        input_shapes={graph.inputs[0].name: list(INPUT_SHAPE)},
    )
    backend = KunlunBackend(KunlunConfig(sdk_adapter="stub", precision="fp16"))
    artifact = backend.compile(xpu_graph, str(xpu_dir / "model.xpu"))
    return artifact, source


class ExporterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="xpu_exporter_"))
        self.artifact, self.onnx_path = build_stub_artifact(self.tmp)
        self.wheel_dir = self.tmp / "wheels"
        self.wheel_dir.mkdir()
        (self.wheel_dir / "flask-3.0.0-py3-none-any.whl").write_bytes(b"wheel-flask")
        (self.wheel_dir / "kafka_python-2.0.2.tar.gz").write_bytes(b"wheel-kafka")
        (self.wheel_dir / "notes.md").write_text("不属于依赖, 不应被复制", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------ 公共 Runtime
    def test_runtime_package_is_model_agnostic(self):
        packager = RuntimePackager(task="detection", version="v1.0")
        archive_path = self.tmp / "runtime.tgz"
        packager.build(str(archive_path))

        with tarfile.open(str(archive_path), "r:gz") as tar:
            names = {member.name.lstrip("./") for member in tar.getmembers() if member.isfile()}

        # 平铺结构: Runtime 模块直接用顶层 import, 与 service.sh 的 cd 行为一致
        for expected in ("runtime_server.py", "runtime_settings.py", "service.sh",
                         "runtime_backend.py", "send_log_webserver.py"):
            self.assertIn(expected, names)
        self.assertFalse([name for name in names if name.startswith("common/")], names)
        self.assertFalse([name for name in names if "__pycache__" in name], names)

        # common + detection 两个目录的文件全部合并进来
        expected_count = sum(
            1 for source in packager.source_dirs() for path in source.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
        self.assertEqual(len(names), expected_count)

    def test_runtime_shared_by_all_models(self):
        """不同模型导出的 runtime.tgz 内容一致(建设目标 §5: Runtime 与模型解耦)。"""
        first = self.tmp / "shared_a.tgz"
        second = self.tmp / "shared_b.tgz"
        RuntimePackager(task="detection", version="v1.0").build(str(first))
        RuntimePackager(task="detection", version="v1.0").build(str(second))
        # gzip 头内嵌时间戳, 直接比字节没有意义; 比较成员内容指纹
        self.assertEqual(self._fingerprint(first), self._fingerprint(second))

    @staticmethod
    def _fingerprint(archive_path: Path):
        import hashlib

        entries = {}
        with tarfile.open(str(archive_path), "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                payload = tar.extractfile(member).read()
                entries[member.name.lstrip("./")] = hashlib.sha256(payload).hexdigest()
        return entries

    # ------------------------------------------------------------------ packages/
    def test_packages_dir_is_copied_and_checksummed(self):
        package_dir = self._export("pkg_with_wheels", packages_dir=str(self.wheel_dir))
        packages = package_dir / "packages"

        copied = sorted(path.name for path in packages.iterdir())
        self.assertEqual(copied, ["flask-3.0.0-py3-none-any.whl", "kafka_python-2.0.2.tar.gz"])

        manifest = json.loads((package_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        for relative in ("packages/flask-3.0.0-py3-none-any.whl",
                         "packages/kafka_python-2.0.2.tar.gz"):
            self.assertIn(relative, manifest["files"])
            self.assertEqual(manifest["checksums"][relative], sha256_file(package_dir / relative))

    def test_packages_falls_back_to_hardware_config(self):
        """未显式传 packages_dir 时, 从 hardware_config.docker.packages_dir 读取。"""
        package_dir = self._export("pkg_from_config", docker_packages_dir=str(self.wheel_dir))
        self.assertTrue((package_dir / "packages" / "flask-3.0.0-py3-none-any.whl").is_file())

    def test_packages_placeholder_when_empty(self):
        package_dir = self._export("pkg_no_wheels")
        packages = package_dir / "packages"
        self.assertEqual([path.name for path in packages.iterdir()], ["README.txt"])
        manifest = json.loads((package_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        self.assertIn("packages/README.txt", manifest["files"])

    # ------------------------------------------------------------------ 交付层一致性
    def test_readme_matches_package_content(self):
        """§15: readme 中的 docker load / docker build 描述必须与包内实际文件一致。"""
        built = self._export("pkg_build")
        readme = (built / "readme.txt").read_text(encoding="utf-8")
        self.assertIn("docker build -t demo:v1.0 .", readme)
        self.assertNotIn("docker load", readme)

        loaded = self._export("pkg_load", image_tar="demo_v1.0.tar")
        readme = (loaded / "readme.txt").read_text(encoding="utf-8")
        self.assertIn("docker load -i demo_v1.0.tar", readme)
        manifest = json.loads((loaded / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(manifest["image"]["tar"], "demo_v1.0.tar")

    def test_zip_contains_package_root(self):
        package_dir = self._export("pkg_zip", packages_dir=str(self.wheel_dir))
        zip_path = package_dir.parent / (package_dir.name + ".zip")
        self.assertTrue(zip_path.is_file())
        with zipfile.ZipFile(str(zip_path)) as archive:
            names = {name for name in archive.namelist() if not name.endswith("/")}
        prefix = package_dir.name + "/"
        self.assertTrue(all(name.startswith(prefix) for name in names), names)
        self.assertTrue(prefix + "manifest.json" in names)

    # ------------------------------------------------------------------ 内部
    def _export(self, name: str, packages_dir=None, docker_packages_dir=None,
                image_tar=None) -> Path:
        docker_conf = {"template": "kunlun", "include_packages": True}
        if docker_packages_dir:
            docker_conf["packages_dir"] = docker_packages_dir
        hardware_config = HardwareConfig(name="kunlun", precision="fp16", docker=docker_conf)
        model_config = ModelConfig.from_dict({
            "model_type": "yolov10",
            "task": "detection",
            "input_shape": list(INPUT_SHAPE),
            "num_classes": 3,
            "class_names": ["cat", "dog", "person"],
        })
        manifest = BuildManifest.from_dict({
            "name": "demo",
            "version": "v1.0",
            "task": "detection",
            "source": {"framework": "pytorch", "model_type": "yolov10", "file": "best.pt"},
            "input": {"shape": list(INPUT_SHAPE), "dtype": "float32"},
            "target": {"hardware": "kunlun", "precision": "fp16", "batch_size": 1},
            "runtime": {"type": "detection", "port": 58025},
            "package": {"docker": True},
        })

        output_dir = self.tmp / name
        exporter = DockerExporter(
            manifest=manifest,
            model_config=model_config,
            hardware_config=hardware_config,
            runtime="detection",
            packages_dir=packages_dir,
        )
        exporter.export(
            model_path=self.artifact.model_path,
            output_dir=str(output_dir),
            artifact=self.artifact,
            onnx_path=str(self.onnx_path),
            image_tar=image_tar,
        )
        return output_dir / manifest.package_name


if __name__ == "__main__":
    unittest.main()
