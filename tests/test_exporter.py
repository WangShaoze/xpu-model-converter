# -*- coding: utf-8 -*-
"""Exporter 与公共 Runtime 测试(建设目标 §5 / §6 / §15)。

覆盖四件事:

1. ``RuntimePackager``: 公共 Runtime 由 ``runtime/common`` 与 ``runtime/<task>``
   平铺进**算法目录**, 所有模型共用, 且不含 ``__pycache__``;
2. ``packages/`` 目录: 配置 ``docker.packages_dir`` 时复制本地依赖轮子; 未配置时
   写入占位 README.txt 保证 ``COPY packages/`` 不失败;
3. 交付包结构与客户标准包一致: 顶层 Dockerfile/build.sh/readme.txt + 算法同名目录;
4. assets_dir: 说明书(.docx)与测试图(testimage.*)复制到交付包顶层, 缺失时告警。
"""
import hashlib
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests import _models
from tests import requires_onnx
from xpu_converter.backend.kunlun import KunlunBackend, KunlunConfig, XpuGraphBuilder
from xpu_converter.config import BuildManifest, HardwareConfig, ModelConfig
from xpu_converter.exporter.docker_exporter import DockerExporter, RuntimePackager
from xpu_converter.exporter.manifest import ARTIFACT_MANIFEST_FILENAME, sha256_file
from xpu_converter.ir import onnx as onnx_ir

INPUT_SHAPE = [1, 3, 32, 32]
ALGORITHM_DIR = "demo"


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
    backend = KunlunBackend(KunlunConfig(sdk_adapter="stub", precision="fp16", allow_degraded=True))
    artifact = backend.compile(xpu_graph, str(xpu_dir / "model.xpu"))
    return artifact, source


@requires_onnx
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
    def test_runtime_staged_flat_into_algorithm_dir(self):
        packager = RuntimePackager(task="detection", version="v1.0")
        staged = self.tmp / "staged"
        packager.stage(staged)
        names = {path.relative_to(staged).as_posix() for path in staged.rglob("*") if path.is_file()}

        # 平铺结构: Runtime 模块直接平铺到算法目录, 支持顶层 import
        for expected in ("nwai_webserver.py", "nwai_settings.py", "nwai_gunicorn.py",
                         "nwai_backend.py", "send_log_webserver.py", "send_log_settings.json"):
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
        """不同模型导出的 Runtime 内容一致(建设目标 §5: Runtime 与模型解耦)。"""
        first = self.tmp / "shared_a"
        second = self.tmp / "shared_b"
        RuntimePackager(task="detection", version="v1.0").stage(first)
        RuntimePackager(task="detection", version="v1.0").stage(second)
        self.assertEqual(self._fingerprint(first), self._fingerprint(second))

    def test_runtime_falls_back_to_generic_for_non_detection_task(self):
        """多任务模型(segment 等)无专属 runtime/<task> 目录时, 复用通用检测 Runtime。

        对应 ChatGPT 修改意见 P0-1: Runtime 是任务无关实现, ``task`` 不应绑定目录;
        否则 ``package`` 命令对 ``task=segment`` 等模型会因目录缺失而抛 PackageError。
        """
        generic = self.tmp / "generic"
        RuntimePackager(task="detection", version="v1.0").stage(generic)
        seg = self.tmp / "seg"
        RuntimePackager(task="segment", version="v1.0").stage(seg)
        self.assertEqual(self._fingerprint(generic), self._fingerprint(seg))
        # 解析结果应回退到通用实现(不因 runtime/segment 缺失而抛错)
        packager = RuntimePackager(task="pose", version="v1.0")
        self.assertTrue(all(path.is_dir() for path in packager.runtime_dirs()))

    @staticmethod
    def _fingerprint(directory: Path):
        entries = {}
        for path in sorted(Path(directory).rglob("*")):
            if path.is_file():
                entries[path.relative_to(directory).as_posix()] = hashlib.sha256(
                    path.read_bytes()).hexdigest()
        return entries

    # ------------------------------------------------------------------ packages/
    def test_packages_dir_is_copied(self):
        package_dir = self._export("pkg_with_wheels", packages_dir=str(self.wheel_dir))
        copied = sorted(path.name for path in (package_dir / "packages").iterdir())
        self.assertEqual(copied, ["flask-3.0.0-py3-none-any.whl", "kafka_python-2.0.2.tar.gz"])

    def test_packages_falls_back_to_hardware_config(self):
        """未显式传 packages_dir 时, 从 hardware_config.docker.packages_dir 读取。"""
        package_dir = self._export("pkg_from_config", docker_packages_dir=str(self.wheel_dir))
        self.assertTrue((package_dir / "packages" / "flask-3.0.0-py3-none-any.whl").is_file())

    def test_packages_placeholder_when_empty(self):
        package_dir = self._export("pkg_no_wheels")
        self.assertEqual([path.name for path in (package_dir / "packages").iterdir()], ["README.txt"])

    # ------------------------------------------------------------------ 交付包结构
    def test_package_layout_matches_customer_standard(self):
        package_dir = self._export("pkg_layout")
        top_level = sorted(path.name for path in package_dir.iterdir())
        self.assertEqual(top_level, ["Dockerfile", "build.sh", "demo", "packages", "readme.txt"])

        algorithm_dir = package_dir / ALGORITHM_DIR
        for expected in ("start.sh", "runtime.yaml", "confidence.json", "metadata.json",
                         ARTIFACT_MANIFEST_FILENAME, "model.yaml",
                         Path(self.artifact.model_path).name, "nwai_webserver.py"):
            self.assertTrue((algorithm_dir / expected).is_file(), expected)

        # 旧结构(manifest.json / runtime.tgz / model/ / config/)已移除
        for legacy in ("manifest.json", "runtime.tgz", "model", "config", "install.conf", "start.sh"):
            self.assertFalse((package_dir / legacy).exists(), legacy)

    def test_readme_describes_build_and_load(self):
        built = self._export("pkg_build")
        readme = (built / "readme.txt").read_text(encoding="utf-8")
        self.assertIn("docker load -i 基础镜像.tar", readme)
        self.assertIn("docker build -t demo:v1.0 .", readme)

    def test_zip_contains_package_root(self):
        package_dir = self._export("pkg_zip", packages_dir=str(self.wheel_dir))
        zip_path = package_dir.parent / (package_dir.name + ".zip")
        self.assertTrue(zip_path.is_file())
        with zipfile.ZipFile(str(zip_path)) as archive:
            names = {name for name in archive.namelist() if not name.endswith("/")}
        prefix = package_dir.name + "/"
        self.assertTrue(all(name.startswith(prefix) for name in names), names)
        self.assertIn(prefix + "Dockerfile", names)
        self.assertIn(prefix + ALGORITHM_DIR + "/start.sh", names)

    # ------------------------------------------------------------------ assets
    def test_assets_dir_copies_docx_and_testimage(self):
        assets = self.tmp / "assets"
        assets.mkdir()
        (assets / "接口说明书_v1.0.docx").write_bytes(b"docx")
        (assets / "testimage.jpg").write_bytes(b"jpg")
        (assets / "ignored.txt").write_text("忽略", encoding="utf-8")

        package_dir = self._export("pkg_assets", assets_dir=str(assets))
        copied = sorted(path.name for path in package_dir.iterdir() if path.is_file())
        self.assertIn("接口说明书_v1.0.docx", copied)
        self.assertIn("testimage.jpg", copied)
        self.assertNotIn("ignored.txt", copied)

    def test_missing_assets_dir_warns(self):
        exporter = self._exporter("pkg_missing_assets", assets_dir=str(self.tmp / "not-exist"))
        exporter.export(model_path=self.artifact.model_path,
                        output_dir=str(self.tmp / "pkg_missing_assets"),
                        artifact=self.artifact)
        self.assertTrue(exporter.warnings)
        self.assertIn("assets_dir", exporter.warnings[0])

    # ------------------------------------------------------------------ provenance
    def test_artifact_manifest_records_provenance(self):
        """artifact.json 记录产物来源与编译信息; model.yaml 只描述"如何产生"(§28/§29/§30)。"""
        import yaml

        package_dir = self._export("pkg_provenance")
        algorithm_dir = package_dir / ALGORITHM_DIR
        payload = self._artifact_payload(algorithm_dir)
        for key in ("converter", "source", "export", "optimization", "backend", "artifact"):
            self.assertIn(key, payload)
        self.assertEqual(payload["artifact"]["file"], Path(self.artifact.model_path).name)
        self.assertEqual(payload["artifact"]["sha256"], sha256_file(self.artifact.model_path))
        self.assertTrue(payload["converter"]["version"])
        self.assertEqual(payload["export"]["input_shape"], list(INPUT_SHAPE))

        model_yaml = yaml.safe_load((algorithm_dir / "model.yaml").read_text(encoding="utf-8"))
        self.assertNotIn("package", model_yaml)
        self.assertNotIn("runtime", model_yaml)
        self.assertIn("source", model_yaml)

    def _artifact_payload(self, algorithm_dir: Path):
        import json

        return json.loads((algorithm_dir / ARTIFACT_MANIFEST_FILENAME).read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ 内部
    def _exporter(self, name: str, packages_dir=None, docker_packages_dir=None,
                  assets_dir=None) -> DockerExporter:
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
        return DockerExporter(
            manifest=manifest,
            model_config=model_config,
            hardware_config=hardware_config,
            runtime="detection",
            packages_dir=packages_dir,
            assets_dir=assets_dir,
            # 本用例用 stub 占位产物验证打包机制, 需显式允许降级产物入包
            allow_degraded=True,
        )

    def _export(self, name: str, packages_dir=None, docker_packages_dir=None,
                assets_dir=None) -> Path:
        exporter = self._exporter(name, packages_dir, docker_packages_dir, assets_dir)
        output_dir = self.tmp / name
        exporter.export(
            model_path=self.artifact.model_path,
            output_dir=str(output_dir),
            artifact=self.artifact,
            onnx_path=str(self.onnx_path),
        )
        return output_dir / exporter.manifest.package_name


if __name__ == "__main__":
    unittest.main()
