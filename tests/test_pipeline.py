# -*- coding: utf-8 -*-
"""端到端流水线测试(建设目标 §2 的 10 步输出与交付包结构)。

本机未安装 torch / ultralytics, 因此这里注册一个"合成 ONNX 适配器"替代真实
PyTorch 前端: 它沿用 YOLOv10 的 model_type, 但直接把合成 ONNX 图当作导出结果,
从而在无 GPU / 无真实模型的情况下验证:

1. ``[01] … [10]`` 十个步骤全部 OK;
2. 产物目录包含 onnx / xpu / package 三部分;
3. 交付 ZIP 内 manifest.json 与文件清单一致(交付层一致性, 建设目标 §15);
4. 仅在**显式**打开降级开关(``allow_degraded`` / ``allow_degraded_package``)时,
   SDK 缺失才允许产出降级占位件; 默认严格模式下必须直接失败。
"""
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests import _models
from tests import requires_onnx
from xpu_converter.frontend.base import BaseModelAdapter, FrontendModel
from xpu_converter.ir import onnx as onnx_ir
from xpu_converter.registry import model_registry
from xpu_converter.version import CONVERTER_VERSION

INPUT_SHAPE = [1, 3, 32, 32]


class SyntheticYoloAdapter(BaseModelAdapter):
    """合成 YOLO 适配器: 绕过 torch, 直接产出结构等价的 ONNX。"""

    framework = "pytorch"
    model_type = "synthetic_yolo"
    task = "detection"
    end2end_default = False

    def default_input_shape(self):
        return list(INPUT_SHAPE)

    def default_num_classes(self):
        return 3

    def default_class_names(self):
        return ["cat", "dog", "person"]

    def load_model(self, model_path):
        # 真实适配器在此调用 torch.load; 合成场景只需返回一个占位对象
        return {"path": str(model_path)}

    def export_onnx(self, model_path, output_path, input_shape=None, opset=None, dynamic=None):
        shape = list(input_shape or INPUT_SHAPE)
        model = _models.build_conv_bn_relu(
            batch=int(shape[0]), channels=4, height=int(shape[2]), width=int(shape[3])
        )
        onnx_ir.save_model(model, str(output_path))
        return str(output_path)

    def inspect(self, model_path):
        return FrontendModel(
            framework=self.framework,
            model_type=self.model_type,
            task=self.task,
            source_path=str(model_path),
            input_shape=list(INPUT_SHAPE),
            num_classes=self.default_num_classes(),
            class_names=self.default_class_names(),
            end2end=False,
        )


@requires_onnx
class PipelineEndToEndTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="xpu_pipeline_"))
        # 冻结适配器注册表, 避免 _ensure_loaded() 用内置 YOLO 适配器覆盖合成适配器
        self._loaded_backup = model_registry._LOADED
        self._adapters_backup = dict(model_registry._ADAPTERS)
        model_registry._LOADED = True
        model_registry.register_adapter(SyntheticYoloAdapter)

    def tearDown(self):
        model_registry._ADAPTERS.clear()
        model_registry._ADAPTERS.update(self._adapters_backup)
        model_registry._LOADED = self._loaded_backup
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------ 用例
    def test_end_to_end_ten_steps(self):
        from xpu_converter.pipeline import STEP_TITLES, ConversionPipeline

        model_file = self.tmp / "best.pt"
        model_file.write_bytes(b"synthetic")

        pipeline = ConversionPipeline(
            model_path=str(model_file),
            output_dir=str(self.tmp / "output"),
            model_type=SyntheticYoloAdapter.model_type,
            input_shape=list(INPUT_SHAPE),
            precision="fp16",
            benchmark_iterations=3,
            # 本机没有昆仑 SDK, 显式打开降级开关才能跑通 stub 联调链路
            sdk_adapter="stub",
            allow_degraded=True,
            allow_degraded_package=True,
        )

        captured = io.StringIO()
        with redirect_stdout(captured):
            result = pipeline.run()
        output = captured.getvalue()

        # 1) 十个步骤全部 OK
        self.assertEqual(len(result.steps), len(STEP_TITLES))
        self.assertTrue(all(step["ok"] for step in result.steps), output)
        for index, title in enumerate(STEP_TITLES, start=1):
            self.assertIn("[{:02d}] {} OK".format(index, title), output)

        # 2) 关键中间产物
        self.assertTrue(Path(result.onnx_path).is_file())
        self.assertTrue(Path(result.optimized_onnx_path).is_file())
        self.assertTrue(Path(result.artifact.model_path).is_file())
        self.assertTrue(Path(result.package_zip).is_file())
        # 2b) Final Operator Check 必须在编译前产出报告且通过
        self.assertIsNotNone(result.capability)
        self.assertTrue(result.capability.ok, result.capability.reasons)

        # 3) 交付包结构与 manifest 一致性
        package_dir = Path(result.package_dir)
        for relative in (
            "Dockerfile", "build.sh", "install.conf", "readme.txt", "start.sh",
            "manifest.json", "runtime.tgz",
            "model/model.onnx", "model/model.yaml", "model/metadata.json",
            "config/confidence.json", "config/runtime.yaml",
        ):
            self.assertTrue((package_dir / relative).is_file(), "缺少交付文件: " + relative)

        manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["package_name"], "synthetic_yolo_dockerimg_v1.0")
        self.assertEqual(manifest["conversion"]["converter_version"], CONVERTER_VERSION)
        listed = set(manifest["files"]) | {"manifest.json"}
        actual = {
            p.relative_to(package_dir).as_posix()
            for p in package_dir.rglob("*")
            if p.is_file() and not p.name.endswith(".pyc")
        }
        self.assertEqual(listed, actual)
        for relative, digest in manifest["checksums"].items():
            from xpu_converter.exporter.manifest import sha256_file

            self.assertEqual(digest, sha256_file(package_dir / relative))

        # 4) SDK 缺失 -> 降级标记贯穿交付层
        self.assertTrue(result.degraded)
        self.assertEqual(manifest["conversion"]["sdk_adapter"], "stub")
        dockerfile = (package_dir / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("MODEL_ARTIFACT_DEGRADED", dockerfile)

    def test_zip_matches_package_dir(self):
        from xpu_converter.pipeline import ConversionPipeline

        model_file = self.tmp / "best.pt"
        model_file.write_bytes(b"synthetic")
        pipeline = ConversionPipeline(
            model_path=str(model_file),
            output_dir=str(self.tmp / "output2"),
            model_type=SyntheticYoloAdapter.model_type,
            input_shape=list(INPUT_SHAPE),
            benchmark_iterations=2,
            sdk_adapter="stub",
            allow_degraded=True,
            allow_degraded_package=True,
        )
        with redirect_stdout(io.StringIO()):
            result = pipeline.run()

        with zipfile.ZipFile(result.package_zip) as archive:
            names = {name for name in archive.namelist() if not name.endswith("/")}
        prefix = Path(result.package_dir).name + "/"
        expected = {
            p.relative_to(Path(result.package_dir)).as_posix()
            for p in Path(result.package_dir).rglob("*")
            if p.is_file()
        }
        self.assertEqual({name[len(prefix):] for name in names}, expected)

    def test_strict_mode_rejects_degraded_artifact(self):
        """默认严格模式: 无昆仑 SDK 时直接失败, 不允许静默产出占位件。

        本机可能装有 paddle + x2paddle(auto 会解析到真实 Paddle 后端), 因此这里
        显式模拟"无可用 SDK"的环境, 验证严格模式下的失败行为。
        """
        from unittest import mock

        from xpu_converter.errors import XpuConverterError
        from xpu_converter.pipeline import ConversionPipeline

        model_file = self.tmp / "best.pt"
        model_file.write_bytes(b"synthetic")
        pipeline = ConversionPipeline(
            model_path=str(model_file),
            output_dir=str(self.tmp / "output3"),
            model_type=SyntheticYoloAdapter.model_type,
            input_shape=list(INPUT_SHAPE),
            benchmark_iterations=1,
        )
        unavailable = mock.patch(
            "xpu_converter.backend.kunlun.compiler.PaddleXpuSdkAdapter.available",
            return_value=False,
        )
        with unavailable, redirect_stdout(io.StringIO()):
            with self.assertRaises(XpuConverterError) as ctx:
                pipeline.run()
        self.assertIn("SDK", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
