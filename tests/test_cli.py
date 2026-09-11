# -*- coding: utf-8 -*-
"""CLI 子命令测试(建设目标 §13)。

只覆盖不依赖 torch / ultralytics 的路径: ``analyze`` / ``compile`` /
``validate`` / ``package``, 以及 ``convert`` 的参数校验与失败退出码。
"""
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests import _models
from tests import requires_onnx
from tests.test_pipeline import SyntheticYoloAdapter
from xpu_converter.cli.main import main
from xpu_converter.ir import onnx as onnx_ir
from xpu_converter.registry import model_registry


def run_cli(*argv):
    """执行 CLI 并返回 (退出码, 标准输出)。"""
    captured = io.StringIO()
    with redirect_stdout(captured):
        code = main(list(argv))
    return code, captured.getvalue()


@requires_onnx
class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="xpu_cli_"))
        self.onnx_path = self.tmp / "yolov10_synthetic.onnx"
        onnx_ir.save_model(_models.build_conv_bn_relu(batch=1, channels=4, height=32, width=32),
                           str(self.onnx_path))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------ 用例
    def test_help_lists_all_commands(self):
        code, output = run_cli("--help")
        self.assertEqual(code, 0)
        for command in ("inspect", "convert", "export-onnx", "analyze",
                        "compile", "validate", "package", "build"):
            self.assertIn(command, output)

    def test_analyze_reports_operators(self):
        code, output = run_cli("analyze", str(self.onnx_path), "--device", "kunlun")
        self.assertEqual(code, 0)
        self.assertIn("算子分析", output)
        self.assertIn("supported", output)
        self.assertIn("0 unsupported", output)

    def test_compile_then_package_and_validate(self):
        xpu_dir = self.tmp / "xpu"
        code, output = run_cli(
            "compile", "--model", str(self.onnx_path), "--device", "kunlun",
            "--precision", "fp16", "--input-shape", "1,3,32,32", "--output", str(xpu_dir),
        )
        self.assertEqual(code, 0, output)
        self.assertTrue((xpu_dir / "metadata.json").is_file())
        metadata = json.loads((xpu_dir / "metadata.json").read_text(encoding="utf-8"))
        # 适配器由环境决定: 装有 paddle + x2paddle 时产出真实 Paddle 静态图, 否则退占位件
        self.assertIn(metadata["sdk_adapter"], ("paddle", "stub"))
        artifact = Path(metadata["model_path"])
        self.assertTrue(artifact.is_file(), artifact)
        self.assertEqual(metadata["degraded"], metadata["artifact_format"] == "stub")

        # package 由产物 + 旁路 metadata.json 还原 artifact, 并保持降级标记
        package_root = self.tmp / "package"
        package_args = [
            "package", "--model", str(artifact), "--model-type", "yolov10",
            "--input-shape", "1,3,32,32", "--output", str(package_root),
        ]
        if metadata["degraded"]:
            package_args.append("--dev-package")
        code, output = run_cli(*package_args)
        self.assertEqual(code, 0, output)
        zip_path = package_root / "yolov10_dockerimg_v1.0.zip"
        self.assertTrue(zip_path.is_file())
        manifest = json.loads(
            (package_root / "yolov10_dockerimg_v1.0" / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["degraded"], metadata["degraded"])
        self.assertEqual(manifest["precision"], "fp16")
        # 多文件产物(Paddle .pdmodel + .pdiparams)必须整组进入交付包
        for item in metadata.get("files") or []:
            self.assertIn("model/" + Path(item).name, manifest["files"])

        # validate: 基准 ONNX vs 产物, 数值应当一致
        code, output = run_cli(
            "validate", "--source", str(self.onnx_path), "--target", str(artifact),
            "--input-shape", "1,3,32,32", "--max-samples", "2",
        )
        self.assertEqual(code, 0, output)
        self.assertIn("通过", output)

    def test_convert_reports_missing_model(self):
        code, output = run_cli(
            "convert", "--model", str(self.tmp / "not_exist.pt"), "--model-type", "yolov10",
            "--output", str(self.tmp / "missing_output"),
        )
        self.assertNotEqual(code, 0)

    def test_convert_from_manifest(self):
        """建设目标 §14: 一份 model.yaml 驱动整次转换。"""
        self._freeze_registry()
        manifest_path = self.tmp / "yolov10n.yaml"
        manifest_path.write_text(
            "name: smoke_det\n"
            "version: v2.0\n"
            "source:\n"
            "  framework: pytorch\n"
            "  model_type: {model_type}\n"
            "  file: best.pt\n"
            "input:\n"
            "  shape: [1, 3, 32, 32]\n"
            "  dtype: float32\n"
            "target:\n"
            "  hardware: kunlun\n"
            "  precision: fp32\n"
            "  batch_size: 1\n"
            "runtime:\n"
            "  type: detection\n"
            "  port: 58025\n"
            "validation:\n"
            "  enabled: true\n"
            "package:\n"
            "  docker: true\n"
            "  name: smoke_det_dockerimg_v2.0\n".format(model_type=SyntheticYoloAdapter.model_type),
            encoding="utf-8",
        )
        (self.tmp / "best.pt").write_bytes(b"synthetic")

        output_dir = self.tmp / "manifest_output"
        code, output = run_cli("convert", "--manifest", str(manifest_path), "--output", str(output_dir))
        self.assertEqual(code, 0, output)
        self.assertIn("已加载 manifest", output)

        # manifest 决定包名/版本/精度/输入形状; source.file 相对 manifest 目录解析
        zip_path = output_dir / "smoke_det_dockerimg_v2.0.zip"
        self.assertTrue(zip_path.is_file(), output)
        package_dir = output_dir / "smoke_det_dockerimg_v2.0"
        manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["model_name"], "smoke_det")
        self.assertEqual(manifest["precision"], "fp32")
        self.assertEqual(manifest["input"]["shape"], [1, 3, 32, 32])

    def test_convert_requires_model(self):
        code, output = run_cli("convert")
        self.assertNotEqual(code, 0)

    # ------------------------------------------------------------------ 内部
    def _freeze_registry(self):
        """冻结适配器注册表, 用合成适配器替代需要 torch 的 YOLO 适配器。"""
        self._loaded_backup = model_registry._LOADED
        self._adapters_backup = dict(model_registry._ADAPTERS)
        model_registry._LOADED = True
        model_registry.register_adapter(SyntheticYoloAdapter)
        self.addCleanup(self._restore_registry)

    def _restore_registry(self):
        model_registry._ADAPTERS.clear()
        model_registry._ADAPTERS.update(self._adapters_backup)
        model_registry._LOADED = self._loaded_backup


if __name__ == "__main__":
    unittest.main()
