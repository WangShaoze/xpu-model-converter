# -*- coding: utf-8 -*-
"""模型级 Conformance 框架测试(ChatGPT 修改意见 P0-8)。

用合成 ONNX 图充当"模型", 验证:

1. 逐后端链式比对能为每个环节产出张量指纹(shape/dtype/sha256)与数值
   max_abs_error/cosine, 且同一模型的两个会话之间通过;
2. 可选的(非 required)后端在无法构造时被标记 ``skipped`` 而不阻塞整体;
3. ``required`` 环节不可用/数值不一致时整体门禁失败(而非静默通过)。
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests import _models, requires_onnx
from xpu_converter.backend.kunlun.runtime import OnnxRuntimeSession
from xpu_converter.conformance import ConformanceRunner, ConformanceStage
from xpu_converter.errors import ValidationError
from xpu_converter.ir import onnx as onnx_ir

INPUT_SHAPE = [1, 3, 16, 16]


def build_session(model_path: Path, provider=None):
    return OnnxRuntimeSession(str(model_path), device="cpu", providers=provider)


def feeds(batch: int = 1):
    import numpy as np

    return {"images": np.random.RandomState(7).rand(*[batch, 3, 16, 16]).astype(np.float32)}


@requires_onnx
class ConformanceRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="conformance_"))
        cls.model = cls.tmp / "model.onnx"
        onnx_ir.save_model(
            _models.build_conv_bn_relu(batch=1, channels=4, height=16, width=16), str(cls.model)
        )
        # 几何不同的模型(输出通道不同), 用于制造数值/形状不一致
        cls.model_other = cls.tmp / "model_other.onnx"
        onnx_ir.save_model(
            _models.build_conv_bn_relu(batch=1, channels=5, height=16, width=16), str(cls.model_other)
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _report(self, stages, fail_on_mismatch=True, samples=None):
        runner = ConformanceRunner(fail_on_mismatch=fail_on_mismatch)
        reference = build_session(self.model)
        try:
            return runner.run(
                "yolov8",
                reference,
                stages,
                samples or [feeds()],
                task="detection",
                model_path=str(self.model),
                input_shape=INPUT_SHAPE,
            )
        finally:
            reference.close()

    def test_chain_emits_fingerprint_and_numeric_compare(self):
        reference = build_session(self.model)
        optimized = build_session(self.model)   # 同一模型代表 "optimized/paddle" 环节
        try:
            report = ConformanceRunner().run(
                "yolov8", reference,
                [ConformanceStage(name="optimized-onnx", session=optimized)],
                [feeds()], task="detection", input_shape=INPUT_SHAPE,
            )
        finally:
            reference.close()
            optimized.close()

        self.assertTrue(report.passed)
        self.assertEqual(len(report.stages), 1)
        stage = report.stages[0]
        # 指纹: shape / dtype / sha256 齐备
        self.assertTrue(stage.fingerprints, "应产出环节输出指纹")
        fp = stage.fingerprints[0]
        for key in ("shape", "dtype", "sha256"):
            self.assertIn(key, fp, "指纹缺少 {}".format(key))
        self.assertEqual(fp["dtype"], "float32")
        self.assertTrue(stage.compare.get("ok"))
        item = stage.compare["items"][0]
        for key in ("max_abs_error", "cosine_similarity"):
            self.assertIn(key, item)
        # 报告 JSON 可序列化
        self.assertIsInstance(report.to_dict()["stages"], list)

    def test_optional_stage_skipped_when_backend_unavailable(self):
        def boom():
            raise RuntimeError("无该后端(SDK 缺失)")

        report = self._report([ConformanceStage(name="xpu", factory=boom)])
        self.assertTrue(report.passed, "可选后端缺失不应判整体失败")
        stage = report.stages[0]
        self.assertEqual(stage.status, "skipped")
        self.assertIn("不可用", stage.reason)

    def test_required_stage_failure_raises(self):
        def boom():
            raise RuntimeError("无该后端")

        with self.assertRaises(ValidationError):
            self._report([ConformanceStage(name="xpu", factory=boom, required=True)])

    def test_numeric_mismatch_fails_stage(self):
        # 参考与环节输出几何不同(输出通道数不一致): 同一输入下数值/形状必然不一致
        import numpy as np

        reference = build_session(self.model)
        other = build_session(self.model_other)
        runner = ConformanceRunner(fail_on_mismatch=False)
        try:
            report = runner.run(
                "yolov8", reference,
                [ConformanceStage(name="paddle", session=other)],
                [{"images": np.zeros([1, 3, 16, 16], np.float32)}],
                task="detection", input_shape=INPUT_SHAPE,
            )
        finally:
            reference.close()
            other.close()
        self.assertEqual(report.stages[0].status, "ok")
        self.assertFalse(report.stages[0].passed, "输出不一致应使环节不通过")
        self.assertFalse(report.passed)


if __name__ == "__main__":
    unittest.main()