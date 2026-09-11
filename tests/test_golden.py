# -*- coding: utf-8 -*-
"""Golden Model 回归(ChatGPT 修改意见 §47/§48)。

放置 ``tests/golden/<case>/model.pt`` 后本用例自动生效: 每次改动都会重新走一遍
"框架模型 → ONNX → Optimize → Capability Check", 并(若有参考输出)逐张量比对,
防止 SiLU / Resize / Conv-BN / NMS 等改写把模型改坏。

没有 golden 数据时整体 skip, 不作为 CI 失败项。
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import requires_onnx
from tests.golden import GoldenCase, discover_cases
from xpu_converter.backend.kunlun.compiler import KunlunBackend
from xpu_converter.backend.kunlun.config import KunlunConfig
from xpu_converter.backend.kunlun.runtime import OnnxRuntimeSession
from xpu_converter.ir import onnx as onnx_ir
from xpu_converter.validator.tensor_compare import compare_tensors

GOLDEN_CASES = discover_cases()


@requires_onnx
@unittest.skipUnless(GOLDEN_CASES, "tests/golden/ 下暂无 golden 用例(需自行放置 model.pt)")
class GoldenModelTest(unittest.TestCase):
    """对每个 golden 用例做一次转换 + 能力门禁(+ 可选数值比对)。"""

    def test_golden_cases(self):
        for case in GOLDEN_CASES:
            with self.subTest(case=case.name):
                self._run_case(case)

    # ------------------------------------------------------------------ 内部
    def _run_case(self, case: GoldenCase) -> None:
        onnx_path = self._to_onnx(case)
        graph = onnx_ir.load(str(onnx_path))
        graph.infer_shapes()

        backend = KunlunBackend(KunlunConfig(precision="fp16"))
        report = backend.analyze(graph)
        self.assertTrue(
            report.ok,
            "golden 用例 {} 存在不支持算子: {}".format(case.name, report.unsupported_ops),
        )

        expected_shape = case.expected.get("input_shape")
        if expected_shape:
            self.assertEqual(list(graph.inputs[0].shape), list(expected_shape))

        if case.reference_outputs:
            self._compare_reference(case, onnx_path, graph)

    def _to_onnx(self, case: GoldenCase) -> Path:
        if case.framework == "onnx":
            return case.model_path
        from xpu_converter.registry import model_registry

        adapter = model_registry.create_adapter(case.model_type)
        workdir = Path(tempfile.mkdtemp(prefix="golden_"))
        target = workdir / "{}.onnx".format(case.name)
        exported = adapter.export_onnx(
            str(case.model_path), str(target),
            input_shape=case.expected.get("input_shape") or adapter.input_shape(),
            opset=adapter.opset(), dynamic=False,
        )
        return Path(exported)

    def _compare_reference(self, case: GoldenCase, onnx_path: Path, graph) -> None:
        reference = np.load(str(case.reference_outputs))
        session = OnnxRuntimeSession(str(onnx_path), device="cpu")
        feeds = {}
        for index, name in enumerate(session.input_names):
            shape = list(graph.inputs[index].shape) if index < len(graph.inputs) else [1]
            shape = [1 if (dim is None or int(dim) <= 0) else int(dim) for dim in shape]
            feeds[name] = np.zeros(shape, dtype=np.float32)
        outputs = session.run(feeds)
        actual = next(iter(outputs.values()))
        result = compare_tensors(
            reference.reshape(actual.shape) if reference.size == actual.size else reference,
            actual,
            name=case.name,
            atol=float(case.expected.get("atol", 1e-3)),
            rtol=float(case.expected.get("rtol", 1e-3)),
        )
        self.assertTrue(result.passed, "{} 数值不一致: {}".format(case.name, result.summary()))
