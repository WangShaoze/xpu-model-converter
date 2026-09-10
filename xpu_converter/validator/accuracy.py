# -*- coding: utf-8 -*-
"""精度校验。

V1 约定做 **PyTorch / ONNX / XPU 三路校验**(建设目标 §17):

    reference ─┬─► onnx  ──► compare
               ├─► xpu   ──► compare
               └─► (可选) torch

三者共用同一批输入, 逐张量比对输出; 任意一路不通过都会让整次转换失败, 避免
"模型能编译但结果不对"的静默交付。
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from xpu_converter.backend.base import BaseRuntimeSession
from xpu_converter.errors import ValidationError
from xpu_converter.validator.tensor_compare import (
    DEFAULT_ATOL,
    DEFAULT_COSINE,
    DEFAULT_RTOL,
    TensorCompareResult,
    compare_outputs,
    summarize,
)


@dataclass
class AccuracyReport:
    """精度校验报告。"""

    passed: bool = True
    pairs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    items: List[TensorCompareResult] = field(default_factory=list)
    sample_count: int = 0
    synthetic_inputs: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "sample_count": self.sample_count,
            "synthetic_inputs": self.synthetic_inputs,
            "pairs": self.pairs,
            "overall": summarize(self.items) if self.items else {},
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        if not self.pairs:
            return "跳过(无校验样本)"
        parts = []
        for name, detail in self.pairs.items():
            parts.append("{}:{}".format(name, "OK" if detail.get("ok") else "FAIL"))
        return "{} samples, {}".format(self.sample_count, " ".join(parts))

    def save(self, path: str) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as fw:
            json.dump(self.to_dict(), fw, ensure_ascii=False, indent=2)
        return str(target)


class AccuracyValidator:
    """会话间精度比对器。"""

    def __init__(
        self,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        cosine_thres: float = DEFAULT_COSINE,
        fail_on_mismatch: bool = True,
    ) -> None:
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.cosine_thres = float(cosine_thres)
        self.fail_on_mismatch = fail_on_mismatch

    # ------------------------------------------------------------------ 校验
    def compare_sessions(
        self,
        reference: BaseRuntimeSession,
        target: BaseRuntimeSession,
        samples: Sequence[Dict[str, Any]],
        pair_name: str = "reference-vs-target",
        input_names: Optional[Sequence[str]] = None,
    ) -> Tuple[Dict[str, Any], List[TensorCompareResult]]:
        """把同一批样本分别跑两个会话并逐张量比对。"""
        if not samples:
            return {"ok": True, "skipped": True, "reason": "无校验样本"}, []

        ref_items: List[TensorCompareResult] = []
        for sample in samples:
            feeds = self._adapt_inputs(sample, input_names or reference.input_names)
            reference_outputs = reference.run(feeds)
            target_outputs = target.run(feeds)
            ref_items.extend(
                compare_outputs(
                    reference_outputs,
                    target_outputs,
                    names=reference.output_names,
                    atol=self.atol,
                    rtol=self.rtol,
                    cosine_thres=self.cosine_thres,
                )
            )
        detail = summarize(ref_items)
        detail["pair"] = pair_name
        return detail, ref_items

    def validate(
        self,
        reference: BaseRuntimeSession,
        target: BaseRuntimeSession,
        samples: Sequence[Dict[str, Any]],
        onnx_session: Optional[BaseRuntimeSession] = None,
        synthetic: bool = False,
        notes: Optional[List[str]] = None,
    ) -> AccuracyReport:
        """执行一至三路精度校验。

        ``reference`` 为基准(通常来自 ONNX), ``target`` 为待验证后端(通常为 XPU);
        ``onnx_session`` 不为空时额外做 reference-vs-onnx 的自检。
        """
        report = AccuracyReport(sample_count=len(samples), synthetic_inputs=synthetic)
        report.notes.extend(notes or [])

        if onnx_session is not None:
            detail, items = self.compare_sessions(reference, onnx_session, samples, "reference-vs-onnx")
            report.pairs["onnx"] = detail
            if not detail.get("skipped"):
                report.items.extend(items)

        detail, items = self.compare_sessions(reference, target, samples, "reference-vs-target")
        report.pairs["target"] = detail
        if not detail.get("skipped"):
            report.items.extend(items)

        report.passed = all(bool(detail.get("ok")) for detail in report.pairs.values())
        if not report.passed and self.fail_on_mismatch:
            failures = [name for name, detail in report.pairs.items() if not detail.get("ok")]
            raise ValidationError(
                "精度校验未通过: {}; 详见 report.pairs".format(", ".join(failures))
            )
        return report

    # ------------------------------------------------------------------ 输入
    @staticmethod
    def _adapt_inputs(sample: Dict[str, Any], input_names: Sequence[str]) -> Dict[str, Any]:
        """对齐输入名: 样本键名与模型输入名不一致时按顺序补位。"""
        names = list(input_names)
        if not names:
            return dict(sample)
        if all(name in sample for name in names):
            return {name: sample[name] for name in names}
        values = [np.asarray(value) for value in sample.values()]
        if len(values) == len(names):
            return dict(zip(names, values))
        if len(values) == 1:
            return {names[0]: values[0]}
        raise ValidationError(
            "校验样本与模型输入不匹配: 样本键 {} vs 输入名 {}".format(list(sample.keys()), names)
        )


def load_dataset_inputs(
    dataset: Optional[str],
    input_spec: Dict[str, Sequence[Any]],
    max_samples: int = 8,
    seed: int = 20260101,
) -> Tuple[List[Dict[str, Any]], bool, List[str]]:
    """准备校验输入。

    返回 ``(样本列表, 是否合成数据, 说明)``:

    - 目录内存在 ``.npy`` 时优先加载(文件名升序, 只取前 ``max_samples`` 个);
    - 否则按输入规格生成确定性随机数据, 并标注 ``synthetic``, 避免把"没数据"
      误判成"精度没问题"。
    """
    notes: List[str] = []
    samples: List[Dict[str, Any]] = []

    if dataset:
        root = Path(dataset)
        if not root.is_dir():
            raise ValidationError("校验数据集目录不存在: {}".format(dataset))
        files = sorted(root.glob("*.npy"))[: max(1, int(max_samples))]
        if files:
            for path in files:
                array = np.load(str(path))
                samples.append(_bind_npy(array, input_spec))
            notes.append("从 {} 加载 {} 个 npy 样本".format(dataset, len(files)))
            return samples, False, notes
        notes.append("目录 {} 内未找到 .npy 样本, 改用合成输入".format(dataset))

    for index in range(max(1, int(max_samples))):
        rng = np.random.RandomState(seed + index)
        sample: Dict[str, Any] = {}
        for name, shape in input_spec.items():
            static = [1 if (d is None or int(d) < 0) else int(d) for d in shape]
            sample[name] = rng.rand(*static).astype(np.float32)
        samples.append(sample)
    notes.append("使用确定性随机输入做数值等价校验(seed={}), 不代表真实精度指标".format(seed))
    return samples, True, notes


def _bind_npy(array: np.ndarray, input_spec: Dict[str, Sequence[Any]]) -> Dict[str, Any]:
    """把单个 npy 数组绑定到模型输入(必要时补 batch 维)。"""
    names = list(input_spec.keys())
    if not names:
        return {"input": array}
    expected = list(input_spec[names[0]])
    value = array
    if len(expected) == len(value.shape) + 1:
        value = value[None, ...]
    return {names[0]: value}
