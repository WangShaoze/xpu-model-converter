# -*- coding: utf-8 -*-
"""精度校验(三层, ChatGPT 修改意见 §19/§20)。

按 Validation Level 分成三层, 并分别给出结论:

    Level 1 Graph       : 原 ONNX        vs 优化 ONNX
    Level 2 Backend     : 优化 ONNX(CPU) vs 目标后端(XPU)
    Level 3 Application : 检测指标(IoU / mAP), 需带标注数据集时才有意义

三层各自独立记录, 不再把"数值等价"与"端到端精度"混为一谈; 缺少应用级数据时
显式标注 ``available=false``, 绝不伪造 mAP。
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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

# 应用级校验回调: (reference, target, samples) -> dict, 由调用方注入检测评测逻辑
ApplicationEvaluator = Callable[..., Dict[str, Any]]


@dataclass
class AccuracyReport:
    """三层精度校验报告。

    P0-1(禁止 Validator 自动 CPU fallback): 报告必须携带执行溯源
    (:attr:`execution`), 并且 :attr:`available` 只有在目标确实运行在真实 XPU
    上才为真。CPU 回退下的"数值一致"只能算 NOT_AVAILABLE, 不能算 PASS。
    """

    passed: bool = True
    graph: Dict[str, Any] = field(default_factory=dict)
    backend: Dict[str, Any] = field(default_factory=dict)
    application: Dict[str, Any] = field(default_factory=dict)
    sample_count: int = 0
    synthetic_inputs: bool = False
    # 目标会话是否真的运行在目标 XPU 上; 否则整个精度结论判 NOT_AVAILABLE(P0-1)
    available: bool = True
    # 执行溯源(P1-4): reference/target 各自的实际设备与运行模式
    execution: Dict[str, Any] = field(default_factory=dict)
    items: List[TensorCompareResult] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """验收语义: 目标未跑在真实 XPU 上 → NOT_AVAILABLE; 否则按数值结论 PASS/FAIL。"""
        if not self.available:
            return "NOT_AVAILABLE"
        return "PASS" if self.passed else "FAIL"

    @property
    def pairs(self) -> Dict[str, Dict[str, Any]]:
        """向后兼容: 仅返回已执行的数值层(graph / backend)。"""
        return {name: detail for name, detail in (("graph", self.graph), ("backend", self.backend)) if detail}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "available": self.available,
            "status": self.status,
            "sample_count": self.sample_count,
            "synthetic_inputs": self.synthetic_inputs,
            "graph": dict(self.graph),
            "backend": dict(self.backend),
            "application": dict(self.application),
            "execution": dict(self.execution),
            "overall": summarize(self.items) if self.items else {},
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        parts: List[str] = []
        if not self.available:
            return "NOT_AVAILABLE (target 未运行在真实 XPU: {})".format(
                self.execution.get("target_actual_device") or "unknown"
            )
        for name, detail in (("graph", self.graph), ("backend", self.backend)):
            if not detail:
                continue
            parts.append("{}:{}".format(
                name, "SKIP" if detail.get("skipped") else ("OK" if detail.get("ok") else "FAIL")
            ))
        if self.application:
            parts.append("app:{}".format(
                "N/A" if not self.application.get("available") else
                ("OK" if self.application.get("passed", True) else "FAIL")
            ))
        if not parts:
            return "跳过(无校验样本)"
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
        ref_names = list(input_names or reference.input_names)
        target_names = list(target.input_names)
        for sample in samples:
            ref_feeds = self._adapt_inputs(sample, ref_names)
            target_feeds = self._rebind_inputs(ref_feeds, ref_names, target_names)
            reference_outputs = reference.run(ref_feeds)
            target_outputs = target.run(target_feeds)
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
        detail["reference"] = _session_name(reference)
        detail["target"] = _session_name(target)
        return detail, ref_items

    def validate(
        self,
        reference: BaseRuntimeSession,
        target: BaseRuntimeSession,
        samples: Sequence[Dict[str, Any]],
        graph_reference: Optional[BaseRuntimeSession] = None,
        synthetic: bool = False,
        notes: Optional[List[str]] = None,
        application_evaluator: Optional[ApplicationEvaluator] = None,
        num_classes: Optional[int] = None,
    ) -> AccuracyReport:
        """执行三层精度校验(ChatGPT 修改意见 §19)。

        - Level 1 Graph: ``graph_reference``(原 ONNX) vs ``reference``(优化后 ONNX);
        - Level 2 Backend: ``reference``(优化 ONNX, CPU) vs ``target``(待验证后端);
        - Level 3 Application: 由 ``application_evaluator`` 提供(缺省则标注不可用)。
        """
        report = AccuracyReport(sample_count=len(samples), synthetic_inputs=synthetic)
        report.notes.extend(notes or [])
        report.execution = {
            "reference": _execution_provenance(reference),
            "target": _execution_provenance(target),
            "target_actual_device": str(getattr(target, "actual_device", "") or ""),
            "target_execution_mode": str(getattr(target, "execution_mode", "") or ""),
        }

        # P0-1 硬门禁: 目标后端必须真正运行在昆仑 XPU 上, 否则整个精度结论
        # 判 NOT_AVAILABLE。CPU 回退下的"数值一致"不代表交给客户的精度通过。
        target_device = str(getattr(target, "actual_device", "") or "")
        if target_device and target_device not in ("xpu", "kunlun"):
            report.available = False
            report.notes.append(
                "target 未运行在真实昆仑 XPU(actual_device={}, mode={}): 精度结论标记为 "
                "NOT_AVAILABLE, 需在真实 XPU 环境复验后才能判定 PASS".format(
                    target_device, str(getattr(target, "execution_mode", "") or "")
                )
            )

        if graph_reference is not None:
            detail, items = self.compare_sessions(
                graph_reference, reference, samples, "onnx-vs-optimized"
            )
            report.graph = detail
            if not detail.get("skipped"):
                report.items.extend(items)

        detail, items = self.compare_sessions(reference, target, samples, "onnx-vs-target")
        report.backend = detail
        if not detail.get("skipped"):
            report.items.extend(items)

        if application_evaluator is not None:
            try:
                report.application = dict(application_evaluator(
                    reference=reference, target=target, samples=samples, num_classes=num_classes,
                ) or {})
            except Exception as err:  # 应用级评测失败不应掩盖数值层结论
                report.application = {"available": False, "reason": "应用级校验执行失败: {}".format(err)}
                report.notes.append(report.application["reason"])
        else:
            report.application = {
                "available": False,
                "reason": "未提供带标注的检测数据集, 跳过应用级(IoU/mAP)校验",
            }

        numeric = [detail for detail in (report.graph, report.backend)
                   if detail and not detail.get("skipped")]
        report.passed = all(bool(detail.get("ok")) for detail in numeric) and report.available
        # 仅在目标确实跑在真实 XPU 上时才把数值不一致视为"校验失败"; CPU 回退属
        # 不可用而非不通过, 交给上层按 NOT_AVAILABLE 处理, 不抛 ValidationError。
        if report.available and not all(bool(detail.get("ok")) for detail in numeric) and self.fail_on_mismatch:
            failures = [name for name, detail in (("graph", report.graph), ("backend", report.backend))
                        if detail and not detail.get("skipped") and not detail.get("ok")]
            raise ValidationError(
                "精度校验未通过: {}; 详见 report.graph / report.backend".format(", ".join(failures))
            )
        return report

    # ------------------------------------------------------------------ 输入
    @staticmethod
    def _rebind_inputs(feeds: Dict[str, Any], reference_names: Sequence[str],
                       target_names: Sequence[str]) -> Dict[str, Any]:
        """把参考会话的输入按**位置**重绑到目标会话。

        不同后端会重命名张量(x2paddle 会把输入改名为 ``x2paddle_<name>``), 因此跨
        后端比对只能按位置对齐, 不能按名字匹配。
        """
        names = list(target_names)
        if not names:
            return dict(feeds)
        values = [feeds[name] for name in reference_names if name in feeds] or list(feeds.values())
        if len(values) == len(names):
            return dict(zip(names, values))
        if len(values) == 1:
            return {names[0]: values[0]}
        return dict(feeds)

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


def _session_name(session: BaseRuntimeSession) -> str:
    """会话标识, 用于报告的 reference/target 字段(如 ``onnxruntime`` / ``kunlun-xpu``)。"""
    return str(getattr(session, "backend_name", "") or type(session).__name__)


def _execution_provenance(session: BaseRuntimeSession) -> Dict[str, Any]:
    """提取会话的执行溯源(P0-1/P1-4)。"""
    provenance = getattr(session, "execution_provenance", None)
    if callable(provenance):
        value = provenance()
        return dict(value or {})
    return {
        "backend": _session_name(session),
        "actual_device": str(getattr(session, "actual_device", "") or ""),
        "execution_mode": str(getattr(session, "execution_mode", "") or ""),
        "device_available": bool(getattr(session, "device_available", False)),
        "device_id": int(getattr(session, "device_id", 0) or 0),
    }


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
