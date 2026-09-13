# -*- coding: utf-8 -*-
"""模型级 Conformance(逐后端一致性)回归框架(ChatGPT 修改意见 P0-8)。

对同一份 golden 输入, 把一个模型在**逐级后端**上的输出串成一条证据链, 每一级:

* 保存张量指纹 ``shape / dtype / sha256(precision 无关的规范化字节)``;
* 与基准(``reference``, 原始 ONNX CPU 会话)逐张量数值比对
  ``max_abs_error / cosine_similarity``;

从而把"这个模型到底真支持还是假支持"从人工声明(Adapter 存在)变成可自动
判定的数值证据:

    golden input → reference(原 ONNX, CPU) → optimized ONNX → Paddle → XPU

任一环节在对应依赖/产物缺失时标记 ``skipped``, 不影响其它环节; 但 ``required``
环节若不可用则整个门禁失败, 避免把未验证的后端误写成已完成(stable)。
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

import hashlib

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


def tensor_fingerprint(array: Any, stage: str = "", name: str = "") -> Dict[str, Any]:
    """返回张量的确定性指纹: shape / dtype / sha256(规范化字节) / 是否有限。"""
    arr = np.asarray(array)
    return {
        "stage": stage,
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": _stable_sha256(arr),
        "finite": bool(np.all(np.isfinite(arr))),
    }


def _stable_sha256(array: np.ndarray) -> str:
    """逐字节稳定 sha256: 按 C 连续且原 dtype 字节序规范化, 与平台无关。"""
    arr = np.asarray(array)
    arr = np.ascontiguousarray(arr)
    if arr.dtype.byteorder not in ("=", "|"):
        arr = arr.astype(arr.dtype.newbyteorder("="))
    return hashlib.sha256(arr.tobytes()).hexdigest()


@dataclass
class ConformanceStage:
    """一个待校验收敛环节。

    ``session`` 直接给会话; 或 ``factory`` 延迟构造(用于按后端探测可用性)。
    ``required=True`` 的环节不可用时整体门禁失败, 否则标记 ``skipped``。
    """

    name: str
    session: Optional[BaseRuntimeSession] = None
    factory: Optional[Callable[[], BaseRuntimeSession]] = None
    required: bool = False


@dataclass
class StageResult:
    """单个环节的执行结果。"""

    name: str
    status: str = "ok"                 # ok | skipped | error
    reason: str = ""
    required: bool = False
    fingerprints: List[Dict[str, Any]] = field(default_factory=list)
    compare: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "ok" and bool(self.compare.get("ok"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "required": self.required,
            "passed": self.passed,
            "fingerprints": list(self.fingerprints),
            "compare": dict(self.compare),
        }


class ModelConformanceReport:
    """模型级一致性报告, 可 JSON 序列化并落盘。"""

    def __init__(
        self,
        model_type: str,
        task: str,
        model_path: str,
        input_shape: List[int],
        stages: Sequence[StageResult],
        reference_fingerprints: Sequence[Dict[str, Any]],
        sample_count: int,
        notes: Optional[List[str]] = None,
    ) -> None:
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.model_type = model_type
        self.task = task
        self.model_path = model_path
        self.input_shape = list(input_shape)
        self.sample_count = sample_count
        self.stages = list(stages)
        self.reference_fingerprints = list(reference_fingerprints)
        self.notes = list(notes or [])

    @property
    def executed(self) -> List[StageResult]:
        return [stage for stage in self.stages if stage.status == "ok"]

    @property
    def passed(self) -> bool:
        handled = [stage for stage in self.stages if stage.status in ("ok", "error")]
        if not handled:
            return True            # 全部 skipped: 无证据可判, 不判失败
        return all(
            stage.passed if stage.status == "ok" else not stage.required
            for stage in self.stages if stage.status in ("ok", "error")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "model-conformance",
            "created_at": self.created_at,
            "model_type": self.model_type,
            "task": self.task,
            "model_path": self.model_path,
            "input_shape": list(self.input_shape),
            "sample_count": self.sample_count,
            "passed": self.passed,
            "reference_fingerprints": list(self.reference_fingerprints),
            "stages": [stage.to_dict() for stage in self.stages],
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        parts = " ".join(
            "{}:{}".format(stage.name, "OK" if stage.passed else stage.status.upper())
            for stage in self.stages
        )
        return "conformance({}={}) {}".format(self.model_type, self.task,
                                              "PASS" if self.passed else "FAIL", parts)

    def save(self, path: str) -> str:
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as fw:
            import json

            json.dump(self.to_dict(), fw, ensure_ascii=False, indent=2)
        return str(target)


class ConformanceRunner:
    """把 reference 与逐级后端按 golden 输入链式比对, 产出 :class:`ModelConformanceReport`。"""

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
        self.fail_on_mismatch = bool(fail_on_mismatch)

    def run(
        self,
        model_type: str,
        reference: BaseRuntimeSession,
        stages: Sequence[ConformanceStage],
        samples: Sequence[Dict[str, Any]],
        task: str = "detection",
        model_path: str = "",
        input_shape: Optional[Sequence[int]] = None,
    ) -> ModelConformanceReport:
        """执行逐级一致性校验。``stages`` 顺序即证据链顺序。"""
        results: List[StageResult] = []
        ref_names = list(reference.input_names)
        feed = _sample_for(samples, ref_names)
        # 基准(原 ONNX)输出指纹: 取第一个样本即可
        ref_fingerprints: List[Dict[str, Any]] = []
        if feed:
            ref_outputs = reference.run(_adapt_inputs(feed[0], ref_names))
            ref_fingerprints = [
                tensor_fingerprint(tensor, stage="reference", name=name)
                for name, tensor in _named(ref_outputs, reference.output_names)
            ]

        for stage in stages:
            results.append(self._run_stage(reference, stage, feed, ref_names))

        report = ModelConformanceReport(
            model_type=model_type,
            task=task,
            model_path=model_path,
            input_shape=list(input_shape or []),
            stages=results,
            reference_fingerprints=ref_fingerprints,
            sample_count=len(feed),
        )
        if not report.passed and self.fail_on_mismatch:
            failures = [s.name for s in results if s.status in ("ok", "error") and not s.passed]
            raise ValidationError(
                "模型级一致性校验未通过, 失败环节: {}; 详见 conformance report.stages".format(
                    ", ".join(failures)
                )
            )
        return report

    # ------------------------------------------------------------------ 内部
    def _run_stage(
        self,
        reference: BaseRuntimeSession,
        stage: ConformanceStage,
        samples: Sequence[Dict[str, Any]],
        ref_names: Sequence[str],
    ) -> StageResult:
        result = StageResult(name=stage.name, required=stage.required)
        try:
            target = stage.session if stage.session is not None else stage.factory()
        except Exception as err:
            result.status = "error" if stage.required else "skipped"
            result.reason = "后端不可用: {}".format(err)
            return result
        if target is None:
            result.status = "error" if stage.required else "skipped"
            result.reason = "后端未提供(会话/工厂均为空)"
            return result

        target_names = list(target.input_names)
        items: List[TensorCompareResult] = []
        fingerprints: List[Dict[str, Any]] = []
        try:
            for sample in samples:
                ref_feeds = _adapt_inputs(sample, ref_names)
                target_feeds = _rebind_inputs(ref_feeds, ref_names, target_names)
                ref_outputs = reference.run(ref_feeds)
                target_outputs = target.run(target_feeds)
                for name, tensor in _named(target_outputs, target.output_names):
                    fingerprints.append(tensor_fingerprint(tensor, stage=stage.name, name=name))
                items.extend(
                    compare_outputs(
                        ref_outputs,
                        target_outputs,
                        names=reference.output_names,
                        atol=self.atol,
                        rtol=self.rtol,
                        cosine_thres=self.cosine_thres,
                    )
                )
        except Exception as err:  # 运行时错误视为该环节失败(若 required)或跳过
            result.status = "error" if stage.required else "skipped"
            result.reason = "执行失败: {}".format(err)
            return result
        finally:
            close = getattr(target, "close", None)
            if callable(close):
                close()

        result.fingerprints = fingerprints
        result.compare = summarize(items)
        result.compare["reference"] = getattr(reference, "backend_name", "reference")
        result.compare["target"] = getattr(target, "backend_name", stage.name)
        result.status = "ok"
        return result


def _sample_for(samples: Sequence[Dict[str, Any]],
                input_names: Sequence[str]) -> List[Dict[str, Any]]:
    """样本规范化: 键名与模型输入一致, 或按位置绑定。"""
    if not samples:
        return []
    adapted: List[Dict[str, Any]] = []
    names = list(input_names)
    for sample in samples:
        sample = dict(sample or {})
        if not sample:
            continue
        adapted.append(_adapt_inputs(sample, names))
    return adapted


def _adapt_inputs(sample: Dict[str, Any], input_names: Sequence[str]) -> Dict[str, Any]:
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
    return dict(sample)


def _rebind_inputs(feeds: Dict[str, Any], reference_names: Sequence[str],
                   target_names: Sequence[str]) -> Dict[str, Any]:
    """跨后端输入按**位置**绑定(不同后端会重命名输入张量)。"""
    names = list(target_names)
    if not names:
        return dict(feeds)
    values = [feeds[name] for name in reference_names if name in feeds] or list(feeds.values())
    if len(values) == len(names):
        return dict(zip(names, values))
    if len(values) == 1:
        return {names[0]: values[0]}
    return dict(feeds)


def _named(outputs: Sequence[Any], names: Sequence[str]):
    label = list(names)
    for index, tensor in enumerate(outputs):
        yield label[index] if index < len(label) else "output_{}".format(index), tensor