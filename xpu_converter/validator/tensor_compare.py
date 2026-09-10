# -*- coding: utf-8 -*-
"""张量级数值比对。

用于 PyTorch / ONNX / XPU 三路精度校验: 把参考输出与目标输出逐张量比较,
给出最大绝对误差、相对误差与余弦相似度, 并按阈值判定是否通过。
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_ATOL = 1e-3
DEFAULT_RTOL = 1e-3
DEFAULT_COSINE = 0.999


@dataclass
class TensorCompareResult:
    """单个输出张量的比对结果。"""

    name: str = ""
    shape: Tuple[int, ...] = ()
    max_abs_error: float = 0.0
    mean_abs_error: float = 0.0
    max_rel_error: float = 0.0
    cosine_similarity: float = 1.0
    passed: bool = True
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "max_rel_error": self.max_rel_error,
            "cosine_similarity": self.cosine_similarity,
            "passed": self.passed,
            "reason": self.reason,
        }

    def summary(self) -> str:
        return "{} max_abs={:.3e} cosine={:.6f} {}".format(
            self.name or "output", self.max_abs_error, self.cosine_similarity,
            "OK" if self.passed else "FAIL({})".format(self.reason),
        )


def compare_tensors(
    reference,
    actual,
    name: str = "",
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    cosine_thres: float = DEFAULT_COSINE,
) -> TensorCompareResult:
    """比对两个张量。"""
    ref = np.asarray(reference, dtype=np.float64)
    act = np.asarray(actual, dtype=np.float64)
    result = TensorCompareResult(name=name, shape=tuple(act.shape))

    if ref.shape != act.shape:
        result.passed = False
        result.reason = "形状不一致: {} vs {}".format(tuple(ref.shape), tuple(act.shape))
        return result

    diff = np.abs(ref - act)
    result.max_abs_error = float(diff.max()) if diff.size else 0.0
    result.mean_abs_error = float(diff.mean()) if diff.size else 0.0
    denominator = np.maximum(np.abs(ref), 1e-12)
    result.max_rel_error = float((diff / denominator).max()) if diff.size else 0.0
    result.cosine_similarity = _cosine(ref, act)

    if not np.all(np.isfinite(act)):
        result.passed = False
        result.reason = "目标输出包含 NaN/Inf"
        return result

    tolerance = atol + rtol * np.abs(ref)
    if np.all(diff <= tolerance):
        result.passed = True
        return result
    if result.cosine_similarity >= cosine_thres:
        # 数值抖动但方向一致(常见于 FP16), 记录为通过并标注
        result.passed = True
        result.reason = "超出逐元素阈值但余弦相似度达标(通常为精度损失)"
        return result

    result.passed = False
    result.reason = "超出容差 atol={:.1e} rtol={:.1e} 且余弦相似度 {:.6f} < {:.6f}".format(
        atol, rtol, result.cosine_similarity, cosine_thres
    )
    return result


def compare_outputs(
    reference_outputs: Sequence[Any],
    actual_outputs: Sequence[Any],
    names: Optional[Sequence[str]] = None,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    cosine_thres: float = DEFAULT_COSINE,
) -> List[TensorCompareResult]:
    """按顺序比对两组输出。"""
    if len(reference_outputs) != len(actual_outputs):
        raise ValueError(
            "输出数量不一致: {} vs {}".format(len(reference_outputs), len(actual_outputs))
        )
    names = list(names or [])
    results: List[TensorCompareResult] = []
    for index, (ref, act) in enumerate(zip(reference_outputs, actual_outputs)):
        label = names[index] if index < len(names) else "output_{}".format(index)
        results.append(compare_tensors(ref, act, name=label, atol=atol, rtol=rtol, cosine_thres=cosine_thres))
    return results


def summarize(results: Sequence[TensorCompareResult]) -> Dict[str, Any]:
    """汇总多张量比对结果。"""
    total = len(results)
    failed = [item for item in results if not item.passed]
    return {
        "count": total,
        "passed": total - len(failed),
        "failed": len(failed),
        "ok": not failed,
        "max_abs_error": max((item.max_abs_error for item in results), default=0.0),
        "min_cosine_similarity": min((item.cosine_similarity for item in results), default=1.0),
        "items": [item.to_dict() for item in results],
    }


def _cosine(reference: np.ndarray, actual: np.ndarray) -> float:
    ref = reference.reshape(-1)
    act = actual.reshape(-1)
    if ref.size == 0:
        return 1.0
    ref_norm = float(np.linalg.norm(ref))
    act_norm = float(np.linalg.norm(act))
    if ref_norm == 0.0 or act_norm == 0.0:
        return 1.0 if ref_norm == act_norm else 0.0
    return float(np.dot(ref, act) / (ref_norm * act_norm))
