# -*- coding: utf-8 -*-
"""精度与性能校验层。

- :mod:`xpu_converter.validator.tensor_compare` 张量级数值比对
- :mod:`xpu_converter.validator.accuracy`       三路精度校验
- :mod:`xpu_converter.validator.benchmark`      推理性能基准
"""
from xpu_converter.validator.accuracy import AccuracyReport, AccuracyValidator, load_dataset_inputs
from xpu_converter.validator.application import (
    DET_LAYOUTS,
    build_evaluator,
    cls_evaluator,
    compute_oks,
    decode_cls,
    decode_depth,
    decode_detections,
    decode_obb,
    decode_pose,
    decode_poses,
    decode_segment,
    decode_sem,
    depth_evaluator,
    detection_evaluator,
    eval_detections,
    iou_matrix,
    nms,
    obb_evaluator,
    pose_ap_evaluator,
    pose_evaluator,
    segment_evaluator,
    sem_evaluator,
)
from xpu_converter.validator.benchmark import BenchmarkResult, BenchmarkRunner
from xpu_converter.validator.tensor_compare import (
    DEFAULT_ATOL,
    DEFAULT_COSINE,
    DEFAULT_RTOL,
    TensorCompareResult,
    compare_outputs,
    compare_tensors,
    summarize,
)

__all__ = [
    "DET_LAYOUTS",
    "AccuracyReport",
    "AccuracyValidator",
    "BenchmarkResult",
    "BenchmarkRunner",
    "DEFAULT_ATOL",
    "DEFAULT_COSINE",
    "DEFAULT_RTOL",
    "TensorCompareResult",
    "build_evaluator",
    "cls_evaluator",
    "compare_outputs",
    "compare_tensors",
    "compute_oks",
    "decode_cls",
    "decode_depth",
    "decode_detections",
    "decode_obb",
    "decode_pose",
    "decode_poses",
    "decode_segment",
    "decode_sem",
    "depth_evaluator",
    "detection_evaluator",
    "eval_detections",
    "iou_matrix",
    "load_dataset_inputs",
    "nms",
    "obb_evaluator",
    "pose_ap_evaluator",
    "pose_evaluator",
    "segment_evaluator",
    "sem_evaluator",
    "summarize",
]
