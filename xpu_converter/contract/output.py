# -*- coding: utf-8 -*-
"""输出契约(Output Contract)。

ChatGPT 修改意见 §14 / §15 / §16 的核心结论: **不能再靠 shape 猜后处理**。
本模块把"这个模型的输出到底是什么语义"显式化:

1. 从**实际导出的 ONNX 图**探测 :meth:`ModelOutputContract.detect`;
2. 探测结果写进 ``model.yaml`` / ``runtime.yaml`` 的 ``output`` 段;
3. Runtime 侧由 :class:`OutputContract` 选择解码器(不再用启发式解析)。

支持的布局:

===========  ==========================================================
layout       含义
===========  ==========================================================
``bcn``      单输出 ``(B, 4|5+nc, N)``, YOLOv5/v8 风格
``bnc``      单输出 ``(B, N, 4|5+nc)``
``bnc6``     单输出 ``(B, N, 6)``: ``[x1, y1, x2, y2, conf, cls]`` (端到端)
``pair``     双输出 ``[boxes(B,N,4) xyxy, scores(B,N,nc)]``, NMS 已剥离
``segment``  实例分割: det 分支 ``(B, 4+nc+nm, N)`` + proto ``(B, nm, H/4, W/4)``
``pose``     姿态: det 分支 ``(B, 4+nc+K*3, N)``, 关键点 x/y/conf
``obb``      旋转框: det 分支 ``(B, 4+nc, N)``, rbox(cx,cy,长边,短边,角度)
``cls``      分类: ``(B, nc)`` 类别概率, 无框
``dense``    逐像素图: ``(B, C, H, W)`` (depth=深度, sem=类别索引)
===========  ==========================================================
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

LAYOUT_BCN = "bcn"
LAYOUT_BNC = "bnc"
LAYOUT_BNC6 = "bnc6"
LAYOUT_PAIR = "pair"
LAYOUT_SEGMENT = "segment"
LAYOUT_POSE = "pose"
LAYOUT_OBB = "obb"
LAYOUT_CLS = "cls"
LAYOUT_DENSE = "dense"
LAYOUT_UNKNOWN = "unknown"

# 端到端(图内已含 NMS)产出的列格式
FORMAT_XYXY = "xyxy"
FORMAT_CXCYWH = "cxcywh"
FORMAT_RBOX = "rbox"

SCORE_CLASS = "class"                    # 只有类别置信度
SCORE_OBJECTNESS_CLASS = "objectness_class"
SCORE_CONF_CLS = "conf_cls"              # [x1,y1,x2,y2,conf,cls] 形式


@dataclass
class OutputContract:
    """模型输出的显式契约。"""

    type: str = "detection"
    format: str = FORMAT_CXCYWH
    layout: str = LAYOUT_UNKNOWN
    score_format: str = SCORE_CLASS
    class_axis: Optional[int] = None
    has_objectness: bool = False
    normalized: bool = False
    end2end: bool = False
    nms_embedded: bool = False
    max_det: int = 300
    num_classes: int = 0
    output_shape: List[int] = field(default_factory=list)
    source: str = "detected"          # detected | declared

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "format": self.format,
            "layout": self.layout,
            "score_format": self.score_format,
            "class_axis": self.class_axis,
            "has_objectness": self.has_objectness,
            "normalized": self.normalized,
            "end2end": self.end2end,
            "nms_embedded": self.nms_embedded,
            "max_det": self.max_det,
            "num_classes": self.num_classes,
            "output_shape": list(self.output_shape),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "OutputContract":
        data = dict(data or {})
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        if kwargs.get("output_shape") is None:
            kwargs.pop("output_shape", None)
        return cls(**kwargs)

    @property
    def nms_in_runtime(self) -> bool:
        """是否需要在 Runtime 侧做 CPU NMS。"""
        return self.type == "detection" and not self.end2end

    def summary(self) -> str:
        return "type={} layout={} format={} end2end={} nms_in_runtime={}".format(
            self.type, self.layout, self.format, self.end2end, self.nms_in_runtime
        )


def _static_shape(tensor) -> List[Optional[int]]:
    return [None if dim is None else int(dim) for dim in (getattr(tensor, "shape", None) or [])]


def _is_boxes_like(shape: List[Optional[int]]) -> bool:
    """``(B, N, 4)`` 形态的框张量: 最后一维是 4 且中间维是候选数。"""
    if len(shape) < 2:
        return False
    if shape[-1] != 4:
        return False
    candidates = shape[-2]
    return candidates is None or int(candidates) > 16


class ModelOutputContract:
    """从实际 ONNX 图探测输出契约。"""

    @staticmethod
    def detect(graph, num_classes: int = 0, task: str = "detection",
               declared: Optional[Dict[str, Any]] = None) -> OutputContract:
        """探测输出契约; 若 model.yaml 显式声明了 ``output``, 以声明为准。

        ``task`` 取自 model.yaml 的 ``task`` 字段; 非 detection 任务按任务推断输出
        布局(segment/pose/obb/cls/dense), 避免误判为检测 bnc/bcn。
        """
        detected = _detect_for_task(graph, num_classes=num_classes, task=str(task or "detection"))
        if declared:
            merged = detected.to_dict()
            merged.update({k: v for k, v in declared.items() if v is not None})
            merged["source"] = "declared"
            merged["num_classes"] = int(declared.get("num_classes") or detected.num_classes or num_classes)
            return OutputContract.from_dict(merged)
        return detected


def _detect_for_task(graph, num_classes: int, task: str) -> OutputContract:
    """按任务分派探测: detection 走通用检测布局, 其余按任务语义确定 layout/type。"""
    contract = _detect(graph, num_classes)
    task = (task or "detection").lower()
    contract.type = task

    # 除 detection 外的任务, 布局与语义完全由任务决定(不依赖 shape 启发式)
    if task == "segment":
        contract.layout = LAYOUT_SEGMENT
    elif task == "pose":
        contract.layout = LAYOUT_POSE
        contract.class_axis = 1
    elif task == "obb":
        contract.layout = LAYOUT_OBB
        contract.format = FORMAT_RBOX
        contract.class_axis = 1
    elif task == "cls":
        contract.layout = LAYOUT_CLS
    elif task in ("depth", "sem"):
        contract.layout = LAYOUT_DENSE
    return contract


def _detect(graph, num_classes: int) -> OutputContract:
    nodes = list(getattr(graph, "nodes", []) or [])
    outputs = list(getattr(graph, "outputs", []) or [])
    nms_nodes = [node for node in nodes if getattr(node, "op_type", "") == "NonMaxSuppression"]

    contract = OutputContract(num_classes=int(num_classes or 0))
    contract.nms_embedded = bool(nms_nodes)
    shapes = [_static_shape(tensor) for tensor in outputs]
    contract.output_shape = [dim for dim in (shapes[0] if shapes else []) if dim is not None]

    # 图内仍含 NMS: 端到端输出 (B, N, 6)
    if nms_nodes:
        contract.layout = LAYOUT_BNC6
        contract.format = FORMAT_XYXY
        contract.score_format = SCORE_CONF_CLS
        contract.end2end = True
        contract.class_axis = -1
        if len(shapes) == 1 and len(shapes[0]) == 3 and shapes[0][1]:
            contract.max_det = int(shapes[0][1])
        return contract

    # NMS 已剥离: [boxes(B,N,4), scores(B,N,nc)] 双输出
    if len(outputs) >= 2 and _is_boxes_like(shapes[0]):
        contract.layout = LAYOUT_PAIR
        contract.format = FORMAT_XYXY
        contract.score_format = SCORE_CLASS
        contract.end2end = False
        contract.class_axis = -1
        if len(shapes[1]) >= 3 and shapes[1][-1]:
            contract.num_classes = int(shapes[1][-1])
        if len(shapes[0]) >= 2 and shapes[0][-2]:
            contract.max_det = int(shapes[0][-2])
        return contract

    if not shapes:
        return contract

    shape = [dim for dim in shapes[0] if dim is not None]
    # (B, N, 6): 端到端 6 列
    if len(shape) == 3 and shape[-1] == 6:
        contract.layout = LAYOUT_BNC6
        contract.format = FORMAT_XYXY
        contract.score_format = SCORE_CONF_CLS
        contract.end2end = True
        contract.class_axis = -1
        contract.max_det = int(shape[1])
        return contract

    if len(shape) >= 2:
        last, middle = shape[-1], shape[-2] if len(shape) >= 3 else None
        if _matches_channels(last, num_classes):
            contract.layout = LAYOUT_BNC
            contract.class_axis = -1
            contract.has_objectness = last == 5 + num_classes
            contract.max_det = int(middle) if middle else contract.max_det
            contract.format = FORMAT_CXCYWH
            _fill_num_classes(contract, last, num_classes)
            return contract
        if middle is not None and _matches_channels(middle, num_classes):
            contract.layout = LAYOUT_BCN
            contract.class_axis = 1
            contract.has_objectness = middle == 5 + num_classes
            contract.max_det = int(last) if last else contract.max_det
            contract.format = FORMAT_CXCYWH
            _fill_num_classes(contract, middle, num_classes)
            return contract
    return contract


def _matches_channels(channels: Optional[int], num_classes: int) -> bool:
    if channels is None:
        return False
    channels = int(channels)
    if channels < 4:
        return False
    if num_classes:
        return channels in (4 + num_classes, 5 + num_classes)
    # 无类别数提示时, 只接受"明显是多类别"的通道数, 避免把 4 维框误判为通道
    return channels > 6


def _fill_num_classes(contract: OutputContract, channels: int, num_classes: int) -> None:
    if contract.num_classes:
        return
    # 无显式类别数时按"无 objectness"的 4+nc 推断(YOLOv8/v10 的常见形态)
    contract.num_classes = max(0, int(channels) - 4)
