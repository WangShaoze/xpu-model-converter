# -*- coding: utf-8 -*-
"""检测推理与后处理(公共 Runtime 的检测实现)。

V1 约定: **模型只输出 Raw Detection, NMS 在 CPU 侧完成**(建设目标 §17)。
本模块兼容常见 YOLO Raw 输出布局:

- ``(B, N, 6)``            : 端到端输出, 每行 ``[x1, y1, x2, y2, conf, cls]``
- ``(B, 4+nc, N)``         : YOLOv8/v9/v11 无 objectness, 通道优先
- ``(B, N, 4+nc)``         : 同上, 行优先
- ``(B, 5+nc, N)``         : 带 objectness(YOLOv5 风格)
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import runtime_settings as settings
from runtime_backend import get_engine

# 检测结果元组: (model_id, left, top, right, bottom, confidence)
Detection = Tuple[int, int, int, int, int, float]


def letterbox(image: np.ndarray, new_shape: Sequence[int], color: int = 114,
              scale_up: bool = False) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """等比例缩放并灰边填充, 返回 ``(图像, 缩放比, (dw, dh))``。"""
    import cv2

    height, width = image.shape[:2]
    new_height, new_width = int(new_shape[0]), int(new_shape[1])
    ratio = min(new_height / height, new_width / width)
    if not scale_up:
        ratio = min(ratio, 1.0)
    unpad_w, unpad_h = int(round(width * ratio)), int(round(height * ratio))
    dw, dh = (new_width - unpad_w) / 2, (new_height - unpad_h) / 2

    if (width, height) != (unpad_w, unpad_h):
        image = cv2.resize(image, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    if top or bottom or left or right:
        image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=(color, color, color))
    return image, ratio, (dw, dh)


class Detector:
    """检测器: 预处理 + 推理 + CPU NMS。"""

    def __init__(self) -> None:
        self.engine = get_engine(device=settings.DEVICE)
        self.input_shape = settings.INPUT_SHAPE
        self.conf_thres = settings.CONF_THRES
        self.iou_thres = settings.IOU_THRES
        self.max_det = settings.MAX_DET
        self.num_classes = settings.NUM_CLASSES
        self.layout = settings.OUTPUT_LAYOUT
        # 注意: 不能命名为 self.preprocess, 否则会覆盖下面的 preprocess() 方法
        self.preprocess_cfg = settings.PREPROCESS

    # ------------------------------------------------------------------ 预处理
    def preprocess(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """BGR 原图 -> NCHW float32 张量。"""
        import cv2

        target = self.input_shape[-2:] if len(self.input_shape) >= 4 else (640, 640)
        letterboxed, ratio, pad = letterbox(
            image_bgr, target,
            color=int(self.preprocess_cfg.get("pad_value", 114)),
            scale_up=bool(self.preprocess_cfg.get("letterbox_scale_up", False)),
        )
        color = str(self.preprocess_cfg.get("color", "RGB")).upper()
        if color == "RGB":
            letterboxed = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        scale = float(self.preprocess_cfg.get("scale", 1.0 / 255.0))
        tensor = letterboxed.astype(np.float32) * scale
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]  # NCHW
        return np.ascontiguousarray(tensor), ratio, pad

    # ------------------------------------------------------------------ 推理
    def inference(self, tensor: np.ndarray) -> List[np.ndarray]:
        input_name = self.engine.input_names[0]
        return self.engine.run({input_name: tensor})

    def detect(self, image_bgr: np.ndarray, conf_thres: Optional[float] = None,
               iou_thres: Optional[float] = None) -> List[Detection]:
        """返回 ``[(model_id, left, top, right, bottom, confidence), ...]``。

        ``model_id`` 为 1 基类别编号(与交付包 ``confidence.json`` 的 ``model_id`` 对齐)。
        """
        if image_bgr is None:
            return []
        height, width = image_bgr.shape[:2]
        tensor, ratio, pad = self.preprocess(image_bgr)
        outputs = self.inference(tensor)
        detections = self.decode_and_nms(
            outputs, ratio, pad, (width, height),
            conf_thres=conf_thres if conf_thres is not None else self.conf_thres,
            iou_thres=iou_thres if iou_thres is not None else self.iou_thres,
        )
        return detections

    # ------------------------------------------------------------------ 后处理
    def decode_and_nms(self, outputs: List[np.ndarray], ratio: float, pad: Tuple[float, float],
                       original_shape: Tuple[int, int], conf_thres: float,
                       iou_thres: float) -> List[Detection]:
        """Raw 输出 -> 原图坐标检测框(CPU NMS)。"""
        boxes, scores, class_ids = self._parse_outputs(outputs, conf_thres)
        if boxes.size == 0:
            return []

        # 还原到原图坐标
        dw, dh = pad
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio
        width, height = original_shape
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)

        keep = self._nms(boxes, scores, class_ids, iou_thres)
        if keep.size > self.max_det:
            keep = keep[: self.max_det]

        results: List[Detection] = []
        for index in keep:
            x1, y1, x2, y2 = boxes[index]
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            class_id = int(class_ids[index])
            results.append((
                class_id + 1,
                int(round(float(x1))), int(round(float(y1))),
                int(round(float(x2))), int(round(float(y2))),
                float(scores[index]),
            ))
        return results

    def _parse_outputs(self, outputs: List[np.ndarray],
                       conf_thres: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """把不同布局的 Raw 输出统一成 ``(boxes_xyxy, scores, class_ids)``。"""
        if not outputs:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.float32)

        array = np.asarray(outputs[0]).astype(np.float32)
        if array.ndim == 3:
            array = array[0]
        layout = self.layout.lower()

        # 端到端 / 已含 6 列: [x1, y1, x2, y2, conf, cls]
        if layout in ("auto", "end2end", "nms") and array.ndim == 2 and array.shape[-1] == 6:
            return self._from_end2end(array, conf_thres)
        if layout == "end2end":
            raise ValueError("输出布局与 model.yaml 声明的 end2end 不一致: {}".format(array.shape))

        if array.ndim != 2:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.float32)

        # 通道优先 (4+nc, N) 或 (5+nc, N) → 转置为 (N, C)
        if array.shape[0] < array.shape[1] and array.shape[0] in (6, 5 + self.num_classes, 4 + self.num_classes):
            array = array.T
        return self._from_columns(array, conf_thres)

    @staticmethod
    def _from_end2end(array: np.ndarray, conf_thres: float):
        mask = array[:, 4] >= conf_thres
        rows = array[mask]
        boxes = rows[:, :4].copy()
        scores = rows[:, 4].copy()
        class_ids = rows[:, 5].copy()
        return boxes, scores, class_ids

    def _from_columns(self, array: np.ndarray, conf_thres: float):
        """列布局: 支持 4+nc(无 objectness) 与 5+nc(带 objectness)。"""
        columns = array.shape[1]
        has_objectness = columns in (5 + self.num_classes,)
        if not has_objectness and columns != 4 + self.num_classes and columns < 5:
            # 未知列数: 尽量按 4 + 类别数 推断
            class_count = max(1, columns - 4)
        else:
            class_count = self.num_classes if (has_objectness or columns == 4 + self.num_classes) else columns - 4

        centers = array[:, :4]
        if has_objectness:
            objectness = array[:, 4]
            class_scores = array[:, 5:5 + class_count]
        else:
            objectness = np.ones_like(array[:, 0])
            class_scores = array[:, 4:4 + class_count]

        if class_scores.size == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.float32)

        class_ids = np.argmax(class_scores, axis=1)
        scores = objectness * class_scores[np.arange(class_scores.shape[0]), class_ids]
        mask = scores >= conf_thres
        if not np.any(mask):
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.float32)

        cx, cy, width, height = (centers[mask, 0], centers[mask, 1], centers[mask, 2], centers[mask, 3])
        boxes = np.stack([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2], axis=1)
        return boxes, scores[mask], class_ids[mask].astype(np.float32)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
             iou_thres: float) -> np.ndarray:
        """逐类别 NMS, 返回保留下标(按置信度降序)。"""
        if boxes.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)
        keep_all: List[int] = []
        for class_id in np.unique(class_ids):
            indexes = np.where(class_ids == class_id)[0]
            keep_all.extend(_nms_single(boxes[indexes], scores[indexes], iou_thres, indexes))
        keep_all.sort(key=lambda idx: float(scores[idx]), reverse=True)
        return np.asarray(keep_all, dtype=np.int64)


def _nms_single(boxes: np.ndarray, scores: np.ndarray, iou_thres: float,
                indexes: np.ndarray) -> List[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        current = order[0]
        keep.append(int(indexes[current]))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[current] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_thres]
    return keep


_DETECTOR: Optional[Detector] = None


def get_detector() -> Detector:
    """进程内单例检测器(每个 gunicorn worker 各持有一份)。"""
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = Detector()
    return _DETECTOR


def detect(image_bgr: np.ndarray, conf_thres: Optional[float] = None,
           iou_thres: Optional[float] = None) -> List[Detection]:
    return get_detector().detect(image_bgr, conf_thres=conf_thres, iou_thres=iou_thres)


def describe() -> Dict[str, Any]:
    detector = get_detector()
    return {
        "backend": detector.engine.backend,
        "device": detector.engine.device,
        "input_names": detector.engine.input_names,
        "output_names": detector.engine.output_names,
        "input_shape": detector.input_shape,
        "conf_thres": detector.conf_thres,
        "iou_thres": detector.iou_thres,
    }
