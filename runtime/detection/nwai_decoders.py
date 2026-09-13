# -*- coding: utf-8 -*-
"""检测后处理解码器(Contract-driven)。

ChatGPT 修改意见 §16 / §17: 过去 ``_parse_outputs`` 用
``shape[-1] == 6`` / ``shape[0] < shape[1]`` 这类启发式猜布局, 一旦猜错
就是"能跑但结果错"。现在布局由算法目录下 ``runtime.yaml`` 的 ``output`` 段
(由转换器在导出后对真实 ONNX 探测得到)显式声明, 本模块据其选择解码器。

布局定义与 ``xpu_converter.contract.output`` 一致:
    bcn    (B, 4|5+nc, N)
    bnc    (B, N, 4|5+nc)
    bnc6   (B, N, 6) 端到端
    pair   [boxes(B,N,4), scores(B,N,nc)]  (NMS 已剥离)
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

EMPTY = (
    np.zeros((0, 4), np.float32),
    np.zeros((0,), np.float32),
    np.zeros((0,), np.float32),
)


def _squeeze_batch(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    return array


class BaseDecoder:
    """解码器基类: 输出统一为 ``(boxes_xyxy, scores, class_ids)``。"""

    layout = "base"

    def __init__(self, contract: Optional[Dict[str, Any]] = None, num_classes: int = 80) -> None:
        self.contract = dict(contract or {})
        self.num_classes = int(self.contract.get("num_classes") or num_classes or 0)
        self.has_objectness = bool(self.contract.get("has_objectness"))

    def decode(self, outputs: List[np.ndarray], conf_thres: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    # -------------------------------------------------------------- 工具
    def _from_columns(self, array: np.ndarray, conf_thres: float):
        """``(N, C)`` 列布局: 支持 4+nc / 5+nc。"""
        if array.ndim != 2 or array.shape[1] < 5:
            return EMPTY
        columns = array.shape[1]
        if self.num_classes:
            has_objectness = columns == 5 + self.num_classes
            class_count = self.num_classes
        else:
            has_objectness = bool(self.has_objectness)
            class_count = max(1, columns - (5 if has_objectness else 4))

        centers = array[:, :4]
        if has_objectness:
            objectness = array[:, 4]
            class_scores = array[:, 5:5 + class_count]
        else:
            objectness = np.ones_like(array[:, 0])
            class_scores = array[:, 4:4 + class_count]
        if class_scores.size == 0:
            return EMPTY

        class_ids = np.argmax(class_scores, axis=1)
        scores = objectness * class_scores[np.arange(class_scores.shape[0]), class_ids]
        mask = scores >= conf_thres
        if not np.any(mask):
            return EMPTY
        cx, cy, width, height = (centers[mask, 0], centers[mask, 1], centers[mask, 2], centers[mask, 3])
        boxes = np.stack([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2], axis=1)
        return boxes.astype(np.float32), scores[mask].astype(np.float32), class_ids[mask].astype(np.float32)

    @staticmethod
    def _from_end2end(array: np.ndarray, conf_thres: float):
        """``(N, 6)``: ``[x1, y1, x2, y2, conf, cls]``。"""
        if array.ndim != 2 or array.shape[1] != 6:
            return EMPTY
        mask = array[:, 4] >= conf_thres
        rows = array[mask]
        return rows[:, :4].astype(np.float32), rows[:, 4].astype(np.float32), rows[:, 5].astype(np.float32)


class TaskObjects:
    """多任务解码结果(NMS 后, 按框候选对齐)。detection 之外的字段按任务填充。"""

    __slots__ = ("boxes", "scores", "class_ids", "keypoints", "angles", "rbox",
                 "mask", "topk_scores", "topk_class_ids", "dense")

    def __init__(self, boxes=None, scores=None, class_ids=None, keypoints=None,
                 angles=None, rbox=None, mask=None, topk_scores=None,
                 topk_class_ids=None, dense=None):
        self.boxes = boxes
        self.scores = scores
        self.class_ids = class_ids
        self.keypoints = keypoints          # (K, 3) 每个候选: x, y, conf
        self.angles = angles                # (n,) 每个候选旋转角(弧度)
        self.rbox = rbox                    # (n, 5) cx,cy,长边,短边,角度
        self.mask = mask                    # mask: (nm,h,w) proto; seg 逐实例裁剪在此之上完成
        self.topk_scores = topk_scores      # cls: (k,) 降序
        self.topk_class_ids = topk_class_ids  # cls: (k,)
        self.dense = dense                  # dense: (h,w) 或 (c,h,w) 逐像素图

    @property
    def empty(self) -> bool:
        return self.boxes is None or self.boxes.size == 0


def _as_nc(array: np.ndarray) -> np.ndarray:
    """把 bcn 输出归一化为行优先 ``(N, C)``(检测头列语义: 前 4 坐标 + 类别 + 任务附加)。"""
    array = np.asarray(array, np.float32)
    if array.ndim == 3:
        array = array[0] if array.shape[0] == 1 else np.transpose(array, (1, 2, 0))
    if array.ndim != 2:
        raise ValueError("期望 2D 检测头输出, 实际 shape={}".format(array.shape))
    # bcn: (C, N) → (N, C)
    return array.T if array.shape[0] < array.shape[1] else array


def _to_xyxy(cx, cy, width, height) -> np.ndarray:
    return np.stack([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2], axis=1)


def _filter_det_rows(box_part, conf_thres, num_classes, has_objectness=False):
    """对检测头列 ``(N, 4|5+nc)`` 做置信度过滤, 返回与原始行对齐的过滤结果。

    返回 ``(boxes_xyxy, scores, class_ids, bool_mask)``; ``bool_mask`` 长度等于
    输入行数, 用于对姿态/分割等**并行列**的附加数据做完全对齐的过滤。
    """
    if box_part.ndim != 2 or box_part.shape[1] < 5:
        return None
    columns = box_part.shape[1]
    if num_classes:
        has_obj = columns == 5 + num_classes
        nc = num_classes
    else:
        has_obj = bool(has_objectness)
        nc = max(1, columns - (5 if has_obj else 4))
    centers = box_part[:, :4]
    if has_obj:
        obj = box_part[:, 4]
        cls_scores = box_part[:, 5:5 + nc]
    else:
        obj = np.ones_like(box_part[:, 0])
        cls_scores = box_part[:, 4:4 + nc]
    if cls_scores.size == 0:
        return None
    class_ids = np.argmax(cls_scores, axis=1)
    scores = obj * cls_scores[np.arange(cls_scores.shape[0]), class_ids]
    mask = scores >= conf_thres
    if not np.any(mask):
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), \
            np.zeros((0,), np.float32), mask
    cx, cy, width, height = (centers[mask, 0], centers[mask, 1], centers[mask, 2], centers[mask, 3])
    boxes = _to_xyxy(cx, cy, width, height)
    return boxes.astype(np.float32), scores[mask].astype(np.float32), \
        class_ids[mask].astype(np.float32), mask


class SegmentDecoder(BaseDecoder):
    """实例分割: ``outputs[0]`` 为 det 分支 ``(B, 4+nc+nm, N)``(bcn),
    ``outputs[1]`` 为 proto ``(B, nm, H/4, W/4)``。
    """

    layout = "segment"

    def __init__(self, contract=None, num_classes=80):
        super().__init__(contract, num_classes)
        self.num_masks = int((self.contract or {}).get("num_masks") or 32)

    def decode(self, outputs, conf_thres):
        if not outputs or len(outputs) < 2:
            return TaskObjects()
        nc_array = _as_nc(outputs[0])
        proto = np.asarray(outputs[1], np.float32)
        # proto 可能是 (nm, H, W) 或 (1, nm, H, W); 统一为 (nm, H, W)
        if proto.ndim == 4 and proto.shape[0] == 1:
            proto = proto[0]
        elif proto.ndim == 3 and proto.shape[0] == 1:
            proto = proto[0]
        if nc_array.shape[1] < 5:
            return TaskObjects()
        columns = nc_array.shape[1]
        box_columns = self.num_classes + 4 if self.num_classes else columns - self.num_masks
        if box_columns < 5 or columns >= box_columns + self.num_masks:
            box_columns = columns - self.num_masks
        box_part = nc_array[:, :box_columns]
        coeff_part = nc_array[:, box_columns:box_columns + self.num_masks]  # (N, nm)
        filtered = _filter_det_rows(box_part, conf_thres, self.num_classes, self.has_objectness)
        if filtered is None:
            return TaskObjects()
        boxes, scores, class_ids, mask = filtered
        if boxes.size == 0 or coeff_part.shape[0] == 0:
            return TaskObjects()
        coeffs = coeff_part[mask]  # 对齐过滤后的候选行
        return TaskObjects(boxes=boxes, scores=scores, class_ids=class_ids,
                           mask=(proto, coeffs))


class PoseDecoder(BaseDecoder):
    """姿态: det 分支 ``(B, 4+nc+K*3, N)``(bcn)。关键点为 x,y,conf。"""

    layout = "pose"

    def decode(self, outputs, conf_thres):
        if not outputs:
            return TaskObjects()
        nc_array = _as_nc(outputs[0])
        columns = nc_array.shape[1]
        # 关键点位于末尾 3*K 列。带 objectness 的姿态头(如 yolov7 IKeypoint)布局为
        # [x,y,w,h,obj,cls, kpt*3], 无 objectness 的(ultralytics)为 [x,y,w,h,cls, kpt*3]。
        # 用「剩余长度能否被 3 整除」来判定是否含 objectness。
        if self.num_classes:
            no_obj = self.num_classes + 4
            if (columns - no_obj) % 3 == 0:
                box_columns, has_obj = no_obj, False
            else:
                box_columns, has_obj = self.num_classes + 5, True
        else:
            box_columns = max(4, columns - (columns - 5) % 3)
            has_obj = bool(self.has_objectness)
        box_part = nc_array[:, :box_columns]
        kpt_part = nc_array[:, box_columns:]  # (N, K*3)
        filtered = _filter_det_rows(box_part, conf_thres, self.num_classes, has_obj)
        if filtered is None:
            return TaskObjects()
        boxes, scores, class_ids, mask = filtered
        if boxes.size == 0 or kpt_part.shape[0] == 0:
            return TaskObjects()
        kpt_flat = kpt_part[mask]
        num_kpt = kpt_flat.shape[1] // 3 if kpt_flat.shape[1] % 3 == 0 else 0
        keypoints = None
        if num_kpt:
            keypoints = kpt_flat.reshape(boxes.shape[0], num_kpt, 3)
        return TaskObjects(boxes=boxes, scores=scores, class_ids=class_ids, keypoints=keypoints)


class ObbDecoder(BaseDecoder):
    """旋转框: det 分支 ``(B, 5+nc, N)``。rbox 为 (cx,cy,长边,短边,角度)。"""

    layout = "obb"

    def decode(self, outputs, conf_thres):
        if not outputs:
            return TaskObjects()
        nc_array = _as_nc(outputs[0])
        columns = nc_array.shape[1]
        # obb 列语义为 ``[cx, cy, w, h, angle, cls...]``(5 + nc)。
        # 实际类别数以列数推导, 避免 yaml 的 num_classes 与真实类别数不一致时越界。
        nc_actual = self.num_classes if (columns == 5 + self.num_classes) else (columns - 5 if columns > 5 else 0)
        if nc_actual <= 0:
            return TaskObjects()
        angle = nc_array[:, 4]
        cls_box = np.concatenate([nc_array[:, :4], nc_array[:, 5:5 + nc_actual]], axis=1)  # (N, 4+nc)
        filtered = _filter_det_rows(cls_box, conf_thres, nc_actual, self.has_objectness)
        if filtered is None:
            return TaskObjects()
        boxes, scores, class_ids, mask = filtered
        if boxes.size == 0:
            return TaskObjects()
        angles = angle[mask] if angle is not None else None
        return TaskObjects(boxes=boxes, scores=scores, class_ids=class_ids, angles=angles)


class ClsDecoder(BaseDecoder):
    """分类: ``(B, nc)`` 或 ``(nc,)`` 类别概率, 无框。"""

    layout = "cls"

    def decode(self, outputs, conf_thres):
        if not outputs:
            return TaskObjects()
        array = _squeeze_batch(np.asarray(outputs[0], np.float32))
        array = array.reshape(-1)
        if array.size == 0:
            return TaskObjects()
        probs = np.clip(array, 0.0, 1.0)
        order = np.argsort(probs)[::-1]
        keep_order = order[probs[order] >= conf_thres]
        if keep_order.size == 0:
            keep_order = order[:1]
        return TaskObjects(
            boxes=None, scores=None, class_ids=None,
            topk_scores=probs[keep_order], topk_class_ids=keep_order.astype(np.float32),
        )


class DenseDecoder(BaseDecoder):
    """逐像素图: depth ``(B, 1, H, W)``; sem ``(B, C, H, W)``。"""

    layout = "dense"

    def decode(self, outputs, conf_thres):
        if not outputs:
            return TaskObjects()
        array = _squeeze_batch(np.asarray(outputs[0], np.float32))
        if array.ndim == 3:
            dense = array
        elif array.ndim == 4:
            dense = array[0]
        else:
            dense = array
        return TaskObjects(dense=dense)


class BcnDecoder(BaseDecoder):
    """单输出 ``(B, C, N)``。"""

    layout = "bcn"

    def decode(self, outputs, conf_thres):
        if not outputs:
            return EMPTY
        array = _squeeze_batch(np.asarray(outputs[0]).astype(np.float32))
        if array.ndim != 2:
            return EMPTY
        # 统一列语义, 交由基类处理
        if array.shape[0] < array.shape[1]:
            array = array.T
        return self._from_columns(array, conf_thres)


class BncDecoder(BaseDecoder):
    """单输出 ``(B, N, C)``。"""

    layout = "bnc"

    def decode(self, outputs, conf_thres):
        if not outputs:
            return EMPTY
        array = _squeeze_batch(np.asarray(outputs[0]).astype(np.float32))
        return self._from_columns(array, conf_thres)


class Bnc6Decoder(BaseDecoder):
    """单输出端到端 ``(B, N, 6)``。"""

    layout = "bnc6"

    def decode(self, outputs, conf_thres):
        if not outputs:
            return EMPTY
        array = _squeeze_batch(np.asarray(outputs[0]).astype(np.float32))
        return self._from_end2end(array, conf_thres)


class PairDecoder(BaseDecoder):
    """双输出 ``[boxes(B,N,4) xyxy, scores(B,N,nc)]``(NMS 已剥离)。"""

    layout = "pair"

    def decode(self, outputs, conf_thres):
        if not outputs or len(outputs) < 2:
            return EMPTY
        boxes = _squeeze_batch(np.asarray(outputs[0]).astype(np.float32))
        scores_raw = _squeeze_batch(np.asarray(outputs[1]).astype(np.float32))
        if boxes.ndim != 2 or boxes.shape[1] != 4 or scores_raw.ndim != 2:
            return EMPTY
        if scores_raw.shape[1] == 1:
            scores = scores_raw[:, 0]
            class_ids = np.zeros_like(scores)
        else:
            class_ids = np.argmax(scores_raw, axis=1)
            scores = scores_raw[np.arange(scores_raw.shape[0]), class_ids]
        mask = scores >= conf_thres
        if not np.any(mask):
            return EMPTY
        return (
            boxes[mask].astype(np.float32),
            scores[mask].astype(np.float32),
            class_ids[mask].astype(np.float32),
        )


_DECODERS = {
    "bcn": BcnDecoder,
    "bnc": BncDecoder,
    "bnc6": Bnc6Decoder,
    "pair": PairDecoder,
    "segment": SegmentDecoder,
    "pose": PoseDecoder,
    "obb": ObbDecoder,
    "cls": ClsDecoder,
    "dense": DenseDecoder,
}


class DecoderFactory:
    """按显式契约创建解码器。"""

    @staticmethod
    def create(contract: Optional[Dict[str, Any]], num_classes: int = 80) -> BaseDecoder:
        contract = dict(contract or {})
        layout = str(contract.get("layout") or "").lower()
        decoder_cls = _DECODERS.get(layout)
        if decoder_cls is None:
            # 契约缺失或不识别: 明确报错, 不再静默猜测
            raise ValueError(
                "输出契约未声明或不可识别的 layout={!r}, 拒绝以启发式方式解码。"
                "请在 model.yaml 的 output 段显式声明, 或重新执行转换以自动探测。".format(layout)
            )
        return decoder_cls(contract, num_classes=num_classes)
