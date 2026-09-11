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
