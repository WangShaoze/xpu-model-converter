# -*- coding: utf-8 -*-
"""应用级精度指标(ChatGPT 修改意见 P0-9)。

"不能只靠 generic tensor compare": Level 1/2 的 ``max_abs_error / cosine`` 只能
证明"数值等价", 不能证明"业务正确"。本模块提供**任务级**指标:

* detection: mAP / mAP50 / AP(逐类) / precision / recall / F1(纯 numpy, 无额外依赖,
  兼容 bnc / bcn / bnc6 / pair 契约布局, 解码 → NMS → 匹配 GT);
* pose / seg / cls / dense: 因缺标准标注评估器, 返回 ``available=false`` 并附明确
  原因(P0-9 与 §20 一致: 无数据/未实现绝不伪造指标)。

GT 以与样本对齐的 record 列表传入(每元素 ``{"gt": (N,4)xyxy, "gt_labels": (N,)}``
或含预测 ``pred/scores/labels``), 由调用方/数据集准备阶段提供。缺 GT 时评估器
诚实置 ``available=false``。
"""
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: 可用作 application-level 的检测契约布局
DET_LAYOUTS = frozenset({"bcn", "bnc", "bnc6", "pair"})

#: 与 AccuracyValidator.application_evaluator 的类型契约一致
ApplicationEvaluatorLike = Callable[..., Dict[str, Any]]


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """逐对 IoU(均视为 xyxy)。``boxes_a`` (A,4), ``boxes_b`` (B,4) → (A,B)。"""
    boxes_a = np.asarray(boxes_a, np.float32).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, np.float32).reshape(-1, 4)
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), np.float32)
    a_area = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    b_area = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    inter = np.clip(rb - lt, 0.0, None).prod(axis=2)
    union = a_area[:, None] + b_area[None, :] - inter
    union = np.maximum(union, 1e-6)
    return inter / union


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """贪心 NMS, 返回保留框的索引(按置信度降序)。"""
    boxes = np.asarray(boxes, np.float32)
    scores = np.asarray(scores, np.float32)
    if boxes.ndim != 2 or boxes.shape[0] == 0:
        return np.zeros(0, np.int64)
    order = np.argsort(-scores)
    keep: List[int] = []
    while order.size:
        idx = order[0]
        keep.append(int(idx))
        if order.size == 1:
            break
        rest = order[1:]
        overlaps = iou_matrix(boxes[idx:idx + 1], boxes[rest])[0]
        order = rest[overlaps <= iou_thres]
    return np.asarray(keep, np.int64)


def compute_ap(all_scores: Sequence[float], all_tp: Sequence[bool],
               num_gt: int) -> Tuple[float, float, float]:
    """按置信度降序累计, 101 点插值召回下的 AP, 并返回末操作点 P/R。

    返回 ``(ap, precision, recall)``。
    """
    scores = np.asarray(all_scores, np.float32)
    tp = np.asarray(all_tp, bool)
    if scores.size == 0:
        return 0.0, 0.0, (1.0 if num_gt == 0 else 0.0)
    order = np.argsort(-scores)
    tp = tp[order]
    cum_tp = np.cumsum(tp).astype(np.float64)
    cum_fp = np.cumsum(~tp).astype(np.float64)
    recall = cum_tp / max(num_gt, 1)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    # 101-point interpolated precision → AP(PASCAL VOC / COCO 风格)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    for index in range(len(mpre) - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))
    return ap, float(prec[-1] if prec.size else 0.0), float(recall[-1] if recall.size else 0.0)


def eval_detections(
    records: Sequence[Dict[str, Any]],
    iou_thres: float = 0.5,
) -> Tuple[Dict[int, Dict[str, Any]], float]:
    """对一批图像(record)在不同类别上累积 tp/fp, 计算单阈值 AP 与 mAP。

    每 record: ``pred`` (P,4) xyxy, ``scores`` (P,), ``labels`` (P,),
    ``gt`` (G,4) xyxy, ``gt_labels`` (G,)。
    返回 ``(per_class, mAP@iou_thres)``。
    """
    records = [dict(record) for record in records if record]
    class_ids: set = set()
    for record in records:
        class_ids.update(np.asarray(record.get("labels", []), np.int64).tolist())
        class_ids.update(np.asarray(record.get("gt_labels", []), np.int64).tolist())
    per_class: Dict[int, Dict[str, Any]] = {}
    for cls in sorted(class_ids):
        scores: List[float] = []
        tp: List[bool] = []
        num_gt = 0
        for record in records:
            pred_c = np.asarray(record["labels"], np.int64) == cls
            boxes = np.asarray(record["pred"], np.float32)[pred_c]
            s = np.asarray(record["scores"], np.float32)[pred_c]
            gt_c = np.asarray(record["gt_labels"], np.int64) == cls
            gt = np.asarray(record["gt"], np.float32)[gt_c]
            num_gt += int(gt.shape[0])
            if boxes.shape[0] == 0:
                continue
            if gt.shape[0] == 0:
                scores.extend(s.tolist())
                tp.extend([False] * int(boxes.shape[0]))
                continue
            overlaps = iou_matrix(boxes, gt)
            order = np.argsort(-s)
            matched_gt: set = set()
            for pred_index in order:
                best = int(np.argmax(overlaps[int(pred_index)]))
                if overlaps[int(pred_index), best] >= iou_thres and best not in matched_gt:
                    matched_gt.add(best)
                    scores.append(float(s[int(pred_index)]))
                    tp.append(True)
                else:
                    scores.append(float(s[int(pred_index)]))
                    tp.append(False)
        ap, precision, recall = compute_ap(scores, tp, num_gt)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class[int(cls)] = {
            "ap": ap, "precision": precision, "recall": recall, "f1": f1, "num_gt": num_gt,
        }
    map_value = float(np.mean([v["ap"] for v in per_class.values()])) if per_class else 0.0
    return per_class, map_value


def decode_detections(
    raw_outputs: Sequence[np.ndarray],
    layout: str,
    num_classes: int,
    conf_thres: float,
    iou_thres: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把检测原始输出解码并 NMS 为 ``(boxes_xyxy, scores, labels)``。"""
    layout = (layout or "bnc").lower()
    if layout == "bnc6":
        if not raw_outputs:
            return _empty()
        arr = np.asarray(raw_outputs[0], np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[1] != 6:
            return _empty()
        mask = arr[:, 4] >= conf_thres
        rows = arr[mask]
        boxes = rows[:, :4]
        scores = rows[:, 4]
        labels = rows[:, 5]
    elif layout == "bnc" or layout == "bcn":
        arr = np.asarray(raw_outputs[0], np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            return _empty()
        if layout == "bcn" and arr.shape[0] < arr.shape[1]:
            arr = arr.T
        if arr.shape[1] < 5:
            return _empty()
        columns = arr.shape[1]
        has_obj = columns == 5 + num_classes if num_classes else (columns >= 6)
        if has_obj:
            obj = arr[:, 4]
            cls_scores = arr[:, 5:5 + num_classes] if num_classes else arr[:, 5:]
        else:
            obj = np.ones(arr.shape[0], np.float32)
            cls_scores = arr[:, 4:4 + num_classes] if num_classes else arr[:, 4:]
        if cls_scores.shape[1] == 0:
            return _empty()
        labels = np.argmax(cls_scores, axis=1).astype(np.float32)
        scores = obj * cls_scores[np.arange(cls_scores.shape[0]), labels]
        cx, cy, w, h = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        keep = scores >= conf_thres
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    elif layout == "pair":
        if len(raw_outputs) < 2:
            return _empty()
        boxes = np.asarray(raw_outputs[0], np.float32)
        s_raw = np.asarray(raw_outputs[1], np.float32)
        if boxes.ndim == 3:
            boxes = boxes[0]
        if s_raw.ndim == 3:
            s_raw = s_raw[0]
        if s_raw.ndim == 2 and s_raw.shape[1] == 1:
            s_raw = s_raw[:, 0]
        if s_raw.ndim == 2:
            labels = np.argmax(s_raw, axis=1).astype(np.float32)
            scores = s_raw[np.arange(s_raw.shape[0]), labels]
        else:
            scores = np.asarray(s_raw, np.float32)
            labels = np.zeros(scores.shape[0], np.float32)
        keep = scores >= conf_thres
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    else:  # 非检测布局在 DET_LAYOUTS 之外 → 空(上层据此判 unavailable)
        return _empty()

    if boxes.size == 0:
        return _empty()
    picked = nms(boxes, scores, iou_thres)
    return boxes[picked], scores[picked], labels[picked]


def _empty():
    return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32)


def detection_evaluator(
    contract: Optional[Dict[str, Any]] = None,
    ground_truth: Optional[Sequence[Dict[str, Any]]] = None,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    num_classes: Optional[int] = None,
    decode_fn: Optional[Callable] = None,
) -> ApplicationEvaluatorLike:
    """构造适配 :class:`AccuracyValidator.validate(application_evaluator=...)` 的检测评测器。

    ``ground_truth`` 与样本一一对应; 缺 GT 或非检测布局时诚实返回 ``available=false``
    (绝不伪造 mAP)。``decode_fn`` 可自定义"raw 输出 → (boxes, scores, labels)"。
    """
    gt_list = list(ground_truth or [])
    contract = dict(contract or {})
    layout = str(contract.get("layout") or "bnc").lower()

    def evaluate(reference, target, samples, num_classes_):
        del reference  # 检测评测只以 target 输出与 GT 对比, 不使用参考会话
        nc = int(num_classes_ or num_classes or contract.get("num_classes") or 80)
        if not gt_list:
            return {"available": False, "reason": "未提供带标注 GT, 无法计算应用级 mAP"}
        if layout not in DET_LAYOUTS:
            return {"available": False, "reason": "任务 {} 非检测, 本评测器只支持 detection "
                                                 "(mAP)".format(layout)}
        if len(gt_list) != len(samples):
            return {"available": False, "reason": "GT 数 {} 与样本数 {} 不一致".format(
                len(gt_list), len(samples))}

        records: List[Dict[str, Any]] = []
        try:
            for sample, gt in zip(samples, gt_list):
                feeds = {k: np.asarray(v, np.float32) for k, v in sample.items()}
                outputs = target.run(feeds)
                if decode_fn is not None:
                    pred = decode_fn(outputs)
                else:
                    pred = decode_detections(outputs, layout, nc, conf_thres, iou_thres)
                box_p, score_p, label_p = pred
                records.append({
                    "pred": box_p, "scores": score_p, "labels": label_p.astype(np.int64),
                    "gt": np.asarray(gt["gt"], np.float32),
                    "gt_labels": np.asarray(gt["gt_labels"], np.int64),
                })
        except Exception as err:
            return {"available": False, "reason": "应用级评测执行失败: {}".format(err)}

        _, map50 = eval_detections(records, iou_thres=0.50)
        # mAP @ .5:.95
        aps = [eval_detections(records, iou_thres=(0.50 + 0.05 * step))[1]
               for step in range(10)]
        ap5, _ = eval_detections(records, iou_thres=0.50)
        pr = [_map_final(records)]
        return {
            "available": True,
            "map": float(np.mean(aps)),
            "map50": float(map50),
            "ap": {str(k): round(v["ap"], 4) for k, v in ap5.items()},
            "precision": round(pr[0]["precision"], 4),
            "recall": round(pr[0]["recall"], 4),
            "f1": round(pr[0]["f1"], 4),
            "layout": layout,
            "passing_threshold": 0.0,
            "note": "mAP@0.5:.95 / AP 均为逐类平均; 数值不同步容忍由业务侧设定",
        }

    return evaluate


def _map_final(records):
    """在 IoU@0.5 下给出总体的 precision/recall/F1(合并所有类别)。"""
    per_class, _ = eval_detections(records, iou_thres=0.5)
    precisions = [v["precision"] for v in per_class.values()]
    recalls = [v["recall"] for v in per_class.values()]
    if not precisions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    p = float(np.mean(precisions))
    r = float(np.mean(recalls))
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}