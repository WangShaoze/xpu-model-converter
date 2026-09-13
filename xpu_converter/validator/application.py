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


# ------------------------------------------------------------------ 多任务应用级指标
# ChatGPT 意见 P0-9 / §"多任务 Application Validator": 检测之外,分类用 top-1/top-k 精度、
# 姿态用 OKS(Object Keypoint Similarity)。缺 GT 或无法解码时**诚实返回 available=false**,
# 绝不伪造指标(与 §20 一致)。输出与 runtime/detection/nwai_decoders.py 的解码语义对齐。
LAYOUT_CLS = "cls"
LAYOUT_POSE = "pose"


def _task_array(array: Sequence) -> Optional[np.ndarray]:
    """归一化原始输出为 ``(N, C)`` 2D 行优先数组; 无法归一化返回 None。"""
    arr = np.asarray(array, np.float32)
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] == 1 else arr
    if arr.ndim == 4:  # dense (1,H,W[,C]) 之类的多余批维统一切掉
        arr = arr[0]
    if arr.ndim != 2:
        return None
    # bcn (C,N) → (N,C)
    return arr.T if arr.shape[0] < arr.shape[1] else arr


def decode_cls(
    raw_outputs: Sequence[np.ndarray],
    num_classes: Optional[int] = None,
    topk: int = 5,
) -> Optional[Tuple[int, np.ndarray, np.ndarray]]:
    """解码分类输出 → ``(top1_label, topk_labels, probs)``。

    分类对分数的单调变换(argmax/top-k 排序)不变, 因此 softmax 与否不影响判定。
    """
    if not raw_outputs:
        return None
    del num_classes  # 分类输出长度即类别数, 无需外部指定
    array = np.asarray(raw_outputs[0], np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    array = array.reshape(-1)
    if array.size == 0:
        return None
    order = np.argsort(-array)
    topk_labels = (order[:max(1, int(topk))]).astype(np.int64)
    shifted = array - float(array.max())
    probs = np.exp(shifted)
    probs = probs / probs.sum()
    return int(order[0]), topk_labels, probs


def cls_evaluator(
    contract: Optional[Dict[str, Any]] = None,
    ground_truth: Optional[Sequence[Dict[str, Any]]] = None,
    topk: int = 5,
) -> ApplicationEvaluatorLike:
    """分类应用级评测: top-1 / top-k 精度。

    每样本 GT 为 ``{"gt_label": int}``(或 ``{"label": int}``); 缺 GT / 非 cls 布局
    / 无法解码时诚实返回 ``available=false``。
    """
    contract = dict(contract or {})
    layout = str(contract.get("layout") or LAYOUT_CLS).lower()
    gt_list = list(ground_truth or [])

    def evaluate(reference, target, samples, num_classes_):
        del reference  # 只以 target 输出对比 GT
        if not gt_list:
            return {"available": False, "reason": "未提供带标注 GT, 无法计算分类精度"}
        if layout != LAYOUT_CLS:
            return {"available": False, "reason": "任务 {} 非 classification".format(layout)}
        if len(gt_list) != len(samples):
            return {"available": False, "reason": "GT 数 {} 与样本数 {} 不一致".format(
                len(gt_list), len(samples))}
        top1_num = topk_num = total = 0
        try:
            for sample, gt in zip(samples, gt_list):
                feeds = {k: np.asarray(v, np.float32) for k, v in sample.items()}
                decoded = decode_cls(target.run(feeds), num_classes_, topk)
                if decoded is None:
                    continue
                top1_label, topk_labels, _ = decoded
                gt_label = int(gt.get("gt_label", gt.get("label", -1)))
                total += 1
                if gt_label == top1_label:
                    top1_num += 1
                if gt_label in topk_labels.tolist():
                    topk_num += 1
        except Exception as err:
            return {"available": False, "reason": "分类评测执行失败: {}".format(err)}
        if total == 0:
            return {"available": False, "reason": "无有效样本可评估"}
        return {
            "available": True,
            "accuracy": round(top1_num / total, 4),
            "top{}_accuracy".format(topk): round(topk_num / total, 4),
            "samples": total,
            "layout": layout,
            "passing_threshold": 0.0,
        }

    return evaluate


def _keypoint_scale(keypoints: np.ndarray) -> float:
    """由可见关键点外接框对角估计对象尺度, 供 OKS 归一化(对应 bbox 面积的开方)。"""
    kp = np.asarray(keypoints, np.float32)
    vis = kp[:, 2] > 0 if kp.shape[1] >= 3 else np.ones(kp.shape[0], bool)
    if not np.any(vis):
        return 1.0
    pts = kp[vis, :2]
    width = float(pts[:, 0].max() - pts[:, 0].min())
    height = float(pts[:, 1].max() - pts[:, 1].min())
    return (float(np.sqrt(width * height)) if width > 0 and height > 0 else 1.0)


def decode_pose(
    raw_outputs: Sequence[np.ndarray],
    num_classes: Optional[int] = None,
    has_objectness: Optional[bool] = None,
) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
    """解码姿态头输出 → ``(box_xyxy, score, keypoints (K,3))``(取置信度最高候选)。

    兼容 bcn ``(B,4+nc+K*3,N)`` / bnc ``(B,N,4+nc+K*3)``; 关键点列语义对齐
    runtime/detection/nwai_decoders.PoseDecoder(x,y,conf)。
    """
    if not raw_outputs:
        return None
    arr = _task_array(raw_outputs[0])
    if arr is None:
        return None
    columns = arr.shape[1]
    nc = int(num_classes or 0)
    if nc:
        if (columns - (nc + 4)) % 3 == 0:
            box_cols, has_obj = nc + 4, False
        else:
            box_cols, has_obj = nc + 5, True
    else:
        box_cols = max(4, columns - (columns - 5) % 3)
        has_obj = bool(has_objectness)
    if box_cols < 4 or columns < box_cols + 3:
        return None
    box_part = arr[:, :box_cols]
    kpt_part = arr[:, box_cols:]
    nc_eff = nc if nc else max(1, box_part.shape[1] - (5 if has_obj else 4))
    if has_obj:
        obj = box_part[:, 4]
        cls_scores = box_part[:, 5:5 + nc_eff]
    else:
        obj = np.ones(box_part.shape[0], np.float32)
        cls_scores = box_part[:, 4:4 + nc_eff]
    if cls_scores.shape[1] == 0:
        return None
    scores = obj * cls_scores.max(axis=1)
    best = int(np.argmax(scores))
    cx, cy, width, height = (box_part[best, 0], box_part[best, 1],
                             box_part[best, 2], box_part[best, 3])
    box = np.array([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2], np.float32)
    kpt_flat = kpt_part[best]
    num_kpt = kpt_flat.size // 3
    if num_kpt < 1:
        return None
    keypoints = kpt_flat[:num_kpt * 3].reshape(num_kpt, 3)
    return box, float(scores[best]), keypoints


def compute_oks(
    pred_keypoints: np.ndarray,
    gt_keypoints: np.ndarray,
    gt_scale: Optional[float] = None,
    sigmas: Optional[Sequence[float]] = None,
) -> float:
    """单对象 OKS: ``mean_i exp(-d_i^2 / (2 s^2 sigma_i^2)) * v_i``, 按可见度加权平均。"""
    pred = np.asarray(pred_keypoints, np.float32)
    gt = np.asarray(gt_keypoints, np.float32)
    if pred.shape != gt.shape or pred.size == 0:
        return 0.0
    k = pred.shape[0]
    vis = gt[:, 2] > 0 if gt.shape[1] >= 3 else np.ones(k, bool)
    if not np.any(vis):
        return 0.0
    sigma = 0.05 if sigmas is None else float(np.mean(sigmas))
    scale = float(gt_scale) if gt_scale else _keypoint_scale(gt)
    d2 = np.sum((pred[:, :2] - gt[:, :2]) ** 2, axis=1)
    s2 = max(scale * scale, 1e-6)
    oks_i = np.where(vis, np.exp(-d2 / (2.0 * s2 * (sigma * sigma))), 0.0)
    return float(oks_i.sum() / vis.sum())


def pose_evaluator(
    contract: Optional[Dict[str, Any]] = None,
    ground_truth: Optional[Sequence[Dict[str, Any]]] = None,
    sigma: Optional[float] = None,
) -> ApplicationEvaluatorLike:
    """姿态应用级评测: 平均 OKS(Object Keypoint Similarity)。

    每样本 GT 为 ``{"gt_keypoints": (K,3)[x,y,vis], "gt_scale": float(可选)}``;
    缺 GT / 非 pose 布局 / 无法解码时诚实返回 ``available=false``。
    """
    contract = dict(contract or {})
    layout = str(contract.get("layout") or LAYOUT_POSE).lower()
    gt_list = list(ground_truth or [])

    def evaluate(reference, target, samples, num_classes_):
        del reference
        if not gt_list:
            return {"available": False, "reason": "未提供带标注 GT(关键点), 无法计算 OKS"}
        if layout != LAYOUT_POSE:
            return {"available": False, "reason": "任务 {} 非 pose".format(layout)}
        if len(gt_list) != len(samples):
            return {"available": False, "reason": "GT 数 {} 与样本数 {} 不一致".format(
                len(gt_list), len(samples))}
        oks_list: List[float] = []
        try:
            for sample, gt in zip(samples, gt_list):
                feeds = {k: np.asarray(v, np.float32) for k, v in sample.items()}
                decoded = decode_pose(target.run(feeds), num_classes_,
                                      contract.get("has_objectness"))
                if decoded is None:
                    continue
                _, _, pred_kpts = decoded
                gt_kpts = np.asarray(gt["gt_keypoints"], np.float32)
                scale = gt.get("gt_scale") or _keypoint_scale(gt_kpts)
                oks_list.append(compute_oks(pred_kpts, gt_kpts, scale, sigma))
        except Exception as err:
            return {"available": False, "reason": "姿态评测执行失败: {}".format(err)}
        if not oks_list:
            return {"available": False, "reason": "无有效样本可评估"}
        return {
            "available": True,
            "oks": round(float(np.mean(oks_list)), 4),
            "samples": len(oks_list),
            "layout": layout,
            "passing_threshold": 0.0,
        }

    return evaluate


def build_evaluator(
    task: str,
    contract: Optional[Dict[str, Any]] = None,
    ground_truth: Optional[Sequence[Dict[str, Any]]] = None,
) -> ApplicationEvaluatorLike:
    """按任务分派应用级评测器(ChatGPT 意见 P0-9 多任务 Application Validator)。

    * detection → mAP(mAP50 / mAP@0.5:.95);
    * classification → top-1 / top-k 精度;
    * pose → OKS;
    * 其余(obb/seg/depth/sem)当前无标准标注评估器, 返回诚实 ``available=false``。
    """
    task = str(task or "detection").lower()
    if task == "classification" or task == "cls":
        return cls_evaluator(contract=contract, ground_truth=ground_truth)
    if task == "pose":
        return pose_ap_evaluator(contract=contract, ground_truth=ground_truth)
    if task == "segment":
        return segment_evaluator(contract=contract, ground_truth=ground_truth)
    if task == "obb":
        return obb_evaluator(contract=contract, ground_truth=ground_truth)
    if task == "depth":
        return depth_evaluator(contract=contract, ground_truth=ground_truth)
    if task == "sem":
        return sem_evaluator(contract=contract, ground_truth=ground_truth)
    # 未知任务 → 落到 detection mAP(仅当其契约真是检测布局时才 available)
    return detection_evaluator(contract=contract, ground_truth=ground_truth)


# ------------------------------------------------------------------ 分割 / 旋转框 / 深度 / 语义 / 姿态 AP
# ChatGPT 意见 P0-2~P0-6: Segmentation(mask IoU/AP)、OBB(rotated IoU/mAP)、
# Depth(AbsRel/SqRel/RMSE/δ)、Semantic(pixel acc/mIoU)、Pose AP(OKS matching)。
# 一律纯 numpy; 缺 GT 时诚实返回 available=false(绝不伪造指标, 与 §20 一致)。
LAYOUT_SEG = "segment"
LAYOUT_OBB = "obb"
LAYOUT_DENSE = "dense"
LAYOUT_SEM = "sem"


def _sigmoid(x):
    x = np.clip(np.asarray(x, np.float32), -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def _upsample_nearest(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """最近邻上采样二维图。"""
    h, w = mask.shape
    if (h, w) == (out_h, out_w):
        return mask
    r = (np.arange(out_h) * h) // max(out_h, 1)
    c = (np.arange(out_w) * w) // max(out_w, 1)
    return mask[r][:, c]


def _filter_rows(arr, num_classes, has_objectness=False):
    """检测头列 ``(N, 4|5+nc)`` 置信度过滤, 返回与行对齐的 ``(boxes,scores,class_ids,bool_mask)``。"""
    if arr.ndim != 2 or arr.shape[1] < 5:
        return None
    columns = arr.shape[1]
    if num_classes:
        has_obj = columns == 5 + num_classes
        nc = num_classes
    else:
        has_obj = bool(has_objectness)
        nc = max(1, columns - (5 if has_obj else 4))
    if has_obj:
        obj = arr[:, 4]
        cls_scores = arr[:, 5:5 + nc]
    else:
        obj = np.ones(arr.shape[0], np.float32)
        cls_scores = arr[:, 4:4 + nc]
    if cls_scores.shape[1] == 0:
        return None
    class_ids = np.argmax(cls_scores, axis=1)
    scores = obj * cls_scores[np.arange(cls_scores.shape[0]), class_ids]
    keep = scores >= 0.0
    return arr[:, :4], scores, class_ids.astype(np.float32), keep


def decode_segment(
    raw_outputs: Sequence[np.ndarray],
    num_classes: int,
    out_h: int,
    out_w: int,
    num_masks: int = 32,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> Optional[Tuple[np.ndarray, float, np.ndarray, np.ndarray]]:
    """解码分割输出 → ``(box_xyxy, score, class_id, binary_mask(H,W))``(取置信度最高候选)。

    ``outputs[0]`` 为 det 分支(含 mask 系数列), ``outputs[1]`` 为 proto ``(nm,Hp,Wp)``;
    mask = sigmoid(proto @ coeffs) 后上采样到 ``(out_h, out_w)`` 再阈值化为二值掩码。
    """
    if not raw_outputs or len(raw_outputs) < 2:
        return None
    arr = _task_array(raw_outputs[0])
    proto = np.asarray(raw_outputs[1], np.float32)
    if proto.ndim == 4:  # 去掉 batch 维: (1,nm,Hp,Wp) → (nm,Hp,Wp)
        proto = proto[0]
    # 注意: 不可再剥掉首维 —— (nm,Hp,Wp) 的 shape[0] 即掩码数 nm(nm==1 时剥掉会
    # 变成 2D 而被下方 ndim!=3 拒绝), 保持 3D 以兼容 tensordot(coeffs, proto)。
    if arr is None or arr.shape[1] < 5 or proto.ndim != 3:
        return None
    columns = arr.shape[1]
    box_cols = num_classes + 4 if num_classes else columns - num_masks
    if box_cols < 5:
        box_cols = max(5, columns - num_masks)
    box_part = arr[:, :box_cols]
    coeff_part = arr[:, box_cols:box_cols + num_masks]
    filtered = _filter_rows(box_part, num_classes, False)
    if filtered is None or not np.any(filtered[3]):
        return None
    centers = filtered[0][filtered[3]]
    scores = filtered[1][filtered[3]]
    class_ids = filtered[2][filtered[3]]
    coeffs = coeff_part[filtered[3]]
    if centers.size == 0 or coeffs.shape[0] == 0:
        return None
    keep = nms(_to_xyxy_(centers), scores, iou_thres)
    keep = keep[keep.size // 2:] if keep.size > 1 else keep  # 取高置信度一侧(单候选场景)
    best = int(keep[0]) if keep.size else 0
    cx, cy, w, h = centers[best]
    box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], np.float32)
    mask_raw = np.tensordot(coeffs[best], proto, axes=([0], [0]))  # (Hp,Wp)
    mask_bin = _sigmoid(mask_raw)
    mask_bin = _upsample_nearest(mask_bin, out_h, out_w)
    return box, float(scores[best]), float(class_ids[best]), (mask_bin >= 0.5).astype(np.float32)


def _to_xyxy_(centers):
    return np.stack([centers[:, 0] - centers[:, 2] / 2, centers[:, 1] - centers[:, 3] / 2,
                     centers[:, 0] + centers[:, 2] / 2, centers[:, 1] + centers[:, 3] / 2], axis=1)


def mask_iou_mask(pred: np.ndarray, gt: np.ndarray) -> float:
    """两张二值掩码的 IoU。"""
    inter = float(np.logical_and(pred > 0, gt > 0).sum())
    union = float(np.logical_or(pred > 0, gt > 0).sum())
    return inter / union if union > 0 else (1.0 if inter > 0 else 0.0)


def segment_evaluator(
    contract: Optional[Dict[str, Any]] = None,
    ground_truth: Optional[Sequence[Dict[str, Any]]] = None,
    num_masks: int = 32,
):
    """分割应用级评测: 单对象 mask IoU / mAP50 / mAP50:95(GPT 意见 P0-2)。

    每样本 GT 为 ``{"gt_mask": (H,W) 二值, "gt_bbox": (4,) xyxy(可选)}``;
    缺 GT / 非 segment 布局 / 无法解码时诚实返回 ``available=false``。
    """
    contract = dict(contract or {})
    layout = str(contract.get("layout") or LAYOUT_SEG).lower()
    gt_list = list(ground_truth or [])

    def evaluate(reference, target, samples, num_classes_):
        del reference
        if not gt_list:
            return {"available": False, "reason": "未提供带标注 GT(掩码), 无法计算 mask IoU/AP"}
        if layout != LAYOUT_SEG:
            return {"available": False, "reason": "任务 {} 非 segmentation".format(layout)}
        if len(gt_list) != len(samples):
            return {"available": False, "reason": "GT 数 {} 与样本数 {} 不一致".format(
                len(gt_list), len(samples))}
        records = []
        nc = int(num_classes_ or contract.get("num_classes") or 80)
        try:
            for sample, gt in zip(samples, gt_list):
                gt_mask = np.asarray(gt["gt_mask"], np.float32)
                out_h, out_w = gt_mask.shape[:2]
                feeds = {k: np.asarray(v, np.float32) for k, v in sample.items()}
                decoded = decode_segment(target.run(feeds), nc, out_h, out_w, num_masks)
                if decoded is None:
                    continue
                box, score, label, pred_mask = decoded
                records.append({
                    "pred": box[None, :], "scores": np.array([score], np.float32),
                    "labels": np.array([label], np.int64),
                    "pred_mask": pred_mask, "gt_mask": gt_mask,
                    "gt": np.zeros((0, 4), np.float32), "gt_labels": np.zeros(0, np.int64),
                })
        except Exception as err:
            return {"available": False, "reason": "分割评测执行失败: {}".format(err)}
        if not records:
            return {"available": False, "reason": "无有效样本可评估"}
        # 用 mask IoU 作为匹配相似度, 累积 mask AP
        aps = []
        for thr in [0.50 + 0.05 * i for i in range(10)]:
            scores, tp, num_gt = [], [], 0
            for rec in records:
                num_gt += 1
                if rec["pred_mask"] is None:
                    continue
                scores.append(float(rec["scores"][0]))
                scored = mask_iou_mask(rec["pred_mask"], rec["gt_mask"])
                tp.append(scored >= thr)
            ap, _, _ = compute_ap(scores, tp, num_gt)
            aps.append(ap)
        map50 = aps[0]
        return {"available": True, "map": round(float(np.mean(aps)), 4),
                "map50": round(map50, 4), "mask_iou": round(
                    float(np.mean([mask_iou_mask(r["pred_mask"], r["gt_mask"])
                                   for r in records if r["pred_mask"] is not None]) or 0.0), 4),
                "samples": len(records), "layout": layout, "passing_threshold": 0.0}
    return evaluate


# ------------------------------------------------------------------ OBB(旋转框)
def _rbox_corners(rbox: np.ndarray) -> np.ndarray:
    """rbox ``(...,5)=[cx,cy,w,h,angle(rad)]`` → ``(...,4,2)`` 旋转矩形角点。"""
    rbox = np.asarray(rbox, np.float32)
    cx, cy, w, h, ang = (rbox[..., 0], rbox[..., 1], rbox[..., 2], rbox[..., 3], rbox[..., 4])
    cos, sin = np.cos(ang), np.sin(ang)
    hw, hh = w / 2, h / 2
    dx = np.stack([-hw, hw, hw, -hw], axis=-1)
    dy = np.stack([-hh, -hh, hh, hh], axis=-1)
    x = cx[..., None] + dx * cos[..., None] - dy * sin[..., None]
    y = cy[..., None] + dx * sin[..., None] + dy * cos[..., None]
    return np.stack([x, y], axis=-1)


def _poly_area(poly: np.ndarray) -> float:
    xs = np.asarray(poly, np.float64)
    return abs(0.5 * float(np.dot(xs[:-1, 0], xs[1:, 1]) - np.dot(xs[:-1, 1], xs[1:, 0]))) \
        if xs.shape[0] >= 3 else 0.0


def _clip_poly(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland–Hodgman 凸多边形裁剪(``subject`` 与 ``clip`` 均为 (n,2) 顺时针角点)。"""
    sub = [tuple(p) for p in subject]
    for i in range(len(clip)):
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        edge = (b[0] - a[0], b[1] - a[1])
        def _side(p):
            # 对 CCW 多边形, 平面左侧(内侧)判定为 edge×(p-a)>=0
            return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])
        cur = sub
        sub = []
        for j in range(len(cur)):
            cur_p = cur[j]
            prev_p = cur[j - 1]
            if _side(cur_p) >= 0:
                if _side(prev_p) < 0:
                    sub.append(_line_intersect(prev_p, cur_p, a, b))
                sub.append(cur_p)
            elif _side(prev_p) >= 0:
                sub.append(_line_intersect(prev_p, cur_p, a, b))
    return np.asarray(sub, np.float64)


def _line_intersect(p1, p2, a, b):
    """线段 p1-p2 与裁剪边 a-b(无限长直线)的交点。"""
    p1, p2, a, b = (np.asarray(p1, np.float64), np.asarray(p2, np.float64),
                    np.asarray(a, np.float64), np.asarray(b, np.float64))
    d = p2 - p1
    denom = d[0] * (b[1] - a[1]) - d[1] * (b[0] - a[0])
    if abs(denom) < 1e-12:
        return p2
    t = ((a[0] - p1[0]) * (b[1] - a[1]) - (a[1] - p1[1]) * (b[0] - a[0])) / denom
    return p1 + t * d


def _rbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    corners_a = _rbox_corners(a)  # (4,2) 已含全部角点, 不再取 [0]
    corners_b = _rbox_corners(b)
    inter = _poly_area(_clip_poly(corners_a, corners_b))
    union = float(a[2] * a[3] + b[2] * b[3] - inter)
    if union <= 1e-12:  # 完全重合(或退化): 面积并入 inter 使 union 归零
        return 1.0 if inter > 0 else 0.0
    return float(inter / union)


def decode_obb(
    raw_outputs: Sequence[np.ndarray],
    num_classes: int,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """解码旋转框输出 → ``(rbox(N,5), scores, labels, angles)``(conf 过滤+NMS 后)。"""
    if not raw_outputs:
        return None
    arr = _task_array(raw_outputs[0])
    if arr is None or arr.shape[1] < 6:
        return None
    columns = arr.shape[1]
    nc = num_classes if (columns == 5 + num_classes) else (columns - 5 if columns > 5 else 0)
    if nc <= 0:
        return None
    angle = arr[:, 4]
    cls_box = np.concatenate([arr[:, :4], arr[:, 5:5 + nc]], axis=1)
    filtered = _filter_rows(cls_box, nc, False)
    if filtered is None or not np.any(filtered[3]):
        return None
    centers, scores, class_ids, keep = filtered
    mask = keep
    if not np.any(mask):
        return None
    rbox = np.concatenate([centers[mask], angle[mask][:, None]], axis=1).astype(np.float32)
    # 单类 NMS 用近似: 用旋转 IoU 做贪心去重
    order = np.argsort(-scores[mask])
    keep_ids = []
    picked = []
    for idx in order:
        if all(_rbox_iou(rbox[idx], rbox[j]) < iou_thres for j in picked):
            keep_ids.append(idx)
            picked.append(idx)
    keep_ids = np.asarray(keep_ids, np.int64)
    return rbox[keep_ids], scores[mask][keep_ids], class_ids[mask][keep_ids], angle[mask][keep_ids]


def obb_evaluator(
    contract: Optional[Dict[str, Any]] = None,
    ground_truth: Optional[Sequence[Dict[str, Any]]] = None,
    iou_thres: float = 0.5,
):
    """OBB 应用级评测: 旋转框 rotated-IoU 匹配的 mAP(GPT 意见 P0-3)。

    每样本 GT 为 ``{"gt_obb": (n,5)[cx,cy,w,h,angle(rad)], "gt_labels": (n,)}``;
    缺 GT / 非 obb 布局 / 无法解码时诚实返回 ``available=false``。
    """
    contract = dict(contract or {})
    layout = str(contract.get("layout") or LAYOUT_OBB).lower()
    gt_list = list(ground_truth or [])

    def evaluate(reference, target, samples, num_classes_):
        del reference
        if not gt_list:
            return {"available": False, "reason": "未提供带标注 GT(旋转框), 无法计算 OBB mAP"}
        if layout != LAYOUT_OBB:
            return {"available": False, "reason": "任务 {} 非 obb".format(layout)}
        if len(gt_list) != len(samples):
            return {"available": False, "reason": "GT 数 {} 与样本数 {} 不一致".format(
                len(gt_list), len(samples))}
        nc = int(num_classes_ or contract.get("num_classes") or 80)
        records = []
        try:
            for sample, gt in zip(samples, gt_list):
                feeds = {k: np.asarray(v, np.float32) for k, v in sample.items()}
                decoded = decode_obb(target.run(feeds), nc)
                gt_obb = np.asarray(gt["gt_obb"], np.float32)
                gt_labels = np.asarray(gt.get("gt_labels", np.zeros(gt_obb.shape[0], np.int64)), np.int64)
                if decoded is None:
                    records.append({"pred": np.zeros((0, 5), np.float32),
                                    "scores": np.zeros(0, np.float32),
                                    "labels": np.zeros(0, np.int64),
                                    "gt_obb": gt_obb, "gt_labels": gt_labels})
                    continue
                rbox, scores, labels, _ = decoded
                records.append({"pred": rbox, "scores": scores,
                                "labels": labels.astype(np.int64),
                                "gt_obb": gt_obb, "gt_labels": gt_labels})
        except Exception as err:
            return {"available": False, "reason": "OBB 评测执行失败: {}".format(err)}
        # rotated-IoU 匹配, 逐类累积 → mAP@IoU
        class_ids = set()
        for rec in records:
            class_ids.update(rec["labels"].tolist())
            class_ids.update(rec["gt_labels"].tolist())
        aps = []
        for thr in [0.50 + 0.05 * i for i in range(10)]:
            cls_aps = []
            for cls in sorted(class_ids):
                scores, tp, num_gt = [], [], 0
                for rec in records:
                    pred_c = rec["labels"] == cls
                    gt_c = rec["gt_labels"] == cls
                    num_gt += int(gt_c.sum())
                    pred = rec["pred"][pred_c]
                    s = rec["scores"][pred_c]
                    gt = rec["gt_obb"][gt_c]
                    if pred.size == 0:
                        continue
                    for idx in range(pred.shape[0]):
                        if gt.shape[0] == 0:
                            scores.append(float(s[idx])); tp.append(False); continue
                        ious = np.array([_rbox_iou(pred[idx], g) for g in gt], np.float32)
                        bi = int(np.argmax(ious))
                        if ious[bi] >= thr:
                            gt = np.delete(gt, bi, axis=0)
                            scores.append(float(s[idx])); tp.append(True)
                        else:
                            scores.append(float(s[idx])); tp.append(False)
                ap, _, _ = compute_ap(scores, tp, num_gt)
                cls_aps.append(ap)
            aps.append(float(np.mean(cls_aps)) if cls_aps else 0.0)
        map50 = aps[0]
        return {"available": True, "map": round(float(np.mean(aps)), 4),
                "map50": round(map50, 4), "layout": layout, "samples": len(records),
                "passing_threshold": 0.0}
    return evaluate


# ------------------------------------------------------------------ Depth
def decode_depth(raw_outputs: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """解码深度输出 → ``(H,W)`` 浮点深度图。"""
    if not raw_outputs:
        return None
    arr = np.asarray(raw_outputs[0], np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
    if arr.ndim != 2:
        return None
    return arr


def depth_evaluator(
    contract: Optional[Dict[str, Any]] = None,
    ground_truth: Optional[Sequence[Dict[str, Any]]] = None,
):
    """深度应用级评测: AbsRel / SqRel / RMSE / RMSE_log / δ1..δ3(GPT 意见 P0-4)。

    每样本 GT 为 ``{"gt_depth": (H,W), "gt_mask": (H,W) 有效像素掩码(可选)}``;
    缺 GT / 非 depth 布局 / 无法解码时诚实返回 ``available=false``。
    """
    contract = dict(contract or {})
    layout = str(contract.get("layout") or LAYOUT_DENSE).lower()
    gt_list = list(ground_truth or [])

    def _depth_metrics(pred, gt, mask):
        p = np.asarray(pred, np.float32)[mask > 0]
        g = np.asarray(gt, np.float32)[mask > 0]
        if p.size == 0 or g.size == 0:
            return None
        g = np.maximum(g, 1e-6)
        diff = p - g
        abs_rel = float(np.mean(np.abs(diff) / g))
        sq_rel = float(np.mean((diff ** 2) / g))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        rmse_log = float(np.sqrt(np.mean((np.log(p + 1e-6) - np.log(g)) ** 2)))
        gt_nz = g > 0
        ratio = (p / g)[gt_nz] if np.any(gt_nz) else np.ones(0, np.float32)
        pct = lambda t: float(np.mean(ratio < t)) if ratio.size else 0.0
        return {"abs_rel": round(abs_rel, 4), "sq_rel": round(sq_rel, 4),
                "rmse": round(rmse, 4), "rmse_log": round(rmse_log, 4),
                "d1": round(pct(1.25), 4), "d2": round(pct(1.25 ** 2), 4),
                "d3": round(pct(1.25 ** 3), 4)}

    def evaluate(reference, target, samples, num_classes_):
        del reference, num_classes_
        if not gt_list:
            return {"available": False, "reason": "未提供带标注 GT(深度), 无法计算深度指标"}
        if layout != LAYOUT_DENSE:
            return {"available": False, "reason": "任务 {} 非 depth/dense".format(layout)}
        if len(gt_list) != len(samples):
            return {"available": False, "reason": "GT 数 {} 与样本数 {} 不一致".format(
                len(gt_list), len(samples))}
        metrics = []
        try:
            for sample, gt in zip(samples, gt_list):
                gt_depth = np.asarray(gt["gt_depth"], np.float32)
                gt_mask = np.asarray(gt.get("gt_mask", np.ones_like(gt_depth)), np.float32)
                feeds = {k: np.asarray(v, np.float32) for k, v in sample.items()}
                pred = decode_depth(target.run(feeds))
                if pred is None:
                    continue
                m = _depth_metrics(pred, gt_depth, gt_mask)
                if m is not None:
                    metrics.append(m)
        except Exception as err:
            return {"available": False, "reason": "深度评测执行失败: {}".format(err)}
        if not metrics:
            return {"available": False, "reason": "无有效样本可评估"}
        out = {"available": True, "samples": len(metrics), "layout": layout, "passing_threshold": 0.0}
        for key in ("abs_rel", "sq_rel", "rmse", "rmse_log", "d1", "d2", "d3"):
            out[key] = round(float(np.mean([m[key] for m in metrics])), 4)
        return out
    return evaluate


# ------------------------------------------------------------------ Semantic Segmentation
def decode_sem(raw_outputs: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """解码语义分割输出 → ``(H,W)`` 类别索引图。"""
    if not raw_outputs:
        return None
    arr = np.asarray(raw_outputs[0], np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        return None
    return np.argmax(arr, axis=0).astype(np.int64)


def sem_evaluator(
    contract: Optional[Dict[str, Any]] = None,
    ground_truth: Optional[Sequence[Dict[str, Any]]] = None,
    num_classes: Optional[int] = None,
):
    """语义分割应用级评测: pixel acc / mean acc / mIoU / per-class IoU(GPT 意见 P0-5)。

    每样本 GT 为 ``{"gt_label": (H,W) 类别索引图}``; 缺 GT / 非 sem 布局 /
    无法解码时诚实返回 ``available=false``。
    """
    contract = dict(contract or {})
    layout = str(contract.get("layout") or LAYOUT_SEM).lower()
    gt_list = list(ground_truth or [])

    def evaluate(reference, target, samples, num_classes_):
        del reference
        if not gt_list:
            return {"available": False, "reason": "未提供带标注 GT(类别图), 无法计算 mIoU"}
        if layout != LAYOUT_SEM:
            return {"available": False, "reason": "任务 {} 非 semantic".format(layout)}
        if len(gt_list) != len(samples):
            return {"available": False, "reason": "GT 数 {} 与样本数 {} 不一致".format(
                len(gt_list), len(samples))}
        nc = int(num_classes_ or num_classes or contract.get("num_classes") or 1)
        conf_mat = np.zeros((nc, nc), np.int64) if nc else None
        try:
            for sample, gt in zip(samples, gt_list):
                feeds = {k: np.asarray(v, np.float32) for k, v in sample.items()}
                pred = decode_sem(target.run(feeds))
                gt_label = np.asarray(gt["gt_label"], np.int64)
                if pred is None or pred.shape != gt_label.shape:
                    continue
                p = np.clip(pred, 0, nc - 1).ravel()
                g = np.clip(gt_label, 0, nc - 1).ravel()
                conf_mat += np.bincount(g * nc + p, minlength=nc * nc).reshape(nc, nc)
        except Exception as err:
            return {"available": False, "reason": "语义分割评测执行失败: {}".format(err)}
        if conf_mat is None or conf_mat.sum() == 0:
            return {"available": False, "reason": "无有效像素可比对"}
        tp = np.diag(conf_mat)
        total = conf_mat.sum()
        pixel_acc = float(tp.sum()) / total if total else 0.0
        gt_pix = conf_mat.sum(axis=1)
        pred_pix = conf_mat.sum(axis=0)
        union = (gt_pix + pred_pix - tp).astype(np.float64)
        iou = np.where(union > 0, tp / np.maximum(union, 1e-6), 0.0)
        present = gt_pix > 0
        mean_iou = float(iou[present].mean()) if np.any(present) else 0.0
        cls_acc = np.where(gt_pix > 0, tp / np.maximum(gt_pix, 1), 0.0)
        mean_acc = float(cls_acc[present].mean()) if np.any(present) else 0.0
        return {"available": True, "pixel_accuracy": round(pixel_acc, 4),
                "mean_accuracy": round(mean_acc, 4), "miou": round(mean_iou, 4),
                "per_class_iou": [round(float(v), 4) for v in iou.tolist()],
                "samples": None, "layout": layout, "passing_threshold": 0.0}
    return evaluate


# ------------------------------------------------------------------ Pose AP(GPT 意见 P0-6)
def decode_poses(
    raw_outputs: Sequence[np.ndarray],
    num_classes: Optional[int] = None,
    has_objectness: Optional[bool] = None,
    conf_thres: float = 0.1,
    iou_thres: float = 0.5,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """解码姿态输出 → ``(keypoints(P,K,3), scores, labels, boxes)``(多对象, conf+NMS 后)。"""
    if not raw_outputs:
        return None
    arr = _task_array(raw_outputs[0])
    if arr is None:
        return None
    columns = arr.shape[1]
    nc = int(num_classes or 0)
    if nc:
        if (columns - (nc + 4)) % 3 == 0:
            box_cols, has_obj = nc + 4, False
        else:
            box_cols, has_obj = nc + 5, True
    else:
        box_cols = max(4, columns - (columns - 5) % 3)
        has_obj = bool(has_objectness)
    if box_cols < 4 or columns < box_cols + 3:
        return None
    box_part = arr[:, :box_cols]
    kpt_part = arr[:, box_cols:]
    nc_eff = nc if nc else max(1, box_part.shape[1] - (5 if has_obj else 4))
    if has_obj:
        obj = box_part[:, 4]
        cls_scores = box_part[:, 5:5 + nc_eff]
    else:
        obj = np.ones(box_part.shape[0], np.float32)
        cls_scores = box_part[:, 4:4 + nc_eff]
    if cls_scores.shape[1] == 0:
        return None
    class_ids = np.argmax(cls_scores, axis=1)
    scores = obj * cls_scores[np.arange(cls_scores.shape[0]), class_ids]
    keep = scores >= conf_thres
    if not np.any(keep):
        return None
    centers = box_part[keep, :4]
    score_k = scores[keep]
    cls_k = class_ids[keep].astype(np.float32)
    kpt_flat = kpt_part[keep]
    num_kpt = kpt_flat.shape[1] // 3
    if num_kpt < 1:
        return None
    kpts = kpt_flat[:, :num_kpt * 3].reshape(-1, num_kpt, 3)
    boxes = _to_xyxy_(centers)
    keep_nms = nms(boxes, score_k, iou_thres)
    return kpts[keep_nms], score_k[keep_nms], cls_k[keep_nms], boxes[keep_nms]


def match_pose_oks(
    pred_kpts: np.ndarray, chains,
    gt_kpts: np.ndarray, sigma,
) -> np.ndarray:
    """pred(G,K,3) x gt(P,K,3) OKS 匹配矩阵 → (G,P)。"""
    g_num = pred_kpts.shape[0]
    p_num = gt_kpts.shape[0]
    if g_num == 0 or p_num == 0:
        return np.zeros((g_num, p_num), np.float32)
    out = np.zeros((g_num, p_num), np.float32)
    for i, gp in enumerate(pred_kpts):
        for j, gtp in enumerate(gt_kpts):
            out[i, j] = compute_oks(gp, gtp, None, [sigma] if sigma else None)
    return out


def pose_ap_evaluator(
    contract: Optional[Dict[str, Any]] = None,
    ground_truth: Optional[Sequence[Dict[str, Any]]] = None,
    sigma: Optional[float] = None,
):
    """姿态应用级评测(标准): OKS matching → AP50 / AP75 / AP50:95 / AR(GPT 意见 P0-6)。

    每样本 GT 为 ``{"gt_keypoints_list": [(K,3)...], "gt_labels": [int...](可选)}``;
    也兼容单对象 ``{"gt_keypoints": (K,3)}``(视为 1 个 GT 对象)。缺 GT / 非 pose
    布局 / 无法解码时诚实返回 ``available=false``。同一预测仅与一个 GT 配对(贪心)。
    """
    contract = dict(contract or {})
    layout = str(contract.get("layout") or LAYOUT_POSE).lower()
    gt_list = list(ground_truth or [])

    def _build_records(target, samples, num_classes_):
        nc = int(num_classes_ or contract.get("num_classes") or 0)
        records = []
        for sample, gt in zip(samples, gt_list):
            if "gt_keypoints_list" in gt:
                gts = [np.asarray(p, np.float32) for p in gt["gt_keypoints_list"]]
                gt_labels = np.asarray(gt.get("gt_labels", np.zeros(len(gts), np.int64)), np.int64)
            elif "gt_keypoints" in gt:
                gts = [np.asarray(gt["gt_keypoints"], np.float32)]
                gt_labels = np.asarray(gt.get("gt_labels", np.zeros(1, np.int64)), np.int64)
            else:
                gts, gt_labels = [], np.zeros(0, np.int64)
            feeds = {k: np.asarray(v, np.float32) for k, v in sample.items()}
            decoded = decode_poses(target.run(feeds), nc,
                                   contract.get("has_objectness"), conf_thres=0.05)
            if decoded is None:
                kpts = np.zeros((0, 1, 3), np.float32)
                scores = np.zeros(0, np.float32)
                labels = np.zeros(0, np.int64)
            else:
                kpts, scores, labels, _ = decoded
            records.append({"pred": kpts, "scores": np.asarray(scores, np.float32),
                            "labels": np.asarray(labels, np.int64),
                            "gt": gts, "gt_labels": gt_labels})
        return records

    def _ap_at_oks(records, thr):
        classes = set()
        for rec in records:
            classes.update(rec["labels"].tolist())
            classes.update(rec["gt_labels"].tolist())
        cls_aps = []
        for cls in sorted(classes):
            scores, tp, num_gt = [], [], 0
            for rec in records:
                gts = [g for g, lab in zip(rec["gt"], rec["gt_labels"]) if int(lab) == cls]
                pred_idx = [i for i, lab in enumerate(rec["labels"]) if int(lab) == cls]
                num_gt += len(gts)
                for i in sorted(pred_idx, key=lambda k: -float(rec["scores"][k])):
                    if not gts:
                        scores.append(float(rec["scores"][i])); tp.append(False); continue
                    matches = [compute_oks(rec["pred"][i], g, None, [sigma] if sigma else None)
                               for g in gts]
                    bi = int(np.argmax(matches))
                    if matches[bi] >= thr:
                        del gts[bi]
                        scores.append(float(rec["scores"][i])); tp.append(True)
                    else:
                        scores.append(float(rec["scores"][i])); tp.append(False)
            ap, _, _ = compute_ap(scores, tp, num_gt)
            cls_aps.append(ap)
        return float(np.mean(cls_aps)) if cls_aps else 0.0

    def _compute_ar(records, thr):
        tp_sum = 0
        gt_sum = sum(len(rec["gt"]) for rec in records)
        for rec in records:
            gts = list(rec["gt"])
            for i in range(rec["pred"].shape[0]):
                if not gts:
                    break
                matches = [compute_oks(rec["pred"][i], g, None, [sigma] if sigma else None)
                           for g in gts]
                if max(matches) >= thr:
                    tp_sum += 1
                    del gts[int(np.argmax(matches))]
        return float(tp_sum) / gt_sum if gt_sum > 0 else 0.0

    def evaluate(reference, target, samples, num_classes_):
        del reference
        if not gt_list:
            return {"available": False, "reason": "未提供带标注 GT(关键点), 无法计算 Pose AP"}
        if layout != LAYOUT_POSE:
            return {"available": False, "reason": "任务 {} 非 pose".format(layout)}
        if len(gt_list) != len(samples):
            return {"available": False, "reason": "GT 数 {} 与样本数 {} 不一致".format(
                len(gt_list), len(samples))}
        try:
            records = _build_records(target, samples, num_classes_)
        except Exception as err:
            return {"available": False, "reason": "姿态评测执行失败: {}".format(err)}
        only_empty = all(rec["gt"] == [] for rec in records) or \
            all(rec["pred"].shape[0] == 0 for rec in records)
        if only_empty:
            return {"available": False, "reason": "无法从样本解码出姿态输出"}
        thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]  # COCO AP50:95
        ap_at = {t: _ap_at_oks(records, t) for t in thresholds}
        ap50 = ap_at[0.5]
        ap75 = ap_at[0.75]
        ap_mean = float(np.mean(list(ap_at.values())))
        ar = float(np.mean([_compute_ar(records, t) for t in thresholds]))
        return {"available": True, "ap50": round(ap50, 4), "ap75": round(ap75, 4),
                "ap": round(ap_mean, 4), "ar": round(ar, 4),
                "layout": layout, "passing_threshold": 0.0}
    return evaluate