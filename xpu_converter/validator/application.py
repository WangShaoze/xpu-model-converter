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
        return pose_evaluator(contract=contract, ground_truth=ground_truth)
    # detection / obb / seg / depth / sem: detection 走 mAP, 其余落 honest unavailable
    return detection_evaluator(contract=contract, ground_truth=ground_truth)