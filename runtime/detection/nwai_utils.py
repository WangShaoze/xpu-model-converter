# -*- coding: utf-8 -*-
"""识别结果后处理与置信度控制。

输出格式与客户 v1.4 规范一致:
``data=[{title, code, left, top, right, bottom, confidence}, ...]``
类别、导出编号与阈值全部来自交付包算法目录下 ``confidence.json``, 与模型代码解耦。
"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import nwai_config
from nwai_tools import ExceptionMessage

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
_RESULT_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


class NwaiUtils:
    """置信度表与结果格式化工具。"""

    def __init__(self) -> None:
        self.confidence_list: List[Dict[str, Any]] = []
        self.confidence_path = nwai_config.confidence_path()

    # ------------------------------------------------------------------ 置信度表
    def load_confidence(self) -> Tuple[bool, str]:
        ret, err_message = True, "normal"
        if not os.path.exists(self.confidence_path):
            return False, "文件confidence.json:路径{}不存在.请联系对应算法开发人员排查.".format(
                self.confidence_path)
        try:
            with open(self.confidence_path, "r", encoding="utf-8") as fr:
                self.confidence_list = json.load(fr) or []
        except Exception as err:
            return False, "文件confidence.json:路径{}无法正常解析({}).请联系对应算法开发人员排查该文件内容.".format(
                self.confidence_path, ExceptionMessage(err))
        if not len(self.confidence_list):
            return False, "文件confidence.json:路径{}解析后为空列表.请联系对应算法开发人员排查该文件内容.".format(
                self.confidence_path)
        return ret, err_message

    def get_confidence(self) -> List[Dict[str, Any]]:
        return self.confidence_list

    def is_modelid_export(self, model_id: int) -> Tuple[bool, str, str, float]:
        """查询某类别是否导出, 返回 ``(是否导出, 业务编码, 名称, 阈值)``。"""
        for item in self.get_confidence():
            if int(item.get("model_id", -1)) == int(model_id):
                if int(item.get("is_export", 0)) == 1:
                    return True, str(item.get("export_id", "")), str(item.get("name", "")), \
                        float(item.get("confidence", 0.0))
                return False, "", "", 0.0
        return False, "", "", 0.0

    # ------------------------------------------------------------------ 结果格式
    def dest_to_outputformat(self, dests):
        """把检测结果列表转成客户规范结构。"""
        if dests is None:
            return None
        if isinstance(dests, int):
            return dests

        ret, err_message = self.load_confidence()
        if not ret:
            return Exception(err_message)

        try:
            output: List[Dict[str, Any]] = []
            for item in dests:
                model_id = int(item[0])
                exported, export_id, name, confidence = self.is_modelid_export(model_id)
                if not exported:
                    continue
                node = {
                    "title": name,
                    "code": export_id,
                    "left": int(item[1]),
                    "top": int(item[2]),
                    "right": int(item[3]),
                    "bottom": int(item[4]),
                    "confidence": round(float(item[5]), 2),
                }
                if node["confidence"] >= confidence:
                    output.append(node)
            return output
        except Exception as err:
            return Exception("解析推理识别结果异常.源推理识别结果:{}.错误信息:{}".format(
                str(dests), ExceptionMessage(err)))

    def draw_detections(self, image_bgr, detections: List[Dict[str, Any]],
                        out_path: str) -> Tuple[bool, str]:
        """在原图上绘制检测框并保存结果图。"""
        try:
            import cv2
        except ImportError as err:
            return False, "缺少 opencv 依赖: {}".format(ExceptionMessage(err))

        if image_bgr is None:
            return False, "图像数据为空,无法绘制"
        image = image_bgr
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        for index, item in enumerate(detections or []):
            left, top = int(item.get("left", 0)), int(item.get("top", 0))
            right, bottom = int(item.get("right", 0)), int(item.get("bottom", 0))
            color = _color_for(index)
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            label = "{} {:.2f}".format(item.get("title", ""), float(item.get("confidence", 0.0)))
            _draw_label(cv2, image, label, left, top, color)

        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, image)
        except Exception as err:
            return False, "结果图保存失败({}): {}".format(out_path, ExceptionMessage(err))
        return True, "normal"

    # ------------------------------------------------------------------ 多任务格式化/绘制
    def predict_to_outputformat(self, raw, task: str):
        """任务无关的响应格式化。

        detection → 客户检测结构; seg/pose/obb → 检测框 + 附加(掩膜/关键点/角度);
        cls → topk 类别; depth/sem → 编码图。
        """
        task = str(task or "detection").lower()
        if task == "detection":
            return self.dest_to_outputformat(raw)

        if task in ("segment", "pose", "obb") and hasattr(raw, "boxes"):
            return self._format_task_objects(raw, task)
        if task == "cls" and hasattr(raw, "topk_scores"):
            return self._format_cls(raw)
        if task in ("depth", "sem"):
            import base64

            import cv2
            import numpy as np

            dense = np.asarray(raw)
            if dense.ndim == 2:
                if task == "depth":
                    vmin, vmax = float(dense.min()), float(dense.max())
                    if vmax > vmin:
                        dense = (dense - vmin) / (vmax - vmin)
                    dense = np.clip(dense * 255.0, 0, 255).astype(np.uint8)
                    dense = cv2.applyColorMap(dense, cv2.COLORMAP_JET)
                else:
                    dense = (dense / float(max(1, int(dense.max())))).astype(np.uint8) * 255
                    dense = np.dstack([dense] * 3)
                ok, buf = cv2.imencode(".jpg", dense)
                if not ok:
                    return []
                return [{"task": task, "image": base64.b64encode(buf.tobytes()).decode("utf-8")}]
            return []
        return self.dest_to_outputformat(raw)

    def _format_task_objects(self, raw, task: str):
        ret, err_message = self.load_confidence()
        if not ret:
            return Exception(err_message)
        output: List[Dict[str, Any]] = []
        import numpy as np

        for i in range(raw.boxes.shape[0]):
            model_id = int(raw.class_ids[i]) + 1
            exported, export_id, name, confidence = self.is_modelid_export(model_id)
            if not exported:
                continue
            score = float(raw.scores[i])
            if score < confidence:
                continue
            node = {
                "title": name,
                "code": export_id,
                "left": int(round(float(raw.boxes[i, 0]))),
                "top": int(round(float(raw.boxes[i, 1]))),
                "right": int(round(float(raw.boxes[i, 2]))),
                "bottom": int(round(float(raw.boxes[i, 3]))),
                "confidence": round(score, 2),
            }
            if task == "pose" and raw.keypoints is not None:
                node["keypoints"] = [
                    {"x": int(round(float(p[0]))), "y": int(round(float(p[1]))), "conf": round(float(p[2]), 2)}
                    for p in np.asarray(raw.keypoints[i])
                ]
            if task == "obb" and raw.angles is not None:
                node["angle"] = round(float(np.degrees(raw.angles[i])), 2)
            if task == "segment" and raw.mask is not None:
                node["mask"] = _mask_to_rle(raw.mask[i])
            output.append(node)
        return output

    def _format_cls(self, raw):
        output = []
        for cls_id, score in zip(raw.topk_class_ids, raw.topk_scores):
            model_id = int(cls_id) + 1
            exported, export_id, name, _confidence = self.is_modelid_export(model_id)
            if not exported:
                continue
            output.append({"code": export_id, "title": name, "confidence": round(float(score), 2)})
        return output

    def draw_task(self, image_bgr, data, task: str, out_path: str) -> Tuple[bool, str]:
        """按任务绘制结果图(检测框 + 姿态关键点连线 / 旋转框 / 掩膜)。"""
        try:
            import cv2
        except ImportError as err:
            return False, "缺少 opencv 依赖: {}".format(ExceptionMessage(err))
        if image_bgr is None:
            return False, "图像数据为空,无法绘制"
        image = image_bgr
        task = str(task or "detection").lower()
        for index, item in enumerate(data or []):
            color = _color_for(index)
            left, top, right, bottom = (int(item.get("left", 0)), int(item.get("top", 0)),
                                        int(item.get("right", 0)), int(item.get("bottom", 0)))
            if right > left and bottom > top:
                cv2.rectangle(image, (left, top), (right, bottom), color, 2)
                _draw_label(cv2, image,
                            "{} {:.2f}".format(item.get("title", ""), float(item.get("confidence", 0.0))),
                            left, top, color)
            if task == "pose" and item.get("keypoints"):
                pts = [(int(k["x"]), int(k["y"])) for k in item["keypoints"] if k["conf"] > 0.0]
                _draw_skeleton(cv2, image, pts, color)
            if task == "mask" or (task == "segment" and item.get("mask")):
                _draw_mask(cv2, image, item["mask"])
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, image)
        except Exception as err:
            return False, "结果图保存失败({}): {}".format(out_path, ExceptionMessage(err))
        return True, "normal"

    def read_image(self, image_path: str):
        """读取图片为 BGR ndarray。"""
        import cv2

        image = cv2.imdecode(np_from_file(image_path), cv2.IMREAD_COLOR)
        if image is None:  # 兼容中文路径
            image = cv2.imread(image_path)
        return image


def np_from_file(path: str):
    """以二进制读文件并转 numpy 缓冲(规避 OpenCV 中文路径问题)。"""
    import numpy as np

    with open(path, "rb") as fr:
        return np.frombuffer(fr.read(), dtype=np.uint8)


def _color_for(index: int):
    palette = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
               (255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 0, 255)]
    return palette[index % len(palette)]


def _draw_label(cv2, image, label: str, left: int, top: int, color) -> None:
    """绘制带底色的标签(不依赖中文字体, 避免乱码)。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.5, 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
    y = max(top, text_h + baseline)
    cv2.rectangle(image, (left, y - text_h - baseline), (left + text_w, y + baseline), color, -1)
    cv2.putText(image, label, (left, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


#: 姿态骨架连接对(COCO 17 点常用拓扑), 用于绘制关键点连线
_SKELETON_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)


def _draw_skeleton(cv2, image, pts, color) -> None:
    for index, point in enumerate(pts):
        cv2.circle(image, point, 2, color, -1, cv2.LINE_AA)
    for a, b in _SKELETON_EDGES:
        if a < len(pts) and b < len(pts):
            cv2.line(image, pts[a], pts[b], color, 1, cv2.LINE_AA)


def _mask_to_rle(binary_mask) -> str:
    """把二值实例掩膜压缩为 run-length 编码字符串(便于 JSON 传输)。"""
    import base64

    import numpy as np

    mask = np.asarray(binary_mask)
    if mask.dtype != np.uint8:
        mask = np.asarray(mask > 0, np.uint8)
    return base64.b64encode((mask > 0).tobytes()).decode("utf-8")


def _draw_mask(cv2, image, mask) -> None:
    import base64

    import numpy as np

    try:
        if isinstance(mask, str):
            mask_b = base64.b64decode(mask)
            size = int(round(np.sqrt(len(mask_b)))) or 0
            if size:
                img = np.frombuffer(mask_b, np.uint8).reshape(size, size)
            else:
                return
        else:
            img = np.asarray(mask)
        if img.size == 0:
            return
        colored = np.zeros((*img.shape, 3), np.uint8)
        colored[img > 0] = (0, 180, 0)
        image[...] = cv2.addWeighted(image, 1.0, colored, 0.35, 0)
    except Exception:
        pass


def result_image_path(filename: str, result_dir: str) -> Optional[str]:
    """校验结果图文件名, 防止目录穿越。"""
    base = os.path.basename(filename or "")
    if not base or base != filename:
        return None
    if not base.lower().endswith(_RESULT_SUFFIXES):
        return None
    return os.path.join(result_dir, base)
