# -*- coding: utf-8 -*-
"""识别结果后处理与置信度控制。

输出格式与客户 v1.4 规范一致:
``data=[{title, code, left, top, right, bottom, confidence}, ...]``
类别、导出编号与阈值全部来自交付包 ``config/confidence.json``, 与模型代码解耦。
"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import runtime_config
from runtime_tools import ExceptionMessage

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
_RESULT_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


class NwaiUtils:
    """置信度表与结果格式化工具。"""

    def __init__(self) -> None:
        self.confidence_list: List[Dict[str, Any]] = []
        self.confidence_path = runtime_config.confidence_path()

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


def result_image_path(filename: str, result_dir: str) -> Optional[str]:
    """校验结果图文件名, 防止目录穿越。"""
    base = os.path.basename(filename or "")
    if not base or base != filename:
        return None
    if not base.lower().endswith(_RESULT_SUFFIXES):
        return None
    return os.path.join(result_dir, base)
