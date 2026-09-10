# -*- coding: utf-8 -*-
"""PP-OCR 适配器 (第三阶段)。

静态 shape 默认值仅作占位, 落地时按客户实际模型 (det / rec / cls) 调整;
OCR 后处理在公共 Runtime 的 ``ocr`` 模块中实现。
"""
from xpu_converter.frontend.paddle.base import PaddleAdapter


class PPOCRAdapter(PaddleAdapter):
    model_type = "ppocr"
    task = "ocr"
    end2end_default = False

    def default_input_shape(self):
        # PP-OCRv4 检测子模型默认输入
        return [1, 3, 960, 960]

    def default_num_classes(self) -> int:
        return 0

    def default_class_names(self):
        return []
