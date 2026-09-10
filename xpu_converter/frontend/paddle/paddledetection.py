# -*- coding: utf-8 -*-
"""PaddleDetection 适配器 (第三阶段)。"""
from xpu_converter.frontend.paddle.base import PaddleAdapter


class PaddleDetectionAdapter(PaddleAdapter):
    model_type = "paddledetection"
    task = "detection"
    end2end_default = True

    def default_input_shape(self):
        return [1, 3, 640, 640]
