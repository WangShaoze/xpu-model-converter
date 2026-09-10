# -*- coding: utf-8 -*-
"""Paddle 前端适配器 (第三阶段能力, 结构已就位)。"""
from xpu_converter.frontend.paddle.base import PaddleAdapter
from xpu_converter.frontend.paddle.ppocr import PPOCRAdapter
from xpu_converter.frontend.paddle.paddledetection import PaddleDetectionAdapter

__all__ = ["PaddleAdapter", "PPOCRAdapter", "PaddleDetectionAdapter"]
