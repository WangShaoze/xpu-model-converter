# -*- coding: utf-8 -*-
"""Rewrite 层: 把后端不支持的算子改写为等价的可支持形式。"""
from xpu_converter.rewrite.base import RewriteRule
from xpu_converter.rewrite.registry import RewriteRegistry, RewriteResult

__all__ = ["RewriteRule", "RewriteRegistry", "RewriteResult"]
