# -*- coding: utf-8 -*-
"""测试包。

按依赖分层(ChatGPT 修改意见 §45/§46): 缺少可选依赖(onnx 等)时, 相关测试应以
*skip* 结束, 而不是让整套测试全部 ERROR。用 :data:`requires_onnx` 装饰需要 onnx
的测试类/用例即可。
"""
import unittest

from xpu_converter.ir import onnx as onnx_ir


def onnx_available() -> bool:
    """本机是否具备 onnx(ONNX Integration 层测试的前置依赖)。"""
    return onnx_ir.available()


#: 需要 onnx 的测试: 缺失时跳过而非报错
requires_onnx = unittest.skipUnless(onnx_available(), "requires onnx")
