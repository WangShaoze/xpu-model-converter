# -*- coding: utf-8 -*-
"""Paddle 前端适配器基类。

Paddle 侧输入是 PaddleInference 静态图目录 (``model.pdmodel`` + ``model.pdiparams``),
通过 ``paddle2onnx`` 转成 ONNX 后进入统一流水线。

依赖 (目标机器上安装):
    pip install paddlepaddle paddle2onnx
"""
import os
import shutil
import subprocess
import sys
from typing import Any, List, Optional

from xpu_converter.errors import BackendNotAvailableError, ExportError, ModelLoadError
from xpu_converter.frontend.base import BaseModelAdapter, FrontendModel
from xpu_converter.logging_utils import get_logger

logger = get_logger(__name__)

PROGRAM_SUFFIXES = (".json", ".pdmodel")


def has_program_file(directory: str) -> bool:
    """目录下是否存在 PaddleInference 静态图 prog 文件。"""
    if not os.path.isdir(directory):
        return False
    for name in os.listdir(directory):
        if name.endswith(PROGRAM_SUFFIXES) and not name.endswith(".pdiparams"):
            return True
    return False


def is_inference_dir(path: str) -> bool:
    return has_program_file(path) and any(f.endswith(".pdiparams") for f in os.listdir(path))


def resolve_inference_dir(model_path: str) -> str:
    """定位静态图目录, 兼容直接传目录 / 传 prog 文件 / 传父目录。"""
    path = os.path.abspath(str(model_path or ""))
    if os.path.isfile(path):
        path = os.path.dirname(path)
    if not os.path.isdir(path):
        raise ModelLoadError("Paddle 模型目录不存在: {}".format(path))
    if is_inference_dir(path):
        return path
    for candidate in sorted(os.listdir(path)):
        sub = os.path.join(path, candidate)
        if os.path.isdir(sub) and is_inference_dir(sub):
            return sub
    raise ModelLoadError(
        "在 {} 下未找到 PaddleInference 静态图 (需同时存在 prog(.json/.pdmodel) 与 .pdiparams)".format(path)
    )


def find_program_and_params(inference_dir: str):
    programs = sorted(
        f for f in os.listdir(inference_dir) if f.endswith(PROGRAM_SUFFIXES) and not f.endswith(".pdiparams")
    )
    params = sorted(f for f in os.listdir(inference_dir) if f.endswith(".pdiparams"))
    if not programs or not params:
        raise ModelLoadError("{} 缺少 prog 或 params 文件".format(inference_dir))
    return programs[0], params[0]


def paddle2onnx_available() -> bool:
    if shutil.which("paddle2onnx"):
        return True
    try:
        import paddle2onnx  # noqa: F401
        return True
    except Exception:
        return False


class PaddleAdapter(BaseModelAdapter):
    """Paddle 静态图适配器。"""

    framework = "paddle"
    end2end_default = False

    def default_num_classes(self) -> int:
        # 检测类模型的类别数无法从静态图读取, 默认沿用 COCO; 客户模型请在模型配置中覆盖
        return 80

    # ------------------------------------------------------------- 加载识别
    def load_model(self, model_path: str) -> Any:
        inference_dir = resolve_inference_dir(model_path)
        program, params = find_program_and_params(inference_dir)
        return {"inference_dir": inference_dir, "program": program, "params": params}

    def inspect(self, model_path: str) -> FrontendModel:
        inference_dir = resolve_inference_dir(model_path)
        program, params = find_program_and_params(inference_dir)
        info = FrontendModel(
            framework=self.framework,
            model_type=self.model_type,
            task=self.task,
            source_path=inference_dir,
            input_shape=self.input_shape(),
            input_dtype=self.config.input_dtype,
            num_classes=int(self.config.num_classes or self.default_num_classes()),
            class_names=list(self.config.class_names or self.default_class_names()),
            end2end=bool(self.config.end2end) or self.end2end_default,
        )
        info.extra["program"] = program
        info.extra["params"] = params
        return info

    # ------------------------------------------------------------- 导出
    def export_onnx(
        self,
        model_path: str,
        output_path: str,
        input_shape: Optional[List[int]] = None,
        opset: Optional[int] = None,
        dynamic: Optional[bool] = None,
    ) -> str:
        if not paddle2onnx_available():
            raise BackendNotAvailableError(
                "缺少 paddle2onnx, 无法把 Paddle 静态图转换为 ONNX。请执行: pip install paddle2onnx"
            )
        inference_dir = resolve_inference_dir(model_path)
        program, params = find_program_and_params(inference_dir)
        output_path = str(output_path)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        opset_version = int(opset or self.opset())
        dynamic_flag = bool(self.config.dynamic if dynamic is None else dynamic)

        cli = shutil.which("paddle2onnx")
        if cli:
            cmd = [
                cli,
                "--model_dir", inference_dir,
                "--model_filename", program,
                "--params_filename", params,
                "--save_file", output_path,
                "--opset_version", str(opset_version),
                "--enable_onnx_checker", "True",
            ]
            if dynamic_flag:
                cmd += ["--input_shape_dict", "{}"]  # 占位: 动态 shape 由模型配置显式给出
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 or not os.path.isfile(output_path):
                raise ExportError("paddle2onnx 执行失败: {}".format(result.stderr or result.stdout))
            logger.info("已通过 paddle2onnx CLI 导出 ONNX: %s", output_path)
            return output_path

        try:
            import paddle2onnx  # type: ignore

            paddle2onnx.command.program2onnx(  # type: ignore[attr-defined]
                model_dir=inference_dir,
                model_filename=program,
                params_filename=params,
                save_file=output_path,
                opset_version=opset_version,
            )
        except Exception as err:
            raise ExportError("paddle2onnx Python API 导出失败: {} ({})".format(sys.executable, err))
        if not os.path.isfile(output_path):
            raise ExportError("paddle2onnx 未产出 ONNX 文件: {}".format(output_path))
        logger.info("已通过 paddle2onnx API 导出 ONNX: %s", output_path)
        return output_path
