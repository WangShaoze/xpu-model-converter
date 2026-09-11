# -*- coding: utf-8 -*-
"""PyTorch 前端适配器基类。

导出策略 (按优先级):

1. ultralytics 可用且适配器声明了 ``ultralytics_name`` → 走 ``YOLO.export(format="onnx")``,
   这是最贴近官方导出的路径;
2. 回退 ``torch.onnx.export``, 用输出包装器把 forward 返回值统一为 Tensor 元组。
"""
import os
import shutil
from typing import Any, List, Optional, Tuple

from xpu_converter.errors import ExportError, ModelLoadError
from xpu_converter.frontend.base import BaseModelAdapter, FrontendModel
from xpu_converter.logging_utils import get_logger

logger = get_logger(__name__)


def ultralytics_available() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


def require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ModelLoadError("缺少 PyTorch 依赖, 请执行: pip install torch")


def collect_tensors(torch, output: Any) -> List[Any]:
    """深度优先收集 forward 返回值中的所有 Tensor。"""
    tensors: List[Any] = []

    def _walk(obj):
        if isinstance(obj, torch.Tensor):
            tensors.append(obj)
        elif isinstance(obj, dict):
            for value in obj.values():
                _walk(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                _walk(value)

    _walk(output)
    return tensors


def build_output_wrapper(torch, model):
    """把模型包装为「输入 Tensor -> 输出 Tensor(元组)」的 nn.Module。"""

    class OutputWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, x):
            tensors = collect_tensors(torch, self.model(x))
            if len(tensors) == 1:
                return tensors[0]
            return tuple(tensors)

    return OutputWrapper()


class PyTorchAdapter(BaseModelAdapter):
    """PyTorch 通用适配器。"""

    framework = "pytorch"
    #: ultralytics 侧模型名; 为 None 表示不使用 ultralytics 导出
    ultralytics_name: Optional[str] = None

    # ------------------------------------------------------------- 加载识别
    def load_model(self, model_path: str) -> Any:
        torch = require_torch()
        path = self._require_file(model_path)
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # 兼容不支持 weights_only 的旧版本
            obj = torch.load(path, map_location="cpu")
        except Exception as err:
            raise ModelLoadError("torch.load 读取权重失败: {} ({})".format(path, err))
        return self._extract_module(torch, obj, path)

    def _extract_module(self, torch, obj: Any, path: str) -> Any:
        if isinstance(obj, torch.nn.Module):
            return obj
        if isinstance(obj, dict):
            for key in ("model", "ema", "net", "network"):
                candidate = obj.get(key)
                if isinstance(candidate, torch.nn.Module):
                    return candidate
        raise ModelLoadError(
            "{} 中未找到可导出的 nn.Module。请提供: 1) ultralytics 训练得到的 .pt 权重; "
            "2) 含模型结构的 torch.save(model) 文件; 3) 或先导出 ONNX 再执行 compile/package。".format(path)
        )

    @staticmethod
    def _extract_class_names(model: Any) -> List[str]:
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names.keys())]
        if isinstance(names, (list, tuple)):
            return [str(n) for n in names]
        return []

    def inspect(self, model_path: str) -> FrontendModel:
        path = self._require_file(model_path)
        info = FrontendModel(
            framework=self.framework,
            model_type=self.model_type,
            task=self.task,
            source_path=path,
            input_shape=self.input_shape(),
            input_dtype=self.config.input_dtype,
            num_classes=int(self.config.num_classes or self.default_num_classes()),
            class_names=list(self.config.class_names or self.default_class_names()),
            end2end=bool(self.config.end2end) or self.end2end_default,
        )
        try:
            model = self.load_model(path)
        except ModelLoadError as err:
            info.extra["load_warning"] = str(err)
            return info
        info.extra["module_class"] = type(model).__name__
        names = self._extract_class_names(model)
        if names:
            info.class_names = names
            info.num_classes = len(names)
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
        path = self._require_file(model_path)
        shape = [int(v) for v in (input_shape or self.input_shape())]
        if len(shape) != 4:
            raise ExportError("input_shape 必须是 4 维 NCHW, 当前: {}".format(shape))
        opset_version = int(opset or self.opset())
        dynamic_flag = bool(self.config.dynamic if dynamic is None else dynamic)
        output_path = str(output_path)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

        if self.ultralytics_name and ultralytics_available():
            try:
                return self._export_via_ultralytics(path, output_path, shape, opset_version, dynamic_flag)
            except ExportError:
                pass
            except Exception as err:  # ultralytics 失败时回退到通用导出
                logger.warning("ultralytics 导出失败, 回退 torch.onnx.export: %s", err)
        return self._export_via_torch(path, output_path, shape, opset_version, dynamic_flag)

    def _export_via_ultralytics(self, path, output_path, shape, opset, dynamic) -> str:
        from ultralytics import YOLO

        model = YOLO(path)
        imgsz = [int(shape[2]), int(shape[3])]
        exported = model.export(
            format="onnx", imgsz=imgsz, opset=opset, dynamic=dynamic, simplify=False, batch=1, device="cpu"
        )
        if not exported or not os.path.isfile(str(exported)):
            raise ExportError("ultralytics 未产出 ONNX 文件")
        if os.path.abspath(str(exported)) != os.path.abspath(output_path):
            shutil.move(str(exported), output_path)
        return output_path

    def _export_via_torch(self, path, output_path, shape, opset, dynamic) -> str:
        torch = require_torch()
        model = self.load_model(path)
        model.eval()
        wrapper = build_output_wrapper(torch, model)
        dummy = torch.zeros(*shape, dtype=torch.float32)
        with torch.no_grad():
            probed = collect_tensors(torch, wrapper(dummy))
        if not probed:
            raise ExportError("模型 forward 未返回任何 Tensor, 无法导出 ONNX")
        output_names = ["output"] if len(probed) == 1 else ["output{}".format(i + 1) for i in range(len(probed))]
        dynamic_axes = None
        if dynamic:
            dynamic_axes = {"images": {0: "batch"}}
            for name in output_names:
                dynamic_axes[name] = {0: "batch"}
        try:
            torch.onnx.export(
                wrapper,
                dummy,
                output_path,
                input_names=["images"],
                output_names=output_names,
                opset_version=opset,
                do_constant_folding=True,
                dynamic_axes=dynamic_axes,
            )
        except Exception as err:
            raise ExportError("torch.onnx.export 导出失败: {} ({})".format(path, err))
        logger.info("已导出 ONNX: %s (opset=%s, dynamic=%s)", output_path, opset, dynamic)
        return output_path
