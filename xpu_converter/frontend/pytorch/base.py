# -*- coding: utf-8 -*-
"""PyTorch 前端适配器基类。

导出策略 (按优先级):

1. ultralytics 可用且适配器声明了 ``ultralytics_name`` → 走 ``YOLO.export(format="onnx")``,
   这是最贴近官方导出的路径;
2. 回退 ``torch.onnx.export``, 用输出包装器把 forward 返回值统一为 Tensor 元组。
"""
import os
import shutil
import sys
import types
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


#: thuyngch(王建尧/WongKinYiu)系模型在反序列化时会 import ``models.*``(来自其源码
#: 仓库), 而 ``models/common.py -> utils/general.py`` 又会在 import 期触发
#: ``import torchvision``。本环境 torch 与 torchvision 的 ABI 不匹配(装了 torchvision
#: 也无法正常 import), 因此用一份只够 import 期通过的 torchvision 占位模块将其隔离,
#: 让 checkpoint 能顺利反序列化。真正在导出阶段不调用 torchvision 的 NMS, 故不受影响。
_TORCHVISION_STUB_FLAG = "_xpu_stub"


def install_torchvision_stub() -> None:
    """在 sys.modules 注入一份 torchvision 占位模块(幂等)。

    仅当真实 torchvision 无法 import 时才注入; 若真实库可用则不做任何事。
    """
    if sys.modules.get("torchvision") is not None:
        if getattr(sys.modules["torchvision"], _TORCHVISION_STUB_FLAG, False):
            return  # 已注入
        try:
            import torchvision  # noqa: F401
            return  # 真实 torchvision 可用, 不注入
        except Exception:
            pass

    real = types.ModuleType("torchvision")
    real.__path__ = []

    ops = types.ModuleType("torchvision.ops")

    def _nms(boxes, scores, iou_thres=None, *args, **kwargs):
        # 占位实现: 按分数排序取前 30000 个索引; 仅用于满足 import, 导出不调用。
        if boxes is None or not hasattr(boxes, "numel") or boxes.numel() == 0:
            return __import__("torch").empty(0, dtype=__import__("torch").long)
        torch = __import__("torch")
        n = min(scores.numel(), 30000)
        return torch.argsort(scores, descending=True)[:n]

    ops.nms = _nms
    ops.non_max_suppression = _nms
    # thuyngch 系(yolov7/9) utils/common 在 import 期会引用下列符号; 仅需存在即可
    ops.sigmoid_focal_loss = lambda *a, **k: None
    for _op in ("DeformConv2d", "roi_pool", "roi_align", "ps_roi_pool", "ps_roi_align",
                "RoIAlign", "RoIPool", "PSRoIAlign", "PSRoIPool"):
        if not hasattr(ops, _op):
            setattr(ops, _op, None)
    real.ops = ops

    datasets = types.ModuleType("torchvision.datasets")
    for _base in ("ImageFolder", "CocoDetection", "VOCDetection", "MNIST"):
        setattr(datasets, _base, list)
    real.datasets = datasets

    transforms = types.ModuleType("torchvision.transforms")
    for _name in ("Compose", "ToTensor", "Normalize", "Resize", "CenterCrop",
                  "RandomCrop", "RandomHorizontalFlip", "RandomAffine", "Grayscale"):
        setattr(transforms, _name, list if _name == "Compose" else type(_name, (), {}))
    real.transforms = transforms

    functional = types.ModuleType("torchvision.transforms.functional")
    for _name in ("to_tensor", "resize", "normalize", "hflip", "vflip", "affine", "rgb_to_grayscale"):
        setattr(functional, _name, lambda *a, **k: a[0] if a else None)
    real.transforms.functional = functional

    models_mod = types.ModuleType("torchvision.models")
    for _name in ("resnet18", "efficientnet_b0"):
        setattr(models_mod, _name, lambda *a, **k: None)
    real.models = models_mod

    utils = types.ModuleType("torchvision.utils")
    utils.save_image = lambda *a, **k: None
    real.utils = utils

    setattr(real, _TORCHVISION_STUB_FLAG, True)
    for _mod in (real, ops, datasets, transforms, functional, models_mod, utils):
        sys.modules.setdefault(_mod.__name__, _mod)
    # 顶层模块的注册顺序要在子模块之后, 确保属性已挂载
    sys.modules["torchvision"] = real


def prepend_model_source_root(root: str) -> None:
    """把 thuyngch 源码仓库根目录加入 sys.path(前置), 供 checkpoint 反序列化导入 ``models.*``。"""
    root = os.path.abspath(root)
    if root in sys.path:
        return
    sys.path.insert(0, root)


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


def build_raw_output_wrapper(torch, model, index: int):
    """只导出 forward 返回值中的**第 index 个主输出**(去掉后处理/指标等旁支)。

    thuyngch 系检测模型(yolov9 等)的 ``forward`` 常返回
    ``[raw_tensor, postprocess_metrics]``; 若按整体收集会把指标分支一并追踪,
    引入动态 shape。这里只保留主张量, 从而得到静态通道的干净 ONNX。
    """

    class RawOutputWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.index = index

        def forward(self, x):
            output = self.model(x)
            if isinstance(output, dict):
                return list(output.values())[self.index] if self.index is not None else output
            if isinstance(output, (list, tuple)):
                return output[self.index]
            return output

    return RawOutputWrapper()


class PyTorchAdapter(BaseModelAdapter):
    """PyTorch 通用适配器。"""

    framework = "pytorch"
    #: ultralytics 侧模型名; 为 None 表示不使用 ultralytics 导出
    ultralytics_name: Optional[str] = None
    #: 原生 torch.onnx.export 时只导出 forward 返回列表中第 index 个主输出;
    #: None 表示收集全部 Tensor。用于 thuyngch 系(forward 带后处理指标旁支)模型。
    raw_output_index: Optional[int] = None
    #: thuyngch(王建尧/WongKinYiu)系模型需要其源码仓库(models.*) + torchvision 占位
    #: 才能反序列化 checkpoint; 置 True 时导出前自动处理好这两件事。
    requires_torchvision_stub: bool = False
    #: thuyngch 源码仓库根目录(含 models/common.py 等), 用于反序列化导入。
    model_source_root: Optional[str] = None

    # ------------------------------------------------------------- 加载识别
    def load_model(self, model_path: str) -> Any:
        self._prepare_import_env()
        torch = require_torch()
        path = self._require_file(model_path)
        obj = self._torch_load(torch, path)
        return self._extract_module(torch, obj, path)

    @staticmethod
    def _torch_load(torch, path: str) -> Any:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # 兼容不支持 weights_only 的旧版本
            return torch.load(path, map_location="cpu")
        except Exception as err:
            raise ModelLoadError("torch.load 读取权重失败: {} ({})".format(path, err))

    # -------------------------------------------------- thuyngch 系加载前置
    def _prepare_import_env(self) -> None:
        """反序列化 checkpoint 前准备好 import 环境(thuyngch 系专用)。"""
        if self.model_source_root:
            prepend_model_source_root(self.model_source_root)
        if self.requires_torchvision_stub:
            install_torchvision_stub()

    @staticmethod
    def set_export_flag(module: Any, value: bool = True) -> None:
        """把模型内所有检测头模块的 ``export`` 属性置位。

        thuyngch 系 yolo.py 的 Detect 头在 ``export=True`` 时直接返回合并后的
        静态 ``[B, 4+nc, N]`` 主张量(而不是 ``(y, x)`` 双路旁支), 这是得到干净
        ONNX 的关键。
        """
        import torch

        if isinstance(module, torch.nn.Module) and hasattr(module, "export") \
                and not callable(getattr(module, "export", None)):
            module.export = value
        for child in module.children() if isinstance(module, torch.nn.Module) else []:
            PyTorchAdapter.set_export_flag(child, value)

    def _prepare_for_export(self, model: Any) -> None:
        """导出前对已加载模型的定制(如设置 Detect 头的 export 标志)。"""

    def _extract_module(self, torch, obj: Any, path: str) -> Any:
        """从 checkpoint 中取出可导出的 ``nn.Module``(ChatGPT 修改意见 §11)。

        依次尝试: 直接是 Module → 字典里的 model/ema/net/network → 容器内任意
        Module → 交给 ultralytics 按 checkpoint 重建。仍失败则给出可操作提示。
        """
        module = self._find_module(torch, obj)
        if module is not None:
            return module
        module = self._load_via_ultralytics(path)
        if module is not None:
            return module
        raise ModelLoadError(
            "{} 中未找到可导出的 nn.Module(可能是仅含 state_dict 的权重, 缺少结构定义)。"
            "请提供: 1) ultralytics 训练得到的 .pt 权重(含 model/ema); "
            "2) 含模型结构的 torch.save(model) 文件; 3) 或先导出 ONNX 再执行 compile/package。".format(path)
        )

    @classmethod
    def _find_module(cls, torch, obj: Any, depth: int = 0) -> Any:
        """深度受限地查找 checkpoint 内嵌的 ``nn.Module``。"""
        if isinstance(obj, torch.nn.Module):
            return obj
        if depth >= 3:
            return None
        if isinstance(obj, dict):
            for key in ("model", "ema", "net", "network"):
                found = cls._find_module(torch, obj.get(key), depth + 1)
                if found is not None:
                    return found
            for value in obj.values():  # 兜底扫描, 兼容自定义键名
                if isinstance(value, torch.nn.Module):
                    return value
        if isinstance(obj, (list, tuple)):
            for value in obj:
                found = cls._find_module(torch, value, depth + 1)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _load_via_ultralytics(path: str) -> Any:
        """作为最后手段, 用 ultralytics 依据 checkpoint 内的 yaml 重建模型。"""
        if not ultralytics_available():
            return None
        try:
            from ultralytics import YOLO

            loaded = YOLO(path)
            return getattr(loaded, "model", None) or loaded
        except Exception as err:  # ultralytics 不可用时静默回退到报错路径
            logger.warning("ultralytics 无法从 %s 重建模型: %s", path, err)
            return None

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
        self._prepare_import_env()
        model = self.load_model(path)
        model.eval()
        # 原生导出统一以 fp32 进行: 权重可能是 fp16(如部分 thuyngch 权重), 与 dummy 输入对齐
        model = model.float()
        self._prepare_for_export(model)
        if self.raw_output_index is not None:
            wrapper = build_raw_output_wrapper(torch, model, self.raw_output_index)
        else:
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
                dynamo=False,  # 用经典导出器, 避免 torch 2.1x 默认 dynamo 需 onnxscript
            )
        except Exception as err:
            raise ExportError("torch.onnx.export 导出失败: {} ({})".format(path, err))
        logger.info("已导出 ONNX: %s (opset=%s, dynamic=%s)", output_path, opset, dynamic)
        return output_path
