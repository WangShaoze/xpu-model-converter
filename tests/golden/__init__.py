# -*- coding: utf-8 -*-
"""Golden Model 校验数据与发现器(ChatGPT 修改意见 §47/§48)。

目的: 让"改动 SiLU / Resize / Conv-BN / NMS 会不会把 YOLO 搞坏"变成一次自动回归。

目录约定(每个子目录即一个用例, 用例目录名不带前导 ``_``/``.``)::

    tests/golden/<case>/
        model.pt            # 必填(或 model.onnx): 真实框架模型 / ONNX 中间模型
        expected.yaml       # 可选: 任务/输入形状/类别数/阈值等期望值
        images/             # 可选: 真实图片(bus.jpg / zidane.jpg ...)
        reference/
            outputs.npy     # 可选: 参考输出张量(与 ONNX 导出结果逐张量比对)
            detections.json # 可选: 参考检测框

未放置任何用例时, 相关测试会 skip 而非失败; 放入用例后即自动纳入回归。
注意: 这里**不内置任何真实权重**, 由交付团队按客户模型灌入, 避免仓库体积膨胀。
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

GOLDEN_ROOT = Path(__file__).resolve().parent
MODEL_FILENAMES = ("model.pt", "model.onnx", "model.pdmodel")


@dataclass
class GoldenCase:
    """单个 golden 用例。"""

    name: str
    root: Path
    model_path: Path
    expected: Dict[str, Any] = field(default_factory=dict)
    images: List[Path] = field(default_factory=list)
    reference_outputs: Optional[Path] = None
    reference_detections: Optional[Path] = None

    @property
    def framework(self) -> str:
        if self.model_path.suffix.lower() == ".onnx":
            return "onnx"
        if self.model_path.suffix.lower() == ".pdmodel":
            return "paddle"
        return str(self.expected.get("framework") or "pytorch")

    @property
    def model_type(self) -> str:
        return str(self.expected.get("model_type") or self.name.split("_")[0] or "yolov10")


def _load_expected(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    from xpu_converter.config import load_yaml

    data = load_yaml(path)
    return data if isinstance(data, dict) else {}


def discover_cases(root: Optional[Path] = None) -> List[GoldenCase]:
    """扫描 ``tests/golden/<case>/`` 下的 golden 用例(无模型文件的目录会被忽略)。"""
    base = Path(root or GOLDEN_ROOT)
    cases: List[GoldenCase] = []
    if not base.is_dir():
        return cases
    for child in sorted(base.iterdir()):
        if not child.is_dir() or child.name.startswith((".", "_")):
            continue
        model_path = next((child / name for name in MODEL_FILENAMES if (child / name).is_file()), None)
        if model_path is None:
            continue
        images = sorted(p for p in (child / "images").glob("*") if p.is_file()) \
            if (child / "images").is_dir() else []
        reference_dir = child / "reference"
        cases.append(GoldenCase(
            name=child.name,
            root=child,
            model_path=model_path,
            expected=_load_expected(child / "expected.yaml"),
            images=images,
            reference_outputs=(reference_dir / "outputs.npy") if (reference_dir / "outputs.npy").is_file() else None,
            reference_detections=(reference_dir / "detections.json") if (reference_dir / "detections.json").is_file() else None,
        ))
    return cases


def load_reference_detections(case: GoldenCase) -> List[Dict[str, Any]]:
    """读取 ``reference/detections.json``(缺失时返回空列表)。"""
    if not case.reference_detections:
        return []
    with open(case.reference_detections, "r", encoding="utf-8") as fr:
        data = json.load(fr)
    return data if isinstance(data, list) else list((data or {}).get("detections") or [])
