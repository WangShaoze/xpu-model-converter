# -*- coding: utf-8 -*-
"""硬件能力(Environment)探测。

把"这台机器上到底有没有可用的昆仑 XPU"从猜想变成一次显式探测, 并把结果
固化成 :class:`HardwareCapability` 写进 artifact metadata / preflight 报告
(ChatGPT 修改意见 §5 / §13 / §32)。
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

# 昆仑 XPU 运行时/ML 动态库的常见落点
XPU_LIB_DIRS = ("/usr/local/xpu", "/usr/local/xpu-4.31.0/lib64", "/usr/lib64")
XPU_LIB_NAMES = ("libxpurt.so", "libxpuml.so")
# 设备节点: 有 /dev/xpuctrl 通常意味着宿主机挂了加速卡
XPU_DEVICE_NODES = ("/dev/xpuctrl", "/dev/xpu0", "/dev/xpuv3")


@dataclass
class HardwareCapability:
    """目标硬件能力与版本指纹。"""

    name: str = "kunlun"
    chip: str = "auto"
    sdk_version: str = ""
    compiler_version: str = ""
    driver_version: str = ""
    firmware_version: str = ""
    device_count: int = 0
    device_available: bool = False
    supported_precisions: FrozenSet[str] = frozenset({"fp32", "fp16"})
    supported_opsets: Optional[tuple] = None      # (min, max), None 表示不限制
    dynamic_shape: bool = False
    layouts: FrozenSet[str] = frozenset({"NCHW"})
    python_version: str = ""
    paddle_version: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "chip": self.chip,
            "sdk_version": self.sdk_version,
            "compiler_version": self.compiler_version,
            "driver_version": self.driver_version,
            "firmware_version": self.firmware_version,
            "device_count": self.device_count,
            "device_available": self.device_available,
            "supported_precisions": sorted(self.supported_precisions),
            "supported_opsets": list(self.supported_opsets) if self.supported_opsets else None,
            "dynamic_shape": self.dynamic_shape,
            "layouts": sorted(self.layouts),
            "python_version": self.python_version,
            "paddle_version": self.paddle_version,
            "notes": list(self.notes),
        }


def _detect_xpu_lib() -> Optional[str]:
    for directory in XPU_LIB_DIRS:
        root = Path(directory)
        if not root.exists():
            continue
        for name in XPU_LIB_NAMES:
            for found in root.rglob(name):
                return str(found)
    return None


def _detect_sdk_version(lib_path: Optional[str]) -> str:
    # 形如 /usr/local/xpu-4.31.0/lib64/libxpurt.so -> 4.31.0
    if lib_path:
        for part in Path(lib_path).parts:
            if part.startswith("xpu-"):
                return part[len("xpu-"):]
    env = os.environ.get("XPU_SDK_VERSION")
    return str(env or "")


def _detect_paddle_version() -> str:
    try:
        import paddle  # noqa: F401

        return str(getattr(paddle, "__version__", "") or "")
    except Exception:
        return ""


def _detect_xpu_in_paddle() -> bool:
    """Paddle 是否内置 XPU 支持(compile with XPU)。"""
    try:
        from paddle import inference

        return hasattr(inference, "Config") and hasattr(inference.Config(""), "enable_xpu")
    except Exception:
        return False


def probe_kunlun(target_chip: str = "auto", device: str = "auto") -> HardwareCapability:
    """探测当前环境的昆仑 XPU 能力。"""
    import platform

    notes: List[str] = []
    lib_path = _detect_xpu_lib()
    device_nodes = [node for node in XPU_DEVICE_NODES if Path(node).exists()]
    device_available = bool(device_nodes)
    paddle_version = _detect_paddle_version()
    paddle_xpu = _detect_xpu_in_paddle()

    if not lib_path:
        notes.append("未找到昆仑运行时动态库({}), 仅有 paddle 无法判定 XPU 可用".format(
            ", ".join(XPU_LIB_NAMES)))
    if not device_available and str(device or "").lower() != "cpu":
        notes.append("未发现 XPU 设备节点({})".format(", ".join(XPU_DEVICE_NODES)))
    if paddle_version and not paddle_xpu:
        notes.append("当前 paddle({}) 未编译 XPU 支持(Config 无 enable_xpu)".format(paddle_version))
    if not paddle_version:
        notes.append("当前环境未安装 paddlepaddle, 无法通过 Paddle Inference 使用 XPU")

    return HardwareCapability(
        name="kunlun",
        chip=str(target_chip or "auto"),
        sdk_version=_detect_sdk_version(lib_path),
        device_count=len(device_nodes),
        device_available=device_available and paddle_xpu,
        python_version=platform.python_version(),
        paddle_version=paddle_version,
        notes=notes,
    )
