# -*- coding: utf-8 -*-
"""性能基准测试(CLI 步骤 09)。

只做端到端推理时延统计, 不含前后处理, 便于与芯片规格书对齐。
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from xpu_converter.backend.base import BaseRuntimeSession


@dataclass
class BenchmarkResult:
    """单次基准测试结果。

    正式 benchmark 必须带硬件指纹(ChatGPT 修改意见 §23): 同样的模型在不同芯片 /
    SDK / 驱动上的时延没有可比性, 所以把环境信息与数字一起落盘。
    """

    backend: str = ""
    device: str = "auto"
    # 硬件指纹(§23)
    chip: str = ""
    device_id: int = 0
    sdk_version: str = ""
    driver_version: str = ""
    firmware_version: str = ""
    # 被测模型信息(§23)
    model: str = ""
    input_shape: List[int] = field(default_factory=list)
    precision: str = ""
    iterations: int = 0
    warmup: int = 0
    latency_ms: Dict[str, float] = field(default_factory=dict)
    throughput_fps: float = 0.0
    notes: List[str] = field(default_factory=list)
    # 只有在真实目标硬件上跑出来的数据才 available=True; 仿真/占位后端一律 NOT_AVAILABLE
    available: bool = True
    status: str = "OK"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "device": self.device,
            "chip": self.chip,
            "device_id": self.device_id,
            "sdk_version": self.sdk_version,
            "driver_version": self.driver_version,
            "firmware_version": self.firmware_version,
            "model": self.model,
            "input_shape": list(self.input_shape),
            "precision": self.precision,
            "warmup": self.warmup,
            "iterations": self.iterations,
            "latency_ms": dict(self.latency_ms),
            "throughput_fps": self.throughput_fps,
            "notes": list(self.notes),
            "available": self.available,
            "status": self.status,
        }

    def summary(self) -> str:
        if not self.available:
            return "NOT_AVAILABLE ({})".format(self.notes[0] if self.notes else self.backend)
        return "{} iters, avg={:.2f}ms p90={:.2f}ms {:.1f} FPS".format(
            self.iterations,
            self.latency_ms.get("mean", 0.0),
            self.latency_ms.get("p90", 0.0),
            self.throughput_fps,
        )

    def save(self, path: str) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as fw:
            json.dump(self.to_dict(), fw, ensure_ascii=False, indent=2)
        return str(target)


class BenchmarkRunner:
    """会话推理性能测量。"""

    def __init__(self, iterations: int = 50, warmup: int = 5) -> None:
        self.iterations = max(1, int(iterations))
        self.warmup = max(0, int(warmup))

    def run(
        self,
        session: BaseRuntimeSession,
        sample: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        hardware: Optional[Any] = None,
        model: str = "",
        input_shape: Optional[Sequence[int]] = None,
        precision: str = "",
    ) -> BenchmarkResult:
        result = BenchmarkResult(
            backend=session.backend_name,
            device=device,
            model=model,
            input_shape=[int(v) for v in (input_shape or [])],
            precision=precision,
            iterations=self.iterations,
            warmup=self.warmup,
        )
        result.chip, result.sdk_version, result.driver_version, result.firmware_version = (
            self._hardware_fingerprint(hardware)
        )
        result.device_id = 0
        # 仿真/占位后端不得输出任何时延数字: 直接标记 NOT_AVAILABLE, 避免把
        # onnxruntime CPU 的耗时伪装成昆仑 XPU 的性能指标(建设目标 §22)。
        if self._is_simulated(session):
            result.available = False
            result.status = "NOT_AVAILABLE"
            result.notes.append(
                "会话后端为 {} 仿真, 非昆仑 XPU 真实硬件, 性能数据不可用".format(session.backend_name)
            )
            return result

        feeds = self._build_feeds(session, sample)
        for _ in range(self.warmup):
            session.run(feeds)

        timings: List[float] = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            session.run(feeds)
            timings.append((time.perf_counter() - start) * 1000.0)

        result.latency_ms = self._statistics(timings)
        if result.latency_ms.get("mean", 0.0) > 0:
            result.throughput_fps = round(1000.0 / result.latency_ms["mean"], 2)
        return result

    # ------------------------------------------------------------------ 内部
    @staticmethod
    def _hardware_fingerprint(hardware: Optional[Any]) -> tuple:
        """从 HardwareCapability(或 dict) 提取 ``chip/sdk/driver/firmware``(§23)。"""
        if hardware is None:
            return "", "", "", ""
        source = hardware.to_dict() if hasattr(hardware, "to_dict") else dict(hardware or {})
        return (
            str(source.get("chip") or ""),
            str(source.get("sdk_version") or ""),
            str(source.get("driver_version") or ""),
            str(source.get("firmware_version") or ""),
        )

    @staticmethod
    def _build_feeds(session: BaseRuntimeSession, sample: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        names = list(session.input_names)
        if sample:
            if names and all(name in sample for name in names):
                return {name: sample[name] for name in names}
            values = [np.asarray(value) for value in sample.values()]
            if names and len(values) == len(names):
                # 不同后端会重命名张量(如 x2paddle 的 x2paddle_<name>), 按位置对齐
                return dict(zip(names, values))
            return dict(sample)
        shapes = session.input_shapes() or {}
        feeds: Dict[str, Any] = {}
        for name in names:
            shape = shapes.get(name)
            static = [1 if (d is None or int(d) <= 0) else int(d) for d in (shape or [1])]
            feeds[name] = np.zeros(static, dtype=np.float32)
        return feeds

    @staticmethod
    def _is_simulated(session: BaseRuntimeSession) -> bool:
        """仿真/占位后端: 在其上测出的时延不能当作昆仑 XPU 指标。"""
        return str(session.backend_name).lower() in ("onnxruntime", "stub", "simulated", "cpu")

    @staticmethod
    def _statistics(timings: Sequence[float]) -> Dict[str, float]:
        if not timings:
            return {}
        values = np.asarray(timings, dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p99": float(np.percentile(values, 99)),
        }
