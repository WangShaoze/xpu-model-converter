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
    """单次基准测试结果。"""

    backend: str = ""
    device: str = "auto"
    iterations: int = 0
    warmup: int = 0
    latency_ms: Dict[str, float] = field(default_factory=dict)
    throughput_fps: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "device": self.device,
            "iterations": self.iterations,
            "warmup": self.warmup,
            "latency_ms": dict(self.latency_ms),
            "throughput_fps": self.throughput_fps,
            "notes": list(self.notes),
        }

    def summary(self) -> str:
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
    ) -> BenchmarkResult:
        feeds = self._build_feeds(session, sample)
        result = BenchmarkResult(
            backend=session.backend_name,
            device=device,
            iterations=self.iterations,
            warmup=self.warmup,
        )

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
        if self._is_simulated(session):
            result.notes.append(
                "当前会话为 onnxruntime 仿真, 性能数据不代表昆仑 XPU 真实指标"
            )
        return result

    # ------------------------------------------------------------------ 内部
    @staticmethod
    def _build_feeds(session: BaseRuntimeSession, sample: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if sample:
            return dict(sample)
        shapes = session.input_shapes() or {}
        feeds: Dict[str, Any] = {}
        for name in session.input_names:
            shape = shapes.get(name)
            static = [1 if (d is None or int(d) < 0) else int(d) for d in (shape or [1])]
            feeds[name] = np.zeros(static, dtype=np.float32)
        return feeds

    @staticmethod
    def _is_simulated(session: BaseRuntimeSession) -> bool:
        return session.backend_name == "onnxruntime"

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
