# -*- coding: utf-8 -*-
"""设备池: 单机多卡多进程。

gunicorn 每个 worker 绑定一张 XPU 卡(``pre_fork`` 取卡, ``child_exit`` 还卡),
``DEVICE=auto`` 时按探测到的卡数自动决定并发; 探测不到卡时退化为 CPU 单进程。
"""
import os
import threading
from typing import List, Optional

from runtime_backend import xpu_device_count


class DevicePool:
    """线程安全的设备号分配池。"""

    def __init__(self, devices: Optional[List[int]] = None, device_mode: str = "auto") -> None:
        self.device_mode = (device_mode or "auto").lower()
        self._lock = threading.Lock()
        self._devices = list(devices if devices is not None else self._detect_devices())
        self._available = list(self._devices)

    def _detect_devices(self) -> List[int]:
        if self.device_mode == "cpu":
            return []
        count = xpu_device_count()
        if count <= 0:
            return []
        gpu_id = os.environ.get("GPU_ID")
        if gpu_id not in (None, "", "auto"):
            ids = [int(item) for item in str(gpu_id).replace(",", " ").split() if item.strip().isdigit()]
            return [item for item in ids if 0 <= item < count] or list(range(count))
        return list(range(count))

    @property
    def devices(self) -> List[int]:
        return list(self._devices)

    @property
    def device_count(self) -> int:
        return len(self._devices)

    def acquire(self) -> Optional[int]:
        """取一张卡; 无卡时返回 ``None``(表示 CPU 模式)。"""
        with self._lock:
            if not self._available:
                return None
            return self._available.pop(0)

    def release(self, device: Optional[int]) -> None:
        if device is None:
            return
        with self._lock:
            if device not in self._available:
                self._available.append(device)

    def worker_count(self, configured: int = 1) -> int:
        """按卡数修正并发: 有卡时不超过卡数, 无卡时退化为 1。"""
        configured = max(1, int(configured or 1))
        if not self._devices:
            return 1
        return min(configured, len(self._devices))
