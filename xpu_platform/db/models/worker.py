# -*- coding: utf-8 -*-
"""XPU Worker 表(§23): 声明设备指纹, 供 Job 调度到真实 XPU。"""
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from xpu_platform.db.models.base import Base


class Worker(Base):
    __tablename__ = "workers"

    worker_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    hostname: Mapped[str] = mapped_column(String(128), default="")
    device_type: Mapped[str] = mapped_column(String(16), default="xpu")  # xpu / cpu
    chip: Mapped[str] = mapped_column(String(64), default="")            # KUNLUNXIN / auto ...
    device_id: Mapped[str] = mapped_column(String(16), default="0")
    sdk_version: Mapped[str] = mapped_column(String(64), default="")
    driver_version: Mapped[str] = mapped_column(String(64), default="")
    status: Mapped[str] = mapped_column(String(16), default="READY")     # READY/BUSY/OFFLINE
    last_heartbeat: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
