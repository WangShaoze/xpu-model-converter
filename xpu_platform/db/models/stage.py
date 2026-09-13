# -*- coding: utf-8 -*-
"""JobStage 表(§8): 每个 Job 的 10 个阶段状态与产物对账。"""
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from xpu_platform.db.models.base import Base, IdMixin


class JobStage(IdMixin, Base):
    __tablename__ = "job_stages"

    job_id: Mapped[str] = mapped_column(String(32), ForeignKey("conversion_jobs.id"), index=True)
    stage_name: Mapped[str] = mapped_column(String(64))
    stage_order: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="PENDING")  # PENDING/RUNNING/SUCCESS/FAILED/SKIPPED/CANCELLED
    progress: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    error_code: Mapped[str] = mapped_column(String(64), default="")
    error_message: Mapped[str] = mapped_column(Text, default="")
    metrics: Mapped[Dict[str, Any]] = mapped_column(JSON().with_variant(JSONB(), "postgresql"), default=dict)
