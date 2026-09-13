# -*- coding: utf-8 -*-
"""转换任务表(§7): 状态机语义与 Engine JobStatus 对齐。"""
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from xpu_platform.db.models.base import Base, IdMixin


class ConversionJob(IdMixin, Base):
    __tablename__ = "conversion_jobs"

    project_id: Mapped[str] = mapped_column(String(32), ForeignKey("projects.id"), index=True)
    source_model_id: Mapped[str] = mapped_column(String(32), ForeignKey("models.id"))
    status: Mapped[str] = mapped_column(String(20), default="CREATED")  # CREATED/QUEUED/RUNNING/SUCCESS/FAILED/CANCELLED
    pipeline_version: Mapped[str] = mapped_column(String(32), default="10-stage")
    config: Mapped[Dict[str, Any]] = mapped_column(JSON().with_variant(JSONB(), "postgresql"), default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    queued_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    error_code: Mapped[str] = mapped_column(String(64), default="")
    error_message: Mapped[str] = mapped_column(Text, default="")
    worker_id: Mapped[str] = mapped_column(String(64), default="")
