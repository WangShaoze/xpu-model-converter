# -*- coding: utf-8 -*-
"""JobEvent 表(§9/§56): 事件持久化, sequence 同一 Job 内严格递增, 供 SSE 回放。"""
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from xpu_platform.db.models.base import Base, IdMixin


class JobEvent(IdMixin, Base):
    __tablename__ = "job_events"

    job_id: Mapped[str] = mapped_column(String(32), ForeignKey("conversion_jobs.id"), index=True)
    stage: Mapped[str] = mapped_column(String(64), default="")
    event: Mapped[str] = mapped_column(String(64), index=True)  # stage_started/stage_progress/log/...
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    sequence: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    progress: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    message: Mapped[str] = mapped_column(Text, default="")
    data: Mapped[Dict[str, Any]] = mapped_column(JSON().with_variant(JSONB(), "postgresql"), default=dict)
