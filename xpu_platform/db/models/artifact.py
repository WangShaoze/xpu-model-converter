# -*- coding: utf-8 -*-
"""Artifact 表(§11): 产物元数据, 文件本体在 MinIO/S3 storage_key。"""
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from xpu_platform.db.models.base import Base, IdMixin


class Artifact(IdMixin, Base):
    __tablename__ = "artifacts"

    job_id: Mapped[str] = mapped_column(String(32), ForeignKey("conversion_jobs.id"), index=True)
    stage: Mapped[str] = mapped_column(String(64), default="")
    artifact_type: Mapped[str] = mapped_column(String(64), default="")
    filename: Mapped[str] = mapped_column(String(255))
    storage_key: Mapped[str] = mapped_column(String(512), index=True)
    size: Mapped[int] = mapped_column(BigInteger, default=0)
    sha256: Mapped[str] = mapped_column(String(64), default="")
    mime_type: Mapped[str] = mapped_column(String(64), default="application/octet-stream")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
