# -*- coding: utf-8 -*-
"""模型表(§18): 只存元数据, 二进制在 MinIO storage_key 指向。"""
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from xpu_platform.db.models.base import Base, IdMixin


class Model(IdMixin, Base):
    __tablename__ = "models"

    project_id: Mapped[str] = mapped_column(String(32), ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    filename: Mapped[str] = mapped_column(String(255))
    framework: Mapped[str] = mapped_column(String(32), default="pytorch")
    model_type: Mapped[str] = mapped_column(String(64), default="yolov10")
    status: Mapped[str] = mapped_column(String(16), default="UPLOADING")  # UPLOADING/READY/INVALID/DELETED
    storage_key: Mapped[str] = mapped_column(String(512), default="")
    sha256: Mapped[str] = mapped_column(String(64), default="")
    size: Mapped[int] = mapped_column(BigInteger, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
