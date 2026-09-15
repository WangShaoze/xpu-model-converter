# -*- coding: utf-8 -*-
"""审计日志表(§30): 关键动作留痕, 供管理员审计。"""
from datetime import datetime
from typing import Any, Dict

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from xpu_platform.db.models.base import Base, IdMixin


class AuditLog(IdMixin, Base):
    __tablename__ = "audit_logs"

    user_id: Mapped[str] = mapped_column(String(32), ForeignKey("users.id"), index=True)
    action: Mapped[str] = mapped_column(String(64), index=True)  # MODEL_UPLOADED/JOB_CREATED/...
    resource_type: Mapped[str] = mapped_column(String(32), default="")
    resource_id: Mapped[str] = mapped_column(String(64), default="")
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    ip: Mapped[str] = mapped_column(String(64), default="")
    # 属性名避开 Declarative 保留字 metadata, 数据库列名仍为 metadata(§30)
    meta: Mapped[Dict[str, Any]] = mapped_column(
        "metadata", JSON().with_variant(JSONB(), "postgresql"), default=dict)
