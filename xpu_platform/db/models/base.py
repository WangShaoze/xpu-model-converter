# -*- coding: utf-8 -*-
"""SQLAlchemy 2.x DeclarativeBase 与公共列类型(Phase 3 §15)。"""
from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from uuid import uuid4


def uuid_hex() -> str:
    return uuid4().hex


def new_id() -> str:
    return uuid_hex()


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


class IdMixin:
    """通用主键: 32 位 hex 字符串, 与 Engine 的 job_id 语义一致。"""

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
