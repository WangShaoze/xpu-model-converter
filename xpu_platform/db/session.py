# -*- coding: utf-8 -*-
"""数据库连接与会话(Phase 3 §15 / §41)。

PostgreSQL 生产: ``DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/xpu``
开发/测试: 缺省 SQLite 文件, 测试可传入 ``sqlite+pysqlite://`` 内存库。
"""
import os
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xpu_platform.db.models.base import Base


def create_engine_from_url(database_url: Optional[str] = None, **kwargs):
    """按 URL 创建 Engine(默认读环境变量 DATABASE_URL, 缺省本地 SQLite)。"""
    url = database_url or os.environ.get("DATABASE_URL", "sqlite:///./dev.db")
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(url, connect_args=connect_args, **kwargs)


def create_session_factory(database_url: Optional[str] = None) -> sessionmaker:
    engine = create_engine_from_url(database_url)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖: 每次请求一个 Session。"""
    SessionLocal = create_session_factory()
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
