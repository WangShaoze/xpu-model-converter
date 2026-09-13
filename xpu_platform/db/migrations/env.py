# -*- coding: utf-8 -*-
"""Alembic 迁移环境: 连接由 DATABASE_URL(生产 Postgres)或缺省 SQLite 决定。"""
import os
import sys

from alembic import context
from sqlalchemy import create_engine

# 保证可导入 xpu_platform.* (cwd = platform/db 时, 上溯三级到仓库根)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from xpu_platform.db.models import Base  # noqa: E402

config = context.config
target_metadata = Base.metadata


def _database_url() -> str:
    return os.environ.get("DATABASE_URL", config.get_main_option("sqlalchemy.url"))


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(), target_metadata=target_metadata, literal_binds=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = _database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
