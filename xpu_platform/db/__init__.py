# -*- coding: utf-8 -*-
"""PostgreSQL 持久化层(Phase 3, §15)。

PostgreSQL 只保存 metadata, 不保存模型二进制; 大文件走 MinIO/S3。
"""
from xpu_platform.db.session import (
    Base,
    create_engine_from_url,
    create_session_factory,
    get_db,
)

__all__ = ["Base", "create_engine_from_url", "create_session_factory", "get_db"]
