# -*- coding: utf-8 -*-
"""API 路由导出。"""
from xpu_platform.api.routers import (
    artifacts,
    auth,
    events,
    jobs,
    models,
    projects,
    workers,
)

__all__ = ["auth", "projects", "models", "jobs", "artifacts", "workers", "events"]