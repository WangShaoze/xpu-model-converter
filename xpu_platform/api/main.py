# -*- coding: utf-8 -*-
"""FastAPI 应用入口(Phase 5 §58): 统一前缀 /api/v1, 装配中间件与路由。

启动::

    uvicorn xpu_platform.api.main:app --reload
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from xpu_platform.api.config import get_settings
from xpu_platform.api.middleware import (
    RequestIDMiddleware,
    http_exception_handler,
    unified_error_handler,
)
from xpu_platform.api.routers import (
    artifacts,
    auth,
    events,
    jobs,
    models,
    projects,
    workers,
)
from xpu_platform.db.models.base import Base
from xpu_platform.db.session import create_engine_from_url, create_session_factory
from xpu_platform.worker.queue import InMemoryJobQueue

settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 本地 SQLite 开发库: 未迁移时自动建表; 生产 Postgres 走 Alembic 迁移(§15)
    if settings.database_url.startswith("sqlite"):
        Base.metadata.create_all(create_engine_from_url(settings.database_url))
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="模型转换 Web SaaS API(PyTorch/Paddle → ONNX → 昆仑芯 XPU → Docker 交付包)",
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RequestIDMiddleware)

app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, unified_error_handler)

# 任务队列: 生产可设为 Redis; 此处缺省单进程内存队列供开发/测试
app.state.job_queue = InMemoryJobQueue()
# 数据库会话工厂: API 端点(+SSE 回放)统一从这里取, 测试可用同构 factory 替换
app.state.session_factory = create_session_factory(settings.database_url)

api = settings.api_prefix
app.include_router(auth.router, prefix=api)
app.include_router(projects.router, prefix=api)
app.include_router(models.router, prefix=api)
app.include_router(jobs.router, prefix=api)
app.include_router(artifacts.router, prefix=api)
app.include_router(workers.router, prefix=api)
app.include_router(events.router, prefix=api)


@app.get(api + "/health")
def health():
    return {"status": "ok"}