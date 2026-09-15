# -*- coding: utf-8 -*-
"""API 中间件(Phase 5 §43/§44): X-Request-ID、结构化日志、错误拦截。

结构化日志字段: timestamp/level/service/request_id/method/path/status/duration_ms。
"""
import logging
import time
import uuid
from typing import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger("xpu.api")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """为每个请求生成/沿用 X-Request-ID 并注入 request.state。"""

    async def dispatch(self, request: Request, call_next: Callable):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:  # 交回全局异常处理器
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "http",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": "exception",
                },
            )
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "http",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        return response


async def unified_error_handler(request: Request, exc: Exception):
    """兜底通用异常 → 统一错误格式(§42); HTTPException 由 FastAPI 内建处理。"""
    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_ERROR",
            "message": "服务器内部错误",
            "request_id": getattr(request.state, "request_id", ""),
        },
    )


async def http_exception_handler(request: Request, exc):
    """把 FastAPI 的 HTTPException 包装成统一错误格式。"""
    request_id = getattr(request.state, "request_id", "")
    detail = getattr(exc, "detail", "请求错误")
    code = detail if isinstance(detail, str) else "REQUEST_ERROR"
    body = {"code": code, "message": detail if isinstance(detail, str) else str(detail)}
    if isinstance(detail, dict):  # 允许业务代码自定义 {"code","message"}
        body = detail
    body.setdefault("request_id", request_id)
    return JSONResponse(status_code=exc.status_code, content=body)