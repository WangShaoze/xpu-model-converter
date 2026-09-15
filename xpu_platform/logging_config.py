# -*- coding: utf-8 -*-
"""统一结构化日志(§44): JSON 格式输出, 含 timestamp/level/service/request_id/job_id/stage。

API / Worker / Engine 共用同一日志格式, 通过 service 字段区分来源。
生产环境用 JSON 便于 ELK/Loki 采集; 开发环境可设 LOG_LEVEL=DEBUG 看明文。

用法:
    from xpu_platform.logging_config import setup_logging
    setup_logging("api")          # API 进程
    setup_logging("worker")       # Worker 进程
    setup_logging()               # 通用(默认 service=platform)

之后 logging.getLogger("xpu.*") 的输出即带 service 等结构化字段。
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """JSON 单行日志格式器。

    标准 logging extra 字段直接并入 JSON body;
    缺失的上下文字段(request_id/job_id/stage)以空字符串占位, 保证 schema 稳定。
    """

    # §44 必须出现的字段
    REQUIRED_FIELDS = ("timestamp", "level", "service", "message",
                        "request_id", "job_id", "stage")

    def __init__(self, service: str = "platform"):
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        # 基础字段
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "service": getattr(record, "service", self._service),
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", ""),
            "job_id": getattr(record, "job_id", ""),
            "stage": getattr(record, "stage", ""),
        }
        # 异常信息
        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = self.formatException(record.exc_info)

        # 合并用户自定义 extra 字段(排除标准 LogRecord 属性)
        standard_attrs = set(dir(record))
        for key, value in record.__dict__.items():
            if key not in standard_attrs and key not in entry:
                try:
                    json.dumps(value)
                    entry[key] = value
                except (TypeError, ValueError):
                    entry[key] = str(value)

        return json.dumps(entry, ensure_ascii=False)


def setup_logging(service: str = "platform", level: str = "") -> None:
    """配置 root logger 使用 JSONFormatter, 输出到 stdout。

    在 API lifespan / Worker main 启动时调用一次即可。
    """
    log_level = level or os.environ.get("LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()
    root.setLevel(log_level)

    # 清除已有 handler, 避免 uvicorn 默认 handler 重复输出
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter(service=service))
    root.addHandler(handler)

    # 子 logger 继承 root 的 handler, 不额外传播
    for name in ("xpu.api", "xpu.worker", "xpu.engine", "uvicorn", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
