# -*- coding: utf-8 -*-
"""API 配置(Phase 5 §41): 统一 Pydantic Settings, 安全项从环境变量读取, 禁止提交 Secret。"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """API 运行配置。生产通过环境变量 / .env 注入, Secret 不入库不入 Git。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "xpu-model-converter"
    api_prefix: str = "/api/v1"

    database_url: str = "sqlite:///./dev.db"
    redis_url: str = "redis://localhost:6379/0"
    minio_endpoint: str = ""
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "xpu-models"
    storage_backend: str = "local"  # local / minio

    jwt_secret: str = "dev-only-secret-change-me-please-0123456789abcdef"  # ≥32B 生产必须覆盖
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24h

    workspace_root: str = "./worker-workspace"
    log_level: str = "INFO"

    # 上传限制(§46): 大小与扩展名白名单
    max_upload_mb: int = 200
    allowed_extensions: str = ".pt,.pth,.onnx"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def allowed_extensions() -> set:
    return {e.strip().lower() for e in get_settings().allowed_extensions.split(",") if e.strip()}