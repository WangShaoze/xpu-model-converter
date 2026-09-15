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
    # 任务队列后端: memory(单进程开发/测试) / redis(Docker Compose/生产)
    queue_backend: str = "memory"
    minio_endpoint: str = ""
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "xpu-models"
    minio_secure: bool = False
    storage_backend: str = "local"  # local / minio

    jwt_secret: str = "dev-only-secret-change-me-please-0123456789abcdef"  # ≥32B 生产必须覆盖
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24h

    workspace_root: str = "./worker-workspace"
    log_level: str = "INFO"

    # 上传限制(§46): 大小与扩展名白名单
    max_upload_mb: int = 200
    allowed_extensions: str = ".pt,.pth,.onnx"
    # MIME 白名单(§46): content_type 前缀匹配, 防止扩展名伪造
    allowed_mime_types: str = "application/octet-stream,application/x-python-code,application/zip,model/pytorch,model/onnx"

    # Job 执行超时(§46): wall-clock 秒, 0 表示不限制
    job_timeout_seconds: int = 1800  # 30 分钟

    # 登录密码传输加密: RSA-OAEP(SHA-256) 应用层加密(HTTPS 之外的纵深防御)。
    # 开发环境私钥文件缺失时自动生成; 生产必须通过 AUTH_RSA_KEY_PATH 注入持久化密钥,
    # 且该文件权限 0600、不入 Git。
    auth_rsa_key_path: str = "./dev-auth-key.pem"

    # /auth/* 防暴力破解: 同一 IP 登录请求窗口上限; 同账号连续失败锁定
    login_rate_per_minute: int = 10
    login_max_failures: int = 5
    login_lock_minutes: int = 15
    register_rate_per_hour: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()


def allowed_extensions() -> set:
    return {e.strip().lower() for e in get_settings().allowed_extensions.split(",") if e.strip()}