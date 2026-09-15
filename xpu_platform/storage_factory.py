# -*- coding: utf-8 -*-
"""平台存储工厂(Phase 7): 按 Pydantic Settings 构造 Engine 的 ArtifactStore。

- local: 共享 workspace 卷(容器内 api/worker 挂同一路径), 模型上传与产物走磁盘;
- minio: 产物(Artifacts)走对象存储; 模型上传仍走共享卷(大文件预签名直传为后续迭代)。
"""
from pathlib import Path
from typing import Optional

from xpu_converter.engine import create_artifact_store

from xpu_platform.api.config import Settings, get_settings


def build_artifact_store(settings: Optional[Settings] = None):
    """按 STORAGE_BACKEND 构造存储后端; minio 参数来自 §41 统一配置。"""
    s = settings or get_settings()
    if (s.storage_backend or "local").lower() in ("minio", "s3"):
        return create_artifact_store(
            "minio",
            endpoint=s.minio_endpoint,
            access_key=s.minio_access_key,
            secret_key=s.minio_secret_key,
            bucket=s.minio_bucket,
            secure=s.minio_secure,
            cache_dir=str(Path(s.workspace_root) / "minio-cache"),
        )
    return create_artifact_store(
        "local",
        store_dir=str(Path(s.workspace_root) / "artifacts"),
    )
