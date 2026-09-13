# -*- coding: utf-8 -*-
"""Artifact 存储抽象(第2轮 Web SaaS 架构, §39 / Phase 2 §14)。

Pipeline 不直接 ``shutil.copy`` 落盘——通过 :class:`ArtifactStore` 记录每个阶段
产物。开发用 :class:`LocalArtifactStore`, 生产可替换为 S3/MinIO 实现, Converter
本身完全感知不到底层存储(通过 :func:`create_artifact_store` 按
``STORAGE_BACKEND`` 切换)。
"""
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from re import match as _re_match
from typing import Dict, List, Optional


class InvalidArtifactKey(ValueError):
    """非法 artifact key(绝对路径 / 路径穿越 / 空 key)。"""


def validate_artifact_key(key: str) -> str:
    """校验并归一化 key: 允许层级路径, 拒绝穿越 / 绝对路径 / 空 key。"""
    if not key or not isinstance(key, str):
        raise InvalidArtifactKey("artifact key 不能为空")
    if "\x00" in key:
        raise InvalidArtifactKey("artifact key 含 NUL 字节: {!r}".format(key))
    normalized = key.replace("\\", "/")
    # Windows 盘符 / 根路径也按绝对路径拒绝
    if normalized.startswith("/") or _re_match(r"^[A-Za-z]:", normalized):
        raise InvalidArtifactKey("绝对路径 key 不允许: {}".format(key))
    pure = PurePosixPath(normalized)
    if not pure.parts or pure.name == "" and len(pure.parts) == 1:
        raise InvalidArtifactKey("空 key 不允许")
    if any(part in ("..", ".") for part in pure.parts):
        raise InvalidArtifactKey("路径穿越 key 不允许: {}".format(key))
    if pure.is_absolute():
        raise InvalidArtifactKey("绝对路径 key 不允许: {}".format(key))
    return pure.as_posix()


class ArtifactStore(ABC):
    """产物存取抽象: put/get/delete/exists。"""

    @abstractmethod
    def put(self, key: str, source_path: str) -> str:
        """把 ``source_path`` 处文件以 ``key`` 存入存储, 返回可寻址的 storage key。"""

    @abstractmethod
    def get(self, key: str) -> str:
        """取回指定 key 的本地路径; key 不存在时抛 KeyError。"""

    @abstractmethod
    def delete(self, key: str) -> None:
        """删除指定 key; 不存在时静默忽略。"""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """key 是否存在。"""


@dataclass
class StoredArtifact:
    """一条已落盘产物记录(§10 artifacts 表的最小形态)。"""
    key: str
    filename: str
    size: int
    sha256: str


class LocalArtifactStore(ArtifactStore):
    """本地磁盘实现: 把文件复制到 ``store_dir/<key>`` 并计算 sha256。"""

    def __init__(self, store_dir: str) -> None:
        self.root = Path(store_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, StoredArtifact] = {}

    def _resolve(self, key: str) -> Path:
        safe = validate_artifact_key(key)
        return self.root.joinpath(*safe.split("/"))

    def put(self, key: str, source_path: str) -> str:
        import shutil

        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError("产物不存在: {}".format(source))
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(source), str(target))
        digest = sha256_of(str(target))
        self._index[key] = StoredArtifact(
            key=key, filename=Path(key).name, size=target.stat().st_size, sha256=digest,
        )
        return str(target)

    def get(self, key: str) -> str:
        target = self._resolve(key)
        if not target.is_file():
            raise KeyError("artifact 不存在: {}".format(key))
        return str(target)

    def delete(self, key: str) -> None:
        target = self._resolve(key)
        if target.is_file():
            target.unlink()
        self._index.pop(key, None)

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def list(self) -> List[StoredArtifact]:
        return list(self._index.values())


def sha256_of(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PresignedUrlStore(ABC):
    """浏览器直传/直下能力(Phase 2 §14): 只有支持 presigned 的存储实现该接口。"""

    @abstractmethod
    def presigned_upload(self, key: str, expires: int = 3600) -> str:
        """返回可 PUT 的预签名上传 URL。"""

    @abstractmethod
    def presigned_download(self, key: str, expires: int = 3600) -> str:
        """返回可 GET 的预签名下载 URL。"""


class MinioArtifactStore(ArtifactStore, PresignedUrlStore):
    """MinIO/S3 实现: 元数据不入 BLOB, 大文件走对象存储(§14/§19/§29)。"""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
        region: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        try:
            from minio import Minio  # 延迟导入: minio 是可选依赖
        except ImportError as err:
            raise ImportError(
                "MinioArtifactStore 需要 minio 依赖, 请安装: pip install 'minio>=7.2'"
            ) from err
        self.endpoint = endpoint
        self.bucket = bucket
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key,
                            secure=secure, region=region)
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)
        self._cache = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="xpu-minio-"))
        self._cache.mkdir(parents=True, exist_ok=True)

    def _stat(self, key: str):
        from minio.error import S3Error
        try:
            return self.client.stat_object(self.bucket, key)
        except S3Error as err:
            if err.code in ("NoSuchKey", "NoSuchObject", "NotFound"):
                return None
            raise

    def put(self, key: str, source_path: str) -> str:
        safe = validate_artifact_key(key)
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError("产物不存在: {}".format(source))
        self.client.fput_object(self.bucket, safe, str(source), content_type=guess_mime(safe))
        return safe

    def get(self, key: str) -> str:
        safe = validate_artifact_key(key)
        if not self.exists(safe):
            raise KeyError("artifact 不存在: {}".format(key))
        local = self._cache / safe
        local.parent.mkdir(parents=True, exist_ok=True)
        self.client.fget_object(self.bucket, safe, str(local))
        return str(local)

    def delete(self, key: str) -> None:
        safe = validate_artifact_key(key)
        try:
            self.client.remove_object(self.bucket, safe)
        except Exception:
            pass  # 不存在时静默忽略

    def exists(self, key: str) -> bool:
        safe = validate_artifact_key(key)
        return self._stat(safe) is not None

    def presigned_upload(self, key: str, expires: int = 3600) -> str:
        safe = validate_artifact_key(key)
        return self.client.presigned_put_object(self.bucket, safe, expires=expires)

    def presigned_download(self, key: str, expires: int = 3600) -> str:
        safe = validate_artifact_key(key)
        return self.client.presigned_get_object(self.bucket, safe, expires=expires)


def guess_mime(filename: str) -> str:
    """按扩展名推断 MIME(§19 上传校验用, 缺省 application/octet-stream)。"""
    mime = {
        ".pt": "application/octet-stream",
        ".pth": "application/octet-stream",
        ".onnx": "application/onnx",
        ".json": "application/json",
        ".yaml": "text/yaml",
        ".yml": "text/yaml",
        ".zip": "application/zip",
        ".log": "text/plain",
        ".txt": "text/plain",
    }
    return mime.get(Path(filename).suffix.lower(), "application/octet-stream")


def create_artifact_store(backend: str = "local", **kwargs) -> ArtifactStore:
    """按 ``STORAGE_BACKEND`` 创建存储实现(§14): local / minio。

    Engine 只依赖 :class:`ArtifactStore` 抽象, 不感知具体后端。
    """
    backend = (backend or "local").lower()
    if backend == "local":
        return LocalArtifactStore(str(kwargs.get("store_dir") or kwargs.get("root") or "artifacts"))
    if backend in ("minio", "s3"):
        return MinioArtifactStore(
            endpoint=kwargs["endpoint"],
            access_key=kwargs["access_key"],
            secret_key=kwargs["secret_key"],
            bucket=kwargs["bucket"],
            secure=bool(kwargs.get("secure", False)),
            region=kwargs.get("region"),
            cache_dir=kwargs.get("cache_dir"),
        )
    raise ValueError("未知存储后端: {} (支持 local / minio)".format(backend))