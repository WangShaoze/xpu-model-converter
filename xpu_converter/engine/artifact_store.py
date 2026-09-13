# -*- coding: utf-8 -*-
"""Artifact 存储抽象(第2轮 Web SaaS 架构, §39)。

Pipeline 不直接 ``shutil.copy`` 落盘——通过 :class:`ArtifactStore` 记录每个阶段
产物。开发用 :class:`LocalArtifactStore`, 生产可替换为 S3/MinIO 实现, Converter
本身完全感知不到底层存储。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from re import match as _re_match
from typing import Dict, List, Optional


class InvalidArtifactKey(ValueError):
    """非法 artifact key(绝对路径 / 路径穿越 / 空 key)。"""


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

    @staticmethod
    def _validate_key(key: str) -> str:
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

    def _resolve(self, key: str) -> Path:
        safe = self._validate_key(key)
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