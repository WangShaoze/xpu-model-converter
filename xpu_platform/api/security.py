# -*- coding: utf-8 -*-
"""安全工具(Phase 5 §46): 密码哈希(pbkdf2, 标准库, 无第三方)+ JWT 签发/校验。

密码格式: ``pbkdf2_sha256$<iterations>$<salt_b64>$<hash_b64>``。仅存哈希, 禁止明文。
"""
import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import jwt

from xpu_platform.api.config import get_settings

_PBKDF2_ITERATIONS = 260_000


def hash_password(password: str) -> str:
    """用 pbkdf2_hmac(sha256, 随机 salt)生成可校验哈希串。"""
    if password is None:
        raise ValueError("password 不能为空")
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        _PBKDF2_ITERATIONS,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(dk).decode("ascii"),
    )


def verify_password(password: str, stored: str) -> bool:
    """校验明文密码与存量哈希, 常量时间比较防时序攻击。"""
    if stored is None or "$" not in stored or not stored.startswith("pbkdf2_sha256"):
        return False
    _algo, iter_str, salt_b64, hash_b64 = stored.split("$", 3)
    try:
        iterations = int(iter_str)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
    except (ValueError, TypeError):
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


def create_access_token(user_id: str, username: str) -> str:
    """签发 JWT, 携带 sub/username, 过期时间来自配置。"""
    s = get_settings()
    now = datetime.now(timezone.utc)
    payload: Dict[str, Any] = {
        "sub": user_id,
        "username": username,
        "iat": now,
        "exp": now + timedelta(minutes=s.jwt_expire_minutes),
    }
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def decode_access_token(token: str) -> Dict[str, Any]:
    """解码并校验 JWT; 过期/非法抛 :class:`jwt.PyJWTError`。"""
    s = get_settings()
    return jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])