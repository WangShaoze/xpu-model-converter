# -*- coding: utf-8 -*-
"""登录密码应用层加密(§46): RSA-OAEP(SHA-256)。

链路: 前端 GET /auth/public-key 取公钥 → Web Crypto RSA-OAEP 加密密码 →
base64 传输 → 本模块用私钥解密 → pbkdf2 校验。

说明: 应用层加密是 HTTPS/TLS 之外的纵深防御, 防止密码在请求体中以明文出现
(日志/代理/调试工具泄露); 传输层安全仍必须由 TLS 保证。
私钥文件缺失时(仅开发)自动生成并落盘, 生产通过 AUTH_RSA_KEY_PATH 注入。
"""
import base64
import os
import threading
from functools import lru_cache
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from xpu_platform.api.config import get_settings

_RSA_KEY_SIZE = 2048
_OAEP_PADDING = padding.OAEP(
    mgf=padding.MGF1(algorithm=hashes.SHA256()),
    algorithm=hashes.SHA256(),
    label=None,
)

_init_lock = threading.Lock()


class PasswordDecryptError(ValueError):
    """密码密文无法解密(格式损坏/非本服务公钥加密)。"""


def _generate_and_store(path: Path) -> rsa.RSAPrivateKey:
    key = rsa.generate_private_key(public_exponent=65537, key_size=_RSA_KEY_SIZE)
    path.parent.mkdir(parents=True, exist_ok=True)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    # 原子写 + 0600, 避免私钥半写或被其他用户读取
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(pem)
    os.chmod(tmp, 0o600)
    tmp.replace(path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return key


@lru_cache(maxsize=1)
def _private_key() -> rsa.RSAPrivateKey:
    settings = get_settings()
    path = Path(settings.auth_rsa_key_path)
    with _init_lock:
        if path.is_file():
            return serialization.load_pem_private_key(path.read_bytes(), password=None)
        return _generate_and_store(path)


def get_public_key_pem() -> str:
    """返回 SubjectPublicKeyInfo 格式 PEM 公钥(下发给前端)。"""
    pub = _private_key().public_key()
    return pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("ascii")


def decrypt_password(ciphertext_b64: str) -> str:
    """解密前端传来的 base64 RSA-OAEP 密文, 返回明文密码。"""
    if not ciphertext_b64 or len(ciphertext_b64) > 1024:
        raise PasswordDecryptError("密码密文缺失或过长")
    try:
        raw = base64.b64decode(ciphertext_b64, validate=True)
        plain = _private_key().decrypt(raw, _OAEP_PADDING)
    except (ValueError, TypeError) as exc:
        raise PasswordDecryptError("密码解密失败, 请重新获取公钥后重试") from exc
    password = plain.decode("utf-8")
    if not password:
        raise PasswordDecryptError("解密后的密码为空")
    return password


def validate_password_policy(password: str) -> None:
    """密码策略: 8-128 位, 至少包含一个字母和一个数字。"""
    if not (8 <= len(password) <= 128):
        raise ValueError("密码长度需为 8-128 位")
    if not any(c.isalpha() for c in password):
        raise ValueError("密码需至少包含一个字母")
    if not any(c.isdigit() for c in password):
        raise ValueError("密码需至少包含一个数字")
