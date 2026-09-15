# -*- coding: utf-8 -*-
"""认证路由(Phase 5 §19 / §46 加固): 公钥下发 / 注册 / 登录 / 当前用户。

安全链路:
1. 前端 GET /auth/public-key 取 RSA 公钥;
2. 密码经 RSA-OAEP(SHA-256) 加密后 base64 传输, 请求体不出现明文密码;
3. 后端私钥解密 → pbkdf2 哈希校验;
4. IP 窗口限流 + 账号连续失败锁定, 防暴力破解。
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from xpu_platform.api.crypto import (
    PasswordDecryptError,
    decrypt_password,
    get_public_key_pem,
    validate_password_policy,
)
from xpu_platform.api.dependencies import get_current_user, get_db
from xpu_platform.api.ratelimit import client_ip, get_login_guard
from xpu_platform.api.schemas import (
    LoginRequest,
    PublicKeyOut,
    RegisterRequest,
    SessionUser,
    TokenResponse,
)
from xpu_platform.api.security import create_access_token, hash_password, verify_password
from xpu_platform.db.repositories import UserRepository

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/public-key", response_model=PublicKeyOut)
def public_key():
    """下发密码加密公钥(可缓存; 私钥轮换后前端需重新获取)。"""
    return PublicKeyOut(public_key=get_public_key_pem())


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(body: RegisterRequest, request: Request, db: Session = Depends(get_db)):
    guard = get_login_guard(request)
    guard.check_register(client_ip(request))

    try:
        password = decrypt_password(body.password)
        validate_password_policy(password)
    except PasswordDecryptError as exc:
        raise HTTPException(
            400, detail={"code": "PASSWORD_ENCRYPTION_INVALID", "message": str(exc)}
        )
    except ValueError as exc:
        raise HTTPException(
            400, detail={"code": "WEAK_PASSWORD", "message": str(exc)}
        )

    repo = UserRepository(db)
    if repo.get_by_username(body.username) is not None:
        raise HTTPException(409, detail={"code": "USERNAME_EXISTS", "message": "用户名已存在"})
    if repo.get_by_email(body.email) is not None:
        raise HTTPException(409, detail={"code": "EMAIL_EXISTS", "message": "邮箱已注册"})
    user = repo.create(username=body.username, email=body.email,
                       password_hash=hash_password(password))
    return TokenResponse(access_token=create_access_token(user.id, user.username),
                         user=SessionUser.model_validate(user))


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, request: Request, db: Session = Depends(get_db)):
    guard = get_login_guard(request)
    guard.check_login(client_ip(request), body.username)

    try:
        password = decrypt_password(body.password)
    except PasswordDecryptError as exc:
        # 密文异常不计入账号失败次数, 但拦截请求
        raise HTTPException(
            400, detail={"code": "PASSWORD_ENCRYPTION_INVALID", "message": str(exc)}
        )

    user = UserRepository(db).get_by_username(body.username)
    if user is None or not verify_password(password, user.password_hash):
        guard.record_failure(body.username)
        raise HTTPException(401, detail={"code": "INVALID_CREDENTIALS", "message": "用户名或密码错误"})
    if not user.is_active:
        raise HTTPException(403, detail={"code": "USER_DISABLED", "message": "账号已禁用"})
    guard.record_success(body.username)
    return TokenResponse(access_token=create_access_token(user.id, user.username),
                         user=SessionUser.model_validate(user))


@router.get("/me", response_model=SessionUser)
def me(request: Request, current_user=Depends(get_current_user)):
    del request
    return SessionUser.model_validate(current_user)
