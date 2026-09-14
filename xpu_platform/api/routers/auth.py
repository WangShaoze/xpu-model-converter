# -*- coding: utf-8 -*-
"""认证路由(Phase 5 §19): 注册 / 登录 / 当前用户。"""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from xpu_platform.api.dependencies import get_current_user, get_db
from xpu_platform.api.schemas import LoginRequest, RegisterRequest, SessionUser, TokenResponse
from xpu_platform.api.security import create_access_token, hash_password, verify_password
from xpu_platform.db.repositories import UserRepository

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    repo = UserRepository(db)
    if repo.get_by_username(body.username) is not None:
        raise HTTPException(409, detail={"code": "USERNAME_EXISTS", "message": "用户名已存在"})
    if repo.get_by_email(body.email) is not None:
        raise HTTPException(409, detail={"code": "EMAIL_EXISTS", "message": "邮箱已注册"})
    user = repo.create(username=body.username, email=body.email,
                       password_hash=hash_password(body.password))
    return TokenResponse(access_token=create_access_token(user.id, user.username),
                         user=SessionUser.model_validate(user))


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = UserRepository(db).get_by_username(body.username)
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, detail={"code": "INVALID_CREDENTIALS", "message": "用户名或密码错误"})
    if not user.is_active:
        raise HTTPException(403, detail={"code": "USER_DISABLED", "message": "账号已禁用"})
    return TokenResponse(access_token=create_access_token(user.id, user.username),
                         user=SessionUser.model_validate(user))


@router.get("/me", response_model=SessionUser)
def me(request: Request, current_user=Depends(get_current_user)):
    del request
    return SessionUser.model_validate(current_user)