# -*- coding: utf-8 -*-
"""API 依赖项(Phase 5): DB Session、认证用户解析、资源归属性校验。"""
from typing import Generator, Optional

import jwt
from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from xpu_platform.api.config import get_settings
from xpu_platform.api.security import decode_access_token
from xpu_platform.db.repositories import UserRepository
from xpu_platform.db.session import create_session_factory
from xpu_platform.db.models import User


def get_db() -> Generator[Session, None, None]:
    """每个请求一个 Session(复用 Worker/Engine 的数据库会话工厂)。"""
    SessionLocal = create_session_factory(get_settings().database_url)
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _credentials_exception(detail: str = "认证失败") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    """从 Bearer Token 解析当前用户(§授权)。"""
    request_id = getattr(request.state, "request_id", "")
    if not authorization or not authorization.startswith("Bearer "):
        raise _credentials_exception("缺少 Bearer Token")
    token = authorization[7:].strip()
    if not token:
        raise _credentials_exception("Token 为空")
    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError as err:
        raise _credentials_exception("Token 已过期") from err
    except jwt.PyJWTError as err:
        raise _credentials_exception("Token 无效") from err
    user = UserRepository(db).get(payload.get("sub", ""))
    if user is None:
        raise _credentials_exception("用户不存在")
    if not user.is_active:
        raise _credentials_exception("用户已禁用")
    return user


def require_owned_project(project_id: str, db: Session, owner_id: str):
    """校验项目归属当前用户, 不存在或非本人均返回 404(避免泄露。"""
    from xpu_platform.db.repositories import ProjectRepository

    project = ProjectRepository(db).get_owned(owner_id, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="项目不存在或无权访问")
    return project


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")