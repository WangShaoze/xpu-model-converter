# -*- coding: utf-8 -*-
"""User / Project 仓储(Phase 5): 认证与资源归属性。"""
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from xpu_platform.db.models.project import Project
from xpu_platform.db.models.user import User


class UserRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, user_id: str) -> Optional[User]:
        return self._session.get(User, user_id)

    def get_by_username(self, username: str) -> Optional[User]:
        return self._session.scalar(
            select(User).where(User.username == username)
        )

    def get_by_email(self, email: str) -> Optional[User]:
        return self._session.scalar(
            select(User).where(User.email == email)
        )

    def create(self, *, username: str, email: str, password_hash: str) -> User:
        user = User(username=username, email=email, password_hash=password_hash)
        self._session.add(user)
        self._session.commit()
        self._session.refresh(user)
        return user


class ProjectRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, *, owner_id: str, name: str, description: str = "") -> Project:
        project = Project(owner_id=owner_id, name=name, description=description)
        self._session.add(project)
        self._session.commit()
        self._session.refresh(project)
        return project

    def get(self, project_id: str) -> Optional[Project]:
        return self._session.get(Project, project_id)

    def get_owned(self, owner_id: str, project_id: str) -> Optional[Project]:
        return self._session.scalar(
            select(Project).where(Project.id == project_id, Project.owner_id == owner_id)
        )

    def list_by_owner(self, owner_id: str, limit: int = 100) -> List[Project]:
        stmt = (
            select(Project)
            .where(Project.owner_id == owner_id)
            .order_by(Project.created_at.desc())
            .limit(limit)
        )
        return list(self._session.scalars(stmt))