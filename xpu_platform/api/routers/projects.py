# -*- coding: utf-8 -*-
"""项目路由(Phase 5 §17): 创建 / 列表 / 详情。"""
from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from xpu_platform.api.dependencies import get_current_user, get_db, require_owned_project
from xpu_platform.api.schemas import ProjectCreate, ProjectOut
from xpu_platform.db.repositories import ProjectRepository

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=ProjectOut, status_code=201)
def create_project(body: ProjectCreate, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    project = ProjectRepository(db).create(owner_id=current_user.id, name=body.name,
                                           description=body.description)
    return ProjectOut.model_validate(project)


@router.get("", response_model=List[ProjectOut])
def list_projects(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    projects = ProjectRepository(db).list_by_owner(current_user.id)
    return [ProjectOut.model_validate(p) for p in projects]


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project_id: str, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    project = require_owned_project(project_id, db, current_user.id)
    return ProjectOut.model_validate(project)


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    project = require_owned_project(project_id, db, current_user.id)
    db.delete(project)
    db.commit()
    return None