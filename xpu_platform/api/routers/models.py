# -*- coding: utf-8 -*-
"""模型路由(Phase 5 §18/§46): 上传(校验扩展名/MIME/大小/SHA256) / 列表 / 详情 / 删除。"""
import hashlib
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from xpu_platform.api.config import allowed_extensions, get_settings
from xpu_platform.api.dependencies import get_current_user, get_db, require_owned_project
from xpu_platform.api.schemas import ModelOut
from xpu_platform.db.repositories import AuditLogRepository
from xpu_platform.db.models.model import Model

router = APIRouter(prefix="/projects/{project_id}/models", tags=["models"])

_ALLOWED = {".pt", ".pth", ".onnx"}


def _allowed_mime_types() -> set:
    """§46 MIME 白名单: content_type 前缀匹配(如 application/octet-stream)。"""
    return {m.strip().lower() for m in get_settings().allowed_mime_types.split(",") if m.strip()}


@router.post("", response_model=ModelOut, status_code=201)
def upload_model(
    project_id: str,
    file: UploadFile = File(...),
    name: str = Form(default=""),
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    require_owned_project(project_id, db, current_user.id)
    settings = get_settings()
    ext = Path(file.filename).suffix.lower()
    if ext not in _ALLOWED and ext not in allowed_extensions():
        raise HTTPException(400, detail={"code": "UNSUPPORTED_TYPE",
                                         "message": "不支持的文件类型: {}".format(ext or "(无扩展名)")})
    # §46 MIME 检查: content_type 前缀匹配, 防止扩展名伪造
    content_type = (file.content_type or "").lower()
    if content_type and _allowed_mime_types():
        if not any(content_type.startswith(prefix) for prefix in _allowed_mime_types()):
            raise HTTPException(400, detail={"code": "UNSUPPORTED_MIME",
                                             "message": "不支持的文件 MIME 类型: {}".format(content_type)})

    # 隔离存储: workspace/models/<project>/<uuid>/<filename>, 文件名脱敏防穿越
    safe_filename = Path(file.filename).name or "model{}".format(ext)
    store_dir = Path(settings.workspace_root) / "models" / project_id / uuid.uuid4().hex
    store_dir.mkdir(parents=True, exist_ok=True)
    target = store_dir / safe_filename

    size = 0
    h = hashlib.sha256()
    with open(target, "wb") as out:
        while chunk := file.file.read(1 << 20):
            size += len(chunk)
            if size > settings.max_upload_mb * (1 << 20):
                out.close()
                target.unlink(missing_ok=True)
                raise HTTPException(413, detail={"code": "FILE_TOO_LARGE",
                                                 "message": "文件超过 {} MB 限制".format(settings.max_upload_mb)})
            h.update(chunk)
            out.write(chunk)

    if size == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(400, detail={"code": "EMPTY_FILE", "message": "文件为空"})

    model = Model(
        project_id=project_id,
        name=name or safe_filename,
        filename=safe_filename,
        framework="pytorch",
        model_type="",
        status="READY",
        storage_key=str(target),
        sha256=h.hexdigest(),
        size=size,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    AuditLogRepository(db).record(current_user.id, "MODEL_UPLOADED",
                                  resource_type="model", resource_id=model.id,
                                  metadata={"filename": safe_filename, "size": size})
    return ModelOut.model_validate(model)


@router.get("", response_model=List[ModelOut])
def list_models(project_id: str, db=Depends(get_db), current_user=Depends(get_current_user)):
    require_owned_project(project_id, db, current_user.id)
    rows = db.query(Model).filter(Model.project_id == project_id).order_by(Model.created_at.desc())
    return [ModelOut.model_validate(m) for m in rows]


@router.get("/{model_id}", response_model=ModelOut)
def get_model(project_id: str, model_id: str, db=Depends(get_db), current_user=Depends(get_current_user)):
    require_owned_project(project_id, db, current_user.id)
    model = db.get(Model, model_id)
    if model is None or model.project_id != project_id:
        raise HTTPException(404, detail={"code": "MODEL_NOT_FOUND", "message": "模型不存在"})
    return ModelOut.model_validate(model)