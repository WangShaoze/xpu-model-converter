# -*- coding: utf-8 -*-
"""API 请求/响应 Pydantic 模型(Phase 5 §26-§29)。"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SessionUser(BaseModel):
    id: str
    username: str
    email: str

    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: SessionUser


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    email: str
    # 此处为前端 RSA-OAEP 加密后的 base64 密文(RSA2048 ≈ 344 字符),
    # 明文密码策略在后端解密后校验, 见 crypto.validate_password_policy
    password: str = Field(min_length=8, max_length=512)


class LoginRequest(BaseModel):
    username: str
    password: str = Field(min_length=8, max_length=512)


class PublicKeyOut(BaseModel):
    public_key: str
    algorithm: str = "RSA-OAEP-256"


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str = ""


class ProjectOut(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class ModelOut(BaseModel):
    id: str
    project_id: str
    name: str
    filename: str
    framework: str
    model_type: str
    status: str
    sha256: str
    size: int
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class JobConfig(BaseModel):
    model_type: Optional[str] = "yolov10"
    task: str = "detection"
    input_size: List[int] = [640, 640]
    batch_size: int = 1
    precision: str = "fp16"
    target: str = "kunlun_xpu"

    model_config = ConfigDict(extra="allow")


class JobCreate(BaseModel):
    project_id: str
    model_id: str
    config: Dict[str, Any] = Field(default_factory=dict)


class StageOut(BaseModel):
    name: str
    status: str
    progress: int
    error_message: str = ""

    model_config = ConfigDict(from_attributes=True)


class JobOut(BaseModel):
    id: str
    project_id: str
    source_model_id: str
    status: str
    pipeline_version: str
    progress: int = 0
    worker_id: str = ""
    error_message: str = ""
    created_at: Optional[datetime] = None
    queued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    stages: List[StageOut] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class ArtifactOut(BaseModel):
    id: str
    job_id: str
    stage: str
    artifact_type: str
    filename: str
    storage_key: str
    size: int
    sha256: str
    mime_type: str
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class EventOut(BaseModel):
    job_id: str
    sequence: int
    event: str
    stage: str = ""
    message: str = ""
    progress: Optional[int] = None
    timestamp: Optional[datetime] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class WorkerOut(BaseModel):
    worker_id: str
    hostname: str
    device_type: str
    chip: str
    device_id: str
    sdk_version: str
    driver_version: str
    status: str
    last_heartbeat: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class ErrorOut(BaseModel):
    code: str
    message: str
    request_id: str = ""