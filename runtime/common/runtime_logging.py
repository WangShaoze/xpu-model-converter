# -*- coding: utf-8 -*-
"""日志上报(MINIO 结果图 + KAFKA 日志)。

识别服务只负责把日志投递给容器内 ``send_log_webserver``(端口 20000), 由它统一
上传 MINIO / 转投 KAFKA。上报失败 **不影响** 识别服务(建设目标交付基线)。
"""
import json
import os
from typing import Any, Dict, List, Optional

from runtime_tools import (ApplicationJsonMode, GetDateTime, LogDetailFormat,
                           SEND_LOG_WEBSERVER_URL, SendLogEnabled)

SETTINGS_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "send_log_settings.json")

DEFAULT_SETTINGS: Dict[str, Any] = {
    "SEND_LOG_ENABLED": True,
    "MINIO_HOST": "",
    "MINIO_PORT": "9000",
    "MINIO_BUCKET": "alg-log",
    "MINIO_ACCESS_KEY": "",
    "MINIO_SECRET_KEY": "",
    "KAFKA_HOST": "",
    "KAFKA_PORT": "9092",
    "KAFKA_TOPIC": "alg-log",
    "setflag_param": {
        "g_upload_data_empty_minio": {
            "describe": "true:识别结果data为[]会上传文件; false:data为[]时不上传;默认false",
            "key": "g_upload_data_empty_minio",
            "value_type": "int",
            "value": False,
        }
    },
}

# docker run -e 可覆盖的上报参数
ENV_OVERRIDES = {
    "MINIO_HOST": "MINIO_HOST",
    "MINIO_PORT": "MINIO_PORT",
    "MINIO_BUCKET": "MINIO_BUCKET",
    "MINIO_ACCESS_KEY": "MINIO_ACCESS_KEY",
    "MINIO_SECRET_KEY": "MINIO_SECRET_KEY",
    "KAFKA_HOST": "KAFKA_HOST",
    "KAFKA_PORT": "KAFKA_PORT",
    "KAFKA_TOPIC": "KAFKA_TOPIC",
    "PROVINCE_CODE": "PROVINCE_CODE",
    "BUREAU_CODE": "BUREAU_CODE",
    "REGION_CODE": "REGION_CODE",
}


def load_settings() -> Dict[str, Any]:
    """读取 ``send_log_settings.json`` 并叠加环境变量。"""
    settings = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as fr:
            settings.update(json.load(fr) or {})
    except Exception:
        pass
    for key in ENV_OVERRIDES:
        value = os.environ.get(key)
        if value is not None and value != "":
            settings[key] = value
    return settings


def log_enabled() -> bool:
    """上报总开关: 需同时满足 ``SEND_LOG_ENABLED`` 与本地开关。"""
    settings = load_settings()
    if not bool(settings.get("SEND_LOG_ENABLED", True)):
        return False
    if not SendLogEnabled():
        return False
    # MINIO 与 KAFKA 地址都为空时自动关闭上报
    return bool(settings.get("MINIO_HOST") or settings.get("KAFKA_HOST"))


def upload_flag(name: str, default: Any = None) -> Any:
    """读取 ``setflag_param`` 中的动态开关(经 ``/setflag`` 接口可在线修改)。"""
    settings = load_settings()
    params = settings.get("setflag_param") or {}
    return (params.get(name) or {}).get("value", default)


def report(task_id: str,
           algorithm: str,
           code: int,
           message: str,
           data: Optional[List[Dict[str, Any]]] = None,
           original_image: Optional[str] = None,
           result_image: Optional[str] = None,
           extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """上报一次识别日志。

    ``original_image`` / ``result_image`` 为容器内文件路径, 由 send_log_webserver
    上传至 MINIO; 未配置上报时直接返回, 不做任何 IO。
    """
    if not log_enabled():
        return {"sent": False, "reason": "log reporting disabled"}

    settings = load_settings()
    current_datetime, today_strftime, year, month, day = GetDateTime()
    payload = LogDetailFormat(algorithm=algorithm, task_id=task_id,
                              detail=data or [], message=message, code=code)
    payload.update({
        "original_image": original_image or "",
        "result_image": result_image or "",
        "province_code": settings.get("PROVINCE_CODE", ""),
        "bureau_code": settings.get("BUREAU_CODE", ""),
        "region_code": settings.get("REGION_CODE", ""),
        "date": current_datetime,
        "timestamp": today_strftime,
        "year": year,
        "month": month,
        "day": day,
    })
    if extra:
        payload.update(extra)

    ok, detail = ApplicationJsonMode(SEND_LOG_WEBSERVER_URL, json.dumps(payload, ensure_ascii=False))
    return {"sent": bool(ok), "reason": detail}
