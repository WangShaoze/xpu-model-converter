# -*- coding: utf-8 -*-
"""容器内日志转发服务(端口 20000)。

识别服务把日志 POST 到本服务, 由本服务:
1. 把结果图 / 原图上传 MINIO;
2. 把日志转发 KAFKA。

设计约束: MINIO / KAFKA 任一不可用时 **只记录错误**, 绝不影响识别服务。
"""
import json
import os
import threading

from flask import Flask, request

import runtime_logging
from runtime_tools import ExceptionMessage, GetDateTime, WriteLog

app = Flask(__name__)
SETTINGS = runtime_logging.load_settings()

LOG_DIR = os.environ.get("NWAI_LOG_DIR", "/tmp/nwai_log")
PROCESS_LOG = os.path.join(LOG_DIR, "send_log_webserver.log")

_lock = threading.Lock()
_minio_client = None
_kafka_producer = None


def log(message):
    WriteLog(PROCESS_LOG, "[{}] {}".format(GetDateTime()[0], message))


def minio_client():
    """惰性初始化 MinIO 客户端。"""
    global _minio_client
    if _minio_client is not None:
        return _minio_client
    host = SETTINGS.get("MINIO_HOST")
    if not host:
        return None
    try:
        from minio import Minio

        endpoint = "{}:{}".format(host, SETTINGS.get("MINIO_PORT", "9000"))
        _minio_client = Minio(
            endpoint,
            access_key=SETTINGS.get("MINIO_ACCESS_KEY", ""),
            secret_key=SETTINGS.get("MINIO_SECRET_KEY", ""),
            secure=False,
        )
    except Exception as err:
        log("init minio failed: {}".format(ExceptionMessage(err)))
        return None
    return _minio_client


def kafka_producer():
    """惰性初始化 Kafka 生产者。"""
    global _kafka_producer
    if _kafka_producer is not None:
        return _kafka_producer
    host = SETTINGS.get("KAFKA_HOST")
    if not host:
        return None
    try:
        from kafka import KafkaProducer

        _kafka_producer = KafkaProducer(
            bootstrap_servers="{}:{}".format(host, SETTINGS.get("KAFKA_PORT", "9092")),
            value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
            request_timeout_ms=3000,
        )
    except Exception as err:
        log("init kafka failed: {}".format(ExceptionMessage(err)))
        return None
    return _kafka_producer


def upload_file(local_path):
    """上传单个文件到 MINIO, 返回可访问的对象名(失败返回空串)。"""
    if not local_path or not os.path.isfile(local_path):
        return ""
    client = minio_client()
    if client is None:
        return ""
    bucket = SETTINGS.get("MINIO_BUCKET", "alg-log")
    current_datetime, today_strftime, year, month, day = GetDateTime()
    object_name = "{}/{}/{}/{}/{}".format(day, month, year, today_strftime, os.path.basename(local_path))
    try:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
        client.fput_object(bucket, object_name, local_path)
        return object_name
    except Exception as err:
        log("upload {} failed: {}".format(local_path, ExceptionMessage(err)))
        return ""


def forward_kafka(payload):
    """转发日志到 KAFKA。"""
    producer = kafka_producer()
    if producer is None:
        return False
    try:
        producer.send(SETTINGS.get("KAFKA_TOPIC", "alg-log"), payload)
        producer.flush(timeout=3)
        return True
    except Exception as err:
        log("kafka send failed: {}".format(ExceptionMessage(err)))
        return False


@app.route("/log", methods=["POST"])
def receive_log():
    """接收识别服务日志。"""
    try:
        payload = request.get_json(force=True, silent=True) or {}
    except Exception:
        payload = {}
    with _lock:
        original = upload_file(payload.get("original_image"))
        result = upload_file(payload.get("result_image"))
        payload["original_image"] = original
        payload["result_image"] = result
        if not payload.get("data") and not runtime_logging.upload_flag("g_upload_data_empty_minio", False):
            # 无识别结果且未开启空结果上传: 只转发日志, 不保留文件
            payload["original_image"] = ""
            payload["result_image"] = ""
        forward_kafka(payload)
    return {"code": 0, "message": "ok"}, 200


@app.route("/health", methods=["GET"])
def health():
    return {"code": 0, "message": "ok", "service": "send_log_webserver"}, 200


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    port = int(os.environ.get("SEND_LOG_PORT", 20000))
    log("send_log_webserver listening on {}".format(port))
    app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()
