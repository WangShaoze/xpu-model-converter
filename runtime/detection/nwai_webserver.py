# -*- coding: utf-8 -*-
"""检测服务 HTTP 接口(公共 Runtime)。

接口与"南网人工智能组件"规范一致:

======================  ==================================================
路由                    说明
======================  ==================================================
``POST /predict``       识别接口, 支持 multipart / base64 / urlpath 三种方式
``POST /predict_image`` 返回结果图(base64 或二进制)
``GET  /health``        健康检查
``POST /setflag``       在线修改动态开关
``GET  /result_image/<filename>``  拉取结果图
======================  ==================================================
"""
import base64
import json
import os
import uuid

from flask import Flask, request, send_file

import nwai_logging
import nwai_settings as settings
from nwai_detector import detect
from nwai_tools import (RETURN_CODE_INTERNAL_ERROR, RETURN_CODE_OK, RETURN_CODE_PARAM_ERROR,
                        ExceptionMessage, GetResultInfo, Jsonify, LogDetailFormat, WriteLog)
from nwai_utils import NwaiUtils, result_image_path

app = Flask(__name__)

_UTILS = NwaiUtils()


def _log(message):
    WriteLog(settings.PROCESS_LOG_FILE, "[runtime] {}".format(message))


def _respond(result_info, status=200):
    return Jsonify(result_info), status, {"Content-Type": "application/json;charset=utf-8"}


def _algorithm_fields():
    return {
        "algorithm": settings.ALGORITHM,
        "algorithm_name": settings.ALGORITHM_NAME,
        "algorithm_version": settings.ALGORITHM_VERSION,
        "algorithm_type": settings.ALGORITHM_TYPE,
        "algorithm_app_type": settings.ALGORITHM_APP_TYPE,
        "component_code": settings.COMPONENT_CODE,
        "template_version": settings.TEMPLATE_VERSION,
    }


# --------------------------------------------------------------------- 请求解析
def _decode_image(data: bytes):
    import cv2
    import numpy as np

    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def _fetch_url(url: str) -> bytes:
    import urllib3

    urllib3.disable_warnings()
    http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=3.0, read=15.0))
    response = http.request("GET", url)
    if response.status != 200:
        raise RuntimeError("下载图片失败, http status={}".format(response.status))
    return response.data


def _decode_base64(text: str) -> bytes:
    if text.startswith("data:") and "," in text:
        text = text.split(",", 1)[1]
    return base64.b64decode(text)


def get_request_data():
    """解析请求, 返回 ``(image_bgr, params, err_message)``。"""
    params = {}
    for key in ("code", "ai_param"):
        value = request.form.get(key)
        if value:
            params[key] = value

    upload = request.files.get("image") or request.files.get("file")
    if upload is not None:
        image = _decode_image(upload.read())
        if image is None:
            return None, params, "上传图片无法解析"
        return image, params, ""

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        try:
            payload = json.loads(request.get_data(as_text=True) or "{}")
        except Exception:
            payload = {}
    if params.get("ai_param") and isinstance(payload.get("ai_param"), dict):
        payload.update(payload.pop("ai_param"))
    if payload.get("code") is not None:
        params["code"] = payload["code"]

    url = payload.get("urlpath") or payload.get("url")
    if url:
        try:
            image = _decode_image(_fetch_url(str(url)))
        except Exception as err:
            return None, params, "urlpath 下载失败: {}".format(ExceptionMessage(err))
        if image is None:
            return None, params, "urlpath 图片无法解析"
        return image, params, ""

    text = payload.get("base64") or payload.get("image_base64") or payload.get("image")
    if text:
        try:
            image = _decode_image(_decode_base64(str(text)))
        except Exception as err:
            return None, params, "base64 解码失败: {}".format(ExceptionMessage(err))
        if image is None:
            return None, params, "base64 图片无法解析"
        return image, params, ""

    return None, params, "请求未携带图片(image/file/urlpath/base64)"


def _resolve_thresholds(params):
    conf, iou = settings.CONF_THRES, settings.IOU_THRES
    ai_param = params.get("ai_param")
    if isinstance(ai_param, str):
        try:
            ai_param = json.loads(ai_param)
        except Exception:
            ai_param = None
    if isinstance(ai_param, dict):
        if ai_param.get("conf_thres") is not None:
            conf = float(ai_param["conf_thres"])
        if ai_param.get("iou_thres") is not None:
            iou = float(ai_param["iou_thres"])
    return conf, iou


# --------------------------------------------------------------------- 路由
@app.route("/predict", methods=["POST"])
def predict():
    task_id = str(uuid.uuid4())
    image, params, err_message = get_request_data()
    if image is None:
        return _respond(GetResultInfo(task_id=task_id, code=RETURN_CODE_PARAM_ERROR,
                                      message=err_message, data=None, **_algorithm_fields()))

    try:
        conf, iou = _resolve_thresholds(params)
        raw = detect(image, conf_thres=conf, iou_thres=iou)
        data = _UTILS.dest_to_outputformat(raw)
        if isinstance(data, Exception):
            return _respond(GetResultInfo(task_id=task_id, code=RETURN_CODE_INTERNAL_ERROR,
                                          message=str(data), data=None, **_algorithm_fields()))

        original_path, result_path = _persist_images(image, data, task_id)
        nwai_logging.report(task_id=task_id, algorithm=settings.ALGORITHM, code=RETURN_CODE_OK,
                               message="识别正常", data=data, original_image=original_path,
                               result_image=result_path)
        _log(LogDetailFormat(algorithm=settings.ALGORITHM, task_id=task_id,
                             detail=data, message="识别正常"))
        return _respond(GetResultInfo(task_id=task_id, code=RETURN_CODE_OK,
                                      message="识别正常", data=data, **_algorithm_fields()))
    except Exception as err:
        _log("predict failed: {}".format(ExceptionMessage(err)))
        return _respond(GetResultInfo(task_id=task_id, code=RETURN_CODE_INTERNAL_ERROR,
                                      message=ExceptionMessage(err), data=None, **_algorithm_fields()))


@app.route("/predict_image", methods=["POST"])
def predict_image():
    """识别并返回结果图(base64)。"""
    task_id = str(uuid.uuid4())
    image, params, err_message = get_request_data()
    if image is None:
        return _respond(GetResultInfo(task_id=task_id, code=RETURN_CODE_PARAM_ERROR,
                                      message=err_message, data=None, **_algorithm_fields()))
    try:
        conf, iou = _resolve_thresholds(params)
        raw = detect(image, conf_thres=conf, iou_thres=iou)
        data = _UTILS.dest_to_outputformat(raw)
        if isinstance(data, Exception):
            return _respond(GetResultInfo(task_id=task_id, code=RETURN_CODE_INTERNAL_ERROR,
                                          message=str(data), data=None, **_algorithm_fields()))
        _, result_path = _persist_images(image, data, task_id)
        if not result_path or not os.path.isfile(result_path):
            return _respond(GetResultInfo(task_id=task_id, code=RETURN_CODE_INTERNAL_ERROR,
                                          message="结果图生成失败", data=data, **_algorithm_fields()))
        with open(result_path, "rb") as fr:
            encoded = base64.b64encode(fr.read()).decode("utf-8")
        result_info = GetResultInfo(task_id=task_id, code=RETURN_CODE_OK, message="识别正常",
                                    data=data, **_algorithm_fields())
        result_info["result_image"] = encoded
        return _respond(result_info)
    except Exception as err:
        return _respond(GetResultInfo(task_id=task_id, code=RETURN_CODE_INTERNAL_ERROR,
                                      message=ExceptionMessage(err), data=None, **_algorithm_fields()))


@app.route("/health", methods=["GET"])
def health():
    return _respond(GetResultInfo(code=RETURN_CODE_OK, message="服务正常",
                                  data=[{"algorithm": settings.ALGORITHM,
                                         "version": settings.ALGORITHM_VERSION,
                                         "task": settings.TASK,
                                         "device": settings.DEVICE}]))


@app.route("/setflag", methods=["POST"])
def set_flag():
    """在线修改动态开关(落盘到 setflag.json, 不修改交付包内配置文件)。"""
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict) or not payload:
        return _respond(GetResultInfo(code=RETURN_CODE_PARAM_ERROR, message="请求体为空",
                                      data=None))
    path = settings.SETFLAG_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    current = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fr:
                current = json.load(fr) or {}
        except Exception:
            current = {}
    current.update(payload)
    with open(path, "w", encoding="utf-8") as fw:
        json.dump(current, fw, ensure_ascii=False, indent=2)
    return _respond(GetResultInfo(code=RETURN_CODE_OK, message="设置成功", data=current))


@app.route("/result_image/<path:filename>", methods=["GET"])
def result_image(filename):
    path = result_image_path(filename, settings.RESULT_DIR)
    if not path or not os.path.isfile(path):
        return _respond(GetResultInfo(code=RETURN_CODE_PARAM_ERROR, message="结果图不存在", data=None))
    return send_file(path, mimetype="image/jpeg")


# --------------------------------------------------------------------- 内部
def _persist_images(image, data, task_id):
    """保存原图与结果图, 供 MINIO 上报使用。"""
    try:
        import cv2
        import numpy as np

        os.makedirs(settings.RESULT_DIR, exist_ok=True)
        original_path = os.path.join(settings.RESULT_DIR, "{}_src.jpg".format(task_id))
        result_path = os.path.join(settings.RESULT_DIR, "{}_result.jpg".format(task_id))
        cv2.imwrite(original_path, image)
        ok, err = _UTILS.draw_detections(image, list(data or []), result_path)
        if not ok:
            _log("draw detections failed: {}".format(err))
            result_path = ""
        return original_path, result_path
    except Exception as err:
        _log("persist images failed: {}".format(ExceptionMessage(err)))
        return "", ""


def main():
    """调试模式: 单进程 Flask(等价 ``NWAI_FLASK_ONLY=1``)。"""
    settings.os_makedirs()
    _log("flask only mode listening on {}".format(settings.WEB_PORT))
    app.run(host="0.0.0.0", port=settings.WEB_PORT, threaded=True)


if __name__ == "__main__":
    main()
