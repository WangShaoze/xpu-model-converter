# -*- coding: utf-8 -*-
"""/auth/* 防暴力破解(§46): IP 固定窗口限流 + 账号连续失败锁定。

单进程内存实现, 适合开发/单副本部署; 多副本生产环境应替换为 Redis 版
(key 设计保持一致: ratelimit:{scope}:{id} / lock:login:{username})。
"""
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple

from fastapi import HTTPException, Request

LOGIN_WINDOW_SECONDS = 60
REGISTER_WINDOW_SECONDS = 3600


def client_ip(request: Request) -> str:
    """取真实客户端 IP: 网关部署时信任 X-Forwarded-For 首段。"""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@dataclass
class LoginGuard:
    """登录/注册限流与账号锁定状态(线程安全)。"""

    login_rate_per_minute: int = 10
    login_max_failures: int = 5
    login_lock_minutes: int = 15
    register_rate_per_hour: int = 10

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    # key -> (窗口起点 unix ts, 已计次数)
    _windows: Dict[str, Tuple[float, int]] = field(default_factory=dict, init=False)
    # username -> (连续失败次数, 锁定截止 unix ts; 0 表示未锁定)
    _failures: Dict[str, Tuple[int, float]] = field(default_factory=dict, init=False)

    def reset(self) -> None:
        with self._lock:
            self._windows.clear()
            self._failures.clear()

    def _hit_window(self, key: str, limit: int, window_seconds: int) -> None:
        now = time.time()
        with self._lock:
            started, count = self._windows.get(key, (now, 0))
            if now - started >= window_seconds:
                started, count = now, 0
            count += 1
            self._windows[key] = (started, count)
            if count > limit:
                retry_after = int(window_seconds - (now - started)) + 1
                raise HTTPException(
                    status_code=429,
                    detail={
                        "code": "RATE_LIMITED",
                        "message": "请求过于频繁, 请稍后再试",
                    },
                    headers={"Retry-After": str(max(retry_after, 1))},
                )

    def check_register(self, ip: str) -> None:
        self._hit_window(
            "register:{}".format(ip), self.register_rate_per_hour, REGISTER_WINDOW_SECONDS
        )

    def check_login(self, ip: str, username: str) -> None:
        # 账号锁定优先判定
        now = time.time()
        with self._lock:
            fails, locked_until = self._failures.get(username, (0, 0.0))
            if locked_until and now < locked_until:
                retry_after = int(locked_until - now) + 1
                raise HTTPException(
                    status_code=429,
                    detail={
                        "code": "ACCOUNT_LOCKED",
                        "message": "连续登录失败次数过多, 账号已临时锁定, 请 {} 秒后再试".format(
                            retry_after
                        ),
                    },
                    headers={"Retry-After": str(retry_after)},
                )
        self._hit_window(
            "login:{}".format(ip), self.login_rate_per_minute, LOGIN_WINDOW_SECONDS
        )

    def record_failure(self, username: str) -> None:
        now = time.time()
        with self._lock:
            fails, _locked = self._failures.get(username, (0, 0.0))
            fails += 1
            locked_until = 0.0
            if fails >= self.login_max_failures:
                locked_until = now + self.login_lock_minutes * 60
            self._failures[username] = (fails, locked_until)

    def record_success(self, username: str) -> None:
        with self._lock:
            self._failures.pop(username, None)


def get_login_guard(request: Request) -> LoginGuard:
    """从 app.state 取全局限流器, 惰性创建(测试可替换/重置)。"""
    guard = getattr(request.app.state, "login_guard", None)
    if guard is None:
        guard = LoginGuard()
        request.app.state.login_guard = guard
    return guard
