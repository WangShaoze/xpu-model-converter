# -*- coding: utf-8 -*-
"""任务队列抽象(Phase 4 §21): API 只入队, Worker 只出队执行。

- :class:`InMemoryJobQueue`: 单进程开发/测试
- :class:`RedisJobQueue`: 生产(Redis List + 处理中集合, 支持 lease/ack/requeue)
"""
from abc import ABC, abstractmethod
from collections import deque
from typing import List, Optional


class JobQueue(ABC):
    @abstractmethod
    def enqueue(self, job_id: str) -> None:
        """把 Job 投进队列。"""

    @abstractmethod
    def dequeue(self, timeout: float = 0) -> Optional[str]:
        """取一个待执行 Job(阻塞可配); 空返回 None。"""

    @abstractmethod
    def ack(self, job_id: str) -> None:
        """执行成功, 从处理集合移除。"""

    @abstractmethod
    def requeue(self, job_id: str) -> None:
        """执行失败, 放回队列重试。"""

    @abstractmethod
    def pending(self) -> int:
        """待执行数量。"""


class InMemoryJobQueue(JobQueue):
    """单进程队列: 开发/测试用。"""

    def __init__(self) -> None:
        self._queue: deque = deque()
        self._processing: set = set()

    def enqueue(self, job_id: str) -> None:
        if job_id not in self._queue and job_id not in self._processing:
            self._queue.append(job_id)

    def dequeue(self, timeout: float = 0) -> Optional[str]:
        if not self._queue:
            return None
        job_id = self._queue.popleft()
        self._processing.add(job_id)
        return job_id

    def ack(self, job_id: str) -> None:
        self._processing.discard(job_id)

    def requeue(self, job_id: str) -> None:
        self._processing.discard(job_id)
        self._queue.append(job_id)

    def pending(self) -> int:
        return len(self._queue)


class RedisJobQueue(JobQueue):
    """Redis List 队列: ``xpu:jobs:queue``(待执行) + ``xpu:jobs:processing``(lease)。"""

    def __init__(self, client=None, queue_key: str = "xpu:jobs:queue",
                 processing_key: str = "xpu:jobs:processing") -> None:
        if client is None:
            try:
                import redis as _redis  # 延迟导入: redis 是可选依赖
            except ImportError as err:
                raise ImportError(
                    "RedisJobQueue 需要 redis 依赖, 请安装: pip install 'redis>=5.0'"
                ) from err
            client = _redis.Redis.from_url("redis://localhost:6379/0")
        self.client = client
        self.queue_key = queue_key
        self.processing_key = processing_key

    def enqueue(self, job_id: str) -> None:
        self.client.lpush(self.queue_key, job_id)

    def dequeue(self, timeout: float = 0) -> Optional[str]:
        # BRPOP: (key, value) 或 None
        item = self.client.brpop([self.queue_key], timeout=timeout) if timeout else \
            self.client.rpop(self.queue_key)
        if not item:
            return None
        raw = item[1] if isinstance(item, (list, tuple)) else item
        # redis-py 默认返回 bytes; 统一在队列边界转 str, 否则 Postgres 会按 bytea
        # 绑定与 varchar 主键比较报错(SQLite 类型亲和宽松, 不会暴露该问题)。
        job_id = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        self.client.sadd(self.processing_key, job_id)
        return job_id

    def ack(self, job_id: str) -> None:
        self.client.srem(self.processing_key, job_id)

    def requeue(self, job_id: str) -> None:
        self.client.srem(self.processing_key, job_id)
        self.client.lpush(self.queue_key, job_id)

    def pending(self) -> int:
        return int(self.client.llen(self.queue_key))
