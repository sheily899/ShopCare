from __future__ import annotations

import inspect
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

WORKING_MAX = 20
COMPRESS_AT = 15
KEEP_AFTER_COMPRESS = 5
TTL_SECONDS = 86400


class MsgRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    role: MsgRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        if not isinstance(data, dict):
            raise TypeError("消息数据必须是对象")
        raw_timestamp = data.get("timestamp", data.get("ts"))
        if isinstance(raw_timestamp, datetime):
            timestamp = raw_timestamp
        elif isinstance(raw_timestamp, str) and raw_timestamp:
            timestamp = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
        else:
            timestamp = datetime.now()
        raw_metadata = data.get("metadata", {})
        metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        return cls(
            role=MsgRole(data["role"]),
            content=str(data.get("content", "")),
            timestamp=timestamp,
            metadata=metadata,
        )


@dataclass
class MemoryContext:
    summary: str = ""
    recent_messages: List[Message] = field(default_factory=list)
    relevant_history: List[Any] = field(default_factory=list)
    user_profile: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _clean(value: Any) -> str:
        text = value if isinstance(value, str) else str(value)
        return text.encode("utf-8", errors="ignore").decode("utf-8")

    @classmethod
    def _history_text(cls, item: Any) -> str:
        if isinstance(item, Message):
            return f"{item.role.value}: {cls._clean(item.content)}"
        if isinstance(item, dict):
            return cls._clean(json.dumps(item, ensure_ascii=False))
        return cls._clean(item)

    def to_prompt_text(self) -> str:
        parts: List[str] = []
        if self.summary:
            parts.append(f"[会话摘要]\n{self._clean(self.summary)}")
        if self.relevant_history:
            history = "\n".join(
                f"- {self._history_text(item)}"
                for item in self.relevant_history[:3]
            )
            parts.append(f"[相关历史]\n{history}")
        if self.user_profile:
            profile = json.dumps(self.user_profile, ensure_ascii=False)
            parts.append(f"[用户画像]\n{self._clean(profile)}")
        if self.recent_messages:
            recent = [
                f"{message.role.value}: {self._clean(message.content)}"
                for message in self.recent_messages[-8:]
            ]
            parts.append("[最近对话]\n" + "\n".join(recent))
        return "\n\n".join(parts)


class MemoryManager:
    """Redis-backed short-term working memory for the current conversation."""

    WORKING_MAX = WORKING_MAX
    COMPRESS_AT = COMPRESS_AT
    KEEP_AFTER_COMPRESS = KEEP_AFTER_COMPRESS
    TTL_SECONDS = TTL_SECONDS

    def __init__(
        self,
        redis_client: Any = None,
        redis_url: Optional[str] = None,
    ) -> None:
        if redis_client is None:
            try:
                import redis.asyncio as redis
            except Exception as exc:  # pragma: no cover - depends on environment
                raise RuntimeError(f"Redis SDK 不可用: {exc}") from exc
            redis_client = redis.from_url(
                redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                decode_responses=True,
            )
        self._redis = redis_client
        self.last_compression_triggered = False
        self.compression_trigger_key: Optional[str] = None

    @staticmethod
    def _wm_key(user_id: str, conv_id: str) -> str:
        return f"wm:{user_id}:{conv_id}"

    async def add_message(
        self,
        user_id: str,
        conv_id: str,
        role: MsgRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Write one message and return whether the compression entry point was reached."""
        key = self._wm_key(str(user_id), str(conv_id))
        try:
            message = Message(
                role=role if isinstance(role, MsgRole) else MsgRole(role),
                content=str(content),
                metadata=dict(metadata or {}),
            )
            payload = message.to_dict()
            # Keep the source Redis representation compatible with the original
            # project while Message.to_dict exposes the explicit field name.
            payload["ts"] = payload.pop("timestamp")
            await self._redis.lpush(
                key,
                json.dumps(payload, ensure_ascii=False),
            )
            await self._redis.expire(key, self.TTL_SECONDS)
            count = await self._redis.llen(key)
            self.last_compression_triggered = count >= self.COMPRESS_AT
            self.compression_trigger_key = key if self.last_compression_triggered else None
            if self.last_compression_triggered:
                logger.info(
                    "工作记忆达到压缩入口: key=%s count=%s keep_after=%s",
                    key,
                    count,
                    self.KEEP_AFTER_COMPRESS,
                )
            return self.last_compression_triggered
        except Exception as exc:
            self.last_compression_triggered = False
            self.compression_trigger_key = None
            logger.error("工作记忆写入失败: %s", exc)
            return False

    async def get_recent_messages(
        self,
        user_id: str,
        conv_id: str,
        limit: int = WORKING_MAX,
    ) -> List[Message]:
        """Read at most 20 messages and restore chronological order."""
        try:
            bounded_limit = min(self.WORKING_MAX, max(0, int(limit)))
        except (TypeError, ValueError):
            bounded_limit = self.WORKING_MAX
        if bounded_limit == 0:
            return []

        key = self._wm_key(str(user_id), str(conv_id))
        try:
            raw_messages = await self._redis.lrange(key, 0, bounded_limit - 1)
            messages: List[Message] = []
            for raw in reversed(raw_messages or []):
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                messages.append(Message.from_dict(json.loads(raw)))
            return messages
        except Exception as exc:
            logger.error("工作记忆读取失败: %s", exc)
            return []

    async def _get_working_memory(
        self, user_id: str, conv_id: str, limit: int = WORKING_MAX
    ) -> List[Message]:
        return await self.get_recent_messages(user_id, conv_id, limit)

    async def close(self) -> None:
        close_method = getattr(self._redis, "aclose", None)
        if close_method is None:
            close_method = getattr(self._redis, "close", None)
        if close_method is not None:
            result = close_method()
            if inspect.isawaitable(result):
                await result
