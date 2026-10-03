from __future__ import annotations

import inspect
import hashlib
import json
import logging
import os
import time
import uuid
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

WORKING_MAX = 20
COMPRESS_AT = 15
KEEP_AFTER_COMPRESS = 5
TTL_SECONDS = 86400
HISTORY_TOP_K = 5
SUMMARY_MAX_CHARS = 800
EPISODIC_FULL_TEXT_MAX_CHARS = 500


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
    HISTORY_TOP_K = HISTORY_TOP_K
    SUMMARY_MAX_CHARS = SUMMARY_MAX_CHARS
    EPISODIC_FULL_TEXT_MAX_CHARS = EPISODIC_FULL_TEXT_MAX_CHARS

    def __init__(
        self,
        redis_client: Any = None,
        redis_url: Optional[str] = None,
        llm_client: Any = None,
        episodic_collection: Any = None,
        chroma_client: Any = None,
        model: Optional[str] = None,
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
        self._llm_client = llm_client
        self._model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        self._episodic = episodic_collection
        self._chroma_client = chroma_client
        self.last_compression_triggered = False
        self.compression_trigger_key: Optional[str] = None

    @staticmethod
    def _wm_key(user_id: str, conv_id: str) -> str:
        return f"wm:{user_id}:{conv_id}"

    @staticmethod
    def _summary_key(user_id: str, conv_id: str) -> str:
        return f"summary:{user_id}:{conv_id}"

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

    @staticmethod
    def _format_messages(messages: List[Message]) -> str:
        return "\n".join(f"{message.role.value}: {message.content}" for message in messages)

    async def _get_llm_client(self) -> Any:
        if self._llm_client is not None:
            return self._llm_client
        try:
            from anthropic import AsyncAnthropic
        except Exception as exc:  # pragma: no cover - dependency/environment path
            raise RuntimeError(f"Anthropic SDK 不可用: {exc}") from exc
        self._llm_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return self._llm_client

    @staticmethod
    def _extract_llm_text(response: Any) -> str:
        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, (list, tuple)):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    return block["text"]
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    return text
        text = getattr(response, "text", None)
        return text if isinstance(text, str) else ""

    async def _generate_summary(
        self, old_messages: List[Message], llm_client: Any = None
    ) -> str:
        client = llm_client or await self._get_llm_client()
        response = await client.messages.create(
            model=self._model,
            max_tokens=256,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "请将以下对话压缩为简洁、客观的中文摘要，只保留问题、关键事实和待处理事项：\n"
                        + self._format_messages(old_messages)
                    ),
                }
            ],
        )
        summary = self._extract_llm_text(response).strip()
        if not summary:
            raise ValueError("LLM 未返回摘要文本")
        return summary[: self.SUMMARY_MAX_CHARS]

    async def compress_working_memory(
        self,
        user_id: str,
        conv_id: str,
        llm_client: Any = None,
    ) -> Optional[str]:
        """Compress old working-memory messages and persist the episodic summary."""
        key = self._wm_key(str(user_id), str(conv_id))
        try:
            count = int(await self._redis.llen(key))
        except Exception as exc:
            logger.error("工作记忆压缩检查失败: %s", exc)
            return None
        if count < self.COMPRESS_AT:
            self.last_compression_triggered = False
            return None

        messages = await self._get_working_memory(user_id, conv_id, self.WORKING_MAX)
        if len(messages) < self.COMPRESS_AT:
            return None
        old_messages = messages[: -self.KEEP_AFTER_COMPRESS]
        keep_messages = messages[-self.KEEP_AFTER_COMPRESS :]
        try:
            summary = await self._generate_summary(old_messages, llm_client)
        except Exception as exc:
            logger.warning("对话摘要生成失败，使用固定降级文案: %s", exc)
            summary = f"对话包含 {len(old_messages)} 条消息（摘要生成失败）"
        summary = summary[: self.SUMMARY_MAX_CHARS]

        try:
            summary_key = self._summary_key(str(user_id), str(conv_id))
            setex = getattr(self._redis, "setex", None)
            if setex is not None:
                await setex(summary_key, self.TTL_SECONDS, summary)
            else:
                await self._redis.set(summary_key, summary)
                await self._redis.expire(summary_key, self.TTL_SECONDS)

            await self._redis.delete(key)
            # LPUSH inserts at the head; replay chronological messages so the
            # subsequent LRANGE + reverse path restores the original order.
            for message in keep_messages:
                payload = message.to_dict()
                payload["ts"] = payload.pop("timestamp")
                await self._redis.lpush(key, json.dumps(payload, ensure_ascii=False))
            await self._redis.expire(key, self.TTL_SECONDS)
            self.last_compression_triggered = True
            self.compression_trigger_key = key
        except Exception as exc:
            logger.error("工作记忆压缩写回失败: %s", exc)
            return summary

        try:
            await self._store_episodic(
                str(user_id),
                str(conv_id),
                summary,
                self._format_messages(old_messages),
            )
        except Exception as exc:
            logger.warning("情景记忆写入失败，已隔离: %s", exc)
        return summary

    def _get_episodic_collection(self) -> Any:
        if self._episodic is not None:
            return self._episodic
        if self._chroma_client is not None:
            self._episodic = self._chroma_client.get_or_create_collection("episodic")
            return self._episodic
        try:
            import chromadb
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            logger.warning("Chroma SDK 不可用: %s", exc)
            return None

        try:
            host = os.getenv("CHROMA_HOST")
            port = int(os.getenv("CHROMA_PORT", "8000"))
            if host:
                client = chromadb.HttpClient(host=host, port=port)
            else:
                client = chromadb.PersistentClient(
                    path=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma")
                )
            self._episodic = client.get_or_create_collection("episodic")
        except Exception as exc:
            logger.warning("Chroma 情景记忆集合不可用: %s", exc)
            return None
        return self._episodic

    async def _resolve_episodic_collection(self) -> Any:
        """Resolve Chroma lazily without blocking the event loop."""
        return await asyncio.to_thread(self._get_episodic_collection)

    async def _store_episodic(
        self,
        user_id: str,
        conv_id: str,
        summary: str,
        full_text: str,
    ) -> Optional[str]:
        collection = await self._resolve_episodic_collection()
        if collection is None:
            return None
        doc_id = hashlib.sha256(
            f"{user_id}:{conv_id}:{time.time_ns()}:{uuid.uuid4()}".encode("utf-8")
        ).hexdigest()[:32]
        metadata = {
            "user_id": str(user_id),
            "conv_id": str(conv_id),
            "ts": datetime.now().isoformat(),
            "full_text": str(full_text)[: self.EPISODIC_FULL_TEXT_MAX_CHARS],
        }
        await asyncio.to_thread(
            collection.add,
            ids=[doc_id],
            documents=[str(summary)],
            metadatas=[metadata],
        )
        return doc_id

    @staticmethod
    def _normalise_chroma_result(result: Any) -> List[Dict[str, Any]]:
        if not isinstance(result, dict):
            return []

        def first(value: Any) -> List[Any]:
            if isinstance(value, list) and value and isinstance(value[0], list):
                return value[0]
            return value if isinstance(value, list) else []

        ids = first(result.get("ids", []))
        documents = first(result.get("documents", []))
        metadatas = first(result.get("metadatas", []))
        distances = first(result.get("distances", []))
        items: List[Dict[str, Any]] = []
        for index, item_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            items.append(
                {
                    "id": item_id,
                    "document": documents[index] if index < len(documents) else "",
                    "metadata": metadata if isinstance(metadata, dict) else {},
                    "distance": distances[index] if index < len(distances) else None,
                }
            )
        return items

    async def _query_episodic(self, collection: Any, where: Dict[str, str], query: Optional[str], limit: int) -> Any:
        kwargs: Dict[str, Any] = {"where": where, "n_results": limit}
        kwargs["query_texts"] = [query or ""]
        return await asyncio.to_thread(collection.query, **kwargs)

    async def get_episodic_context(
        self,
        user_id: str,
        conv_id: str,
        query: Optional[str] = None,
        limit: int = HISTORY_TOP_K,
    ) -> List[Dict[str, Any]]:
        try:
            bounded_limit = min(self.HISTORY_TOP_K, max(0, int(limit)))
        except (TypeError, ValueError):
            bounded_limit = self.HISTORY_TOP_K
        if bounded_limit == 0:
            return []
        collection = await self._resolve_episodic_collection()
        if collection is None:
            return []
        try:
            same_result = await self._query_episodic(
                collection,
                {"user_id": str(user_id), "conv_id": str(conv_id)},
                query,
                bounded_limit,
            )
            combined = self._normalise_chroma_result(same_result)
            seen = {item["id"] for item in combined}
            if len(combined) < bounded_limit:
                user_result = await self._query_episodic(
                    collection,
                    {"user_id": str(user_id)},
                    query,
                    bounded_limit - len(combined),
                )
                for item in self._normalise_chroma_result(user_result):
                    if item["id"] not in seen:
                        combined.append(item)
                        seen.add(item["id"])
                    if len(combined) >= bounded_limit:
                        break
            return combined[:bounded_limit]
        except Exception as exc:
            logger.warning("情景记忆检索失败: %s", exc)
            return []

    async def close(self) -> None:
        close_method = getattr(self._redis, "aclose", None)
        if close_method is None:
            close_method = getattr(self._redis, "close", None)
        if close_method is not None:
            result = close_method()
            if inspect.isawaitable(result):
                await result
