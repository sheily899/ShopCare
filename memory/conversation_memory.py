"""ShopCare 会话记忆与用户画像存储模块。

本模块属于 memory 层，负责 Redis 工作记忆、摘要压缩、Chroma 情景记忆和
用户画像的异步读写；它只提供上下文持久化与召回，不负责意图路由或真实业务写操作。
所有同步 Chroma 调用都通过线程封装，避免阻塞事件循环。
"""

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

# 20 条读取上限控制单次上下文预算，避免工作记忆无限增长。
WORKING_MAX = 20
# 15 条进入压缩入口，用摘要替换旧消息以控制 Token 成本。
COMPRESS_AT = 15
# 压缩后保留最新 5 条，兼顾短期连贯性与上下文长度。
KEEP_AFTER_COMPRESS = 5
# 86400 秒是一日 TTL，防止过期会话长期占用 Redis 与上下文预算。
TTL_SECONDS = 86400
HISTORY_TOP_K = 5
SUMMARY_MAX_CHARS = 800
EPISODIC_FULL_TEXT_MAX_CHARS = 500
PROFILE_DOC_PREFIX = "profile:"


class MsgRole(str, Enum):
    """消息角色枚举，用于区分用户、助手和系统上下文。"""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """工作记忆中的单条消息。

    Args:
        role: 消息来源角色。
        content: 消息正文。
        timestamp: 消息产生时间，默认使用当前本地时间。
        metadata: 与消息关联的可选结构化元数据。
    """

    role: MsgRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """将消息序列化为可写入 Redis 的 JSON 兼容字典。

        Returns:
            包含角色、正文、ISO 时间戳和元数据的字典。
        """
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """从 Redis 反序列化字典并恢复消息对象。

        Args:
            data: 包含角色、正文和时间戳的消息字典。

        Returns:
            还原后的 ``Message`` 实例。

        Raises:
            TypeError: 输入不是字典时抛出。
            ValueError: 角色或时间戳格式不合法时抛出。
        """
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
    """供 Agent 使用的摘要、近期消息、情景历史和用户画像上下文。"""

    summary: str = ""
    recent_messages: List[Message] = field(default_factory=list)
    relevant_history: List[Any] = field(default_factory=list)
    user_profile: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _clean(value: Any) -> str:
        """将上下文值转换为可安全拼接的 UTF-8 文本。

        Args:
            value: 任意待展示的上下文值。

        Returns:
            清理后的字符串。
        """
        text = value if isinstance(value, str) else str(value)
        return text.encode("utf-8", errors="ignore").decode("utf-8")

    @classmethod
    def _history_text(cls, item: Any) -> str:
        """将一条历史片段转换为提示词中的单行文本。

        Args:
            item: 消息对象、字典或其他历史片段。

        Returns:
            适合拼接到上下文提示词的文本。
        """
        if isinstance(item, Message):
            return f"{item.role.value}: {cls._clean(item.content)}"
        if isinstance(item, dict):
            return cls._clean(json.dumps(item, ensure_ascii=False))
        return cls._clean(item)

    def to_prompt_text(self) -> str:
        """按固定截断规则生成可直接提供给模型的上下文文本。

        Returns:
            依次包含摘要、最多 3 条相关历史、用户画像和最后 8 条消息的文本。
        """
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
    """管理 ShopCare 的工作记忆、长程情景记忆和用户画像。

    所有公共异步方法都将存储故障隔离在记忆层，避免影响主对话链路。
    """

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
        profile_collection: Any = None,
        chroma_client: Any = None,
        model: Optional[str] = None,
    ) -> None:
        """初始化 Redis、LLM 和 Chroma 依赖。

        Args:
            redis_client: 可注入的异步 Redis 客户端。
            redis_url: 未注入客户端时使用的 Redis 地址。
            llm_client: 可注入的 Anthropic 异步客户端。
            episodic_collection: 可注入的情景记忆集合。
            profile_collection: 可注入的用户画像集合。
            chroma_client: 用于惰性创建集合的 Chroma 客户端。
            model: 摘要与画像抽取使用的模型名称。
        """
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
        self._profile = profile_collection
        self._chroma_client = chroma_client
        self.last_compression_triggered = False
        self.compression_trigger_key: Optional[str] = None

    @staticmethod
    def _wm_key(user_id: str, conv_id: str) -> str:
        """构造工作记忆 Redis key。

        Args:
            user_id: 用户标识。
            conv_id: 会话标识。

        Returns:
            ``wm:{user_id}:{conv_id}`` 格式的 key。
        """
        return f"wm:{user_id}:{conv_id}"

    @staticmethod
    def _summary_key(user_id: str, conv_id: str) -> str:
        """构造会话摘要 Redis key。

        Args:
            user_id: 用户标识。
            conv_id: 会话标识。

        Returns:
            ``summary:{user_id}:{conv_id}`` 格式的 key。
        """
        return f"summary:{user_id}:{conv_id}"

    async def add_message(
        self,
        user_id: str,
        conv_id: str,
        role: MsgRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """异步写入一条工作记忆消息并返回是否达到压缩入口。

        Args:
            user_id: 用户标识。
            conv_id: 会话标识。
            role: 消息角色。
            content: 消息正文。
            metadata: 可选消息元数据。

        Returns:
            达到 15 条压缩阈值时返回 ``True``，写入失败或未达到时返回 ``False``。
        """
        key = self._wm_key(str(user_id), str(conv_id))
        try:
            message = Message(
                role=role if isinstance(role, MsgRole) else MsgRole(role),
                content=str(content),
                metadata=dict(metadata or {}),
            )
            payload = message.to_dict()
            # Redis 继续使用原工程的 ts 字段，以兼容既有数据；Message 对外仍提供显式的 timestamp。
            payload["ts"] = payload.pop("timestamp")
            await self._redis.lpush(
                key,
                json.dumps(payload, ensure_ascii=False),
            )
            # 一日 TTL 限制陈旧上下文的生命周期，避免无界累积带来存储和 Token 成本。
            await self._redis.expire(key, self.TTL_SECONDS)
            count = await self._redis.llen(key)
            # 15 条是压缩入口：在上下文接近预算上限前先产生摘要并保留热消息。
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
        """读取最多 20 条工作记忆并恢复为时间正序。

        Args:
            user_id: 用户标识。
            conv_id: 会话标识。
            limit: 调用方请求的条数，最终仍受 20 条硬上限约束。

        Returns:
            按时间先后排列的消息列表；Redis 故障时返回空列表。
        """
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

    async def get_context(
        self,
        user_id: str,
        conv_id: str,
        query: Optional[str] = None,
    ) -> MemoryContext:
        """按 API 主链路顺序组装摘要、近期消息、情景历史和用户画像。

        Args:
            user_id: 用户标识。
            conv_id: 会话标识。
            query: 当前查询，用于情景记忆召回；可为空。

        Returns:
            可直接注入意图识别和 Agent Prompt 的 ``MemoryContext``。任一存储层
            失败只返回该层已有的安全空值，不伪造历史内容。
        """

        recent_messages = await self.get_recent_messages(user_id, conv_id, limit=self.WORKING_MAX)
        summary = ""
        try:
            getter = getattr(self._redis, "get", None)
            if callable(getter):
                raw_summary = getter(self._summary_key(user_id, conv_id))
                summary_value = await raw_summary if inspect.isawaitable(raw_summary) else raw_summary
                summary = str(summary_value or "")
        except Exception as exc:
            logger.warning("读取会话摘要失败：%s", exc)
        relevant_history = await self.get_episodic_context(
            user_id,
            conv_id,
            query=query,
            limit=self.HISTORY_TOP_K,
        )
        user_profile = await self.get_user_profile(user_id)
        return MemoryContext(
            summary=summary,
            recent_messages=recent_messages,
            relevant_history=relevant_history,
            user_profile=user_profile,
        )

    async def _get_working_memory(
        self, user_id: str, conv_id: str, limit: int = WORKING_MAX
    ) -> List[Message]:
        """提供内部统一的工作记忆读取入口。

        Args:
            user_id: 用户标识。
            conv_id: 会话标识。
            limit: 读取上限。

        Returns:
            按时间正序排列的消息列表。
        """
        return await self.get_recent_messages(user_id, conv_id, limit)

    @staticmethod
    def _format_messages(messages: List[Message]) -> str:
        """将消息列表格式化为摘要或画像抽取提示词文本。

        Args:
            messages: 待格式化的消息。

        Returns:
            每行一条 ``role: content`` 的纯文本。
        """
        return "\n".join(f"{message.role.value}: {message.content}" for message in messages)

    async def _get_llm_client(self) -> Any:
        """惰性创建并缓存 Anthropic 异步客户端。

        Returns:
            已注入或新建的异步 LLM 客户端。

        Raises:
            RuntimeError: Anthropic SDK 不可用时抛出。
        """
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
        """从 Anthropic 响应对象或兼容 Mock 中提取文本。

        Args:
            response: LLM SDK 响应对象。

        Returns:
            首个可用文本片段；没有文本时返回空字符串。
        """
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
        """调用 LLM 生成旧消息摘要，并限制在 800 字符内。

        Args:
            old_messages: 待归档的旧消息。
            llm_client: 可选的异步 LLM 客户端。

        Returns:
            截断后的摘要文本。

        Raises:
            Exception: LLM 调用失败或未返回文本时交由上层确定性降级。
        """
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
        """压缩旧工作记忆、写回摘要并异步安全保存情景记忆。

        Args:
            user_id: 用户标识。
            conv_id: 会话标识。
            llm_client: 可选的摘要 LLM 客户端。

        Returns:
            成功时返回摘要，未达到压缩阈值或检查失败时返回 ``None``。
        """
        key = self._wm_key(str(user_id), str(conv_id))
        try:
            count = int(await self._redis.llen(key))
        except Exception as exc:
            logger.error("工作记忆压缩检查失败: %s", exc)
            return None
        # 只有达到 15 条才进入压缩，避免过早摘要损失上下文并增加模型调用。
        if count < self.COMPRESS_AT:
            self.last_compression_triggered = False
            return None

        messages = await self._get_working_memory(user_id, conv_id, self.WORKING_MAX)
        if len(messages) < self.COMPRESS_AT:
            return None
        # 采用“旧消息归档、最新 5 条保留”的滑窗，平衡 Token 预算与即时上下文连贯性。
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
            # LPUSH 从头部插入；按时间正序回放，后续 LRANGE + reverse 才能恢复原始顺序。
            for message in keep_messages:
                payload = message.to_dict()
                payload["ts"] = payload.pop("timestamp")
                await self._redis.lpush(key, json.dumps(payload, ensure_ascii=False))
            # 压缩后的热记忆仍保持一日 TTL，避免摘要重置后绕过生命周期约束。
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
        """同步解析或创建 ``episodic`` Chroma 集合。

        Returns:
            Chroma collection；依赖不可用或初始化失败时返回 ``None``。
        """
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
        """在线程中惰性解析情景集合，避免同步 Chroma 阻塞事件循环。

        Returns:
            ``episodic`` 集合或 ``None``。
        """
        return await asyncio.to_thread(self._get_episodic_collection)

    def _get_profile_collection(self) -> Any:
        """同步解析或创建 ``user_profile`` Chroma 集合。

        Returns:
            Chroma collection；依赖不可用或初始化失败时返回 ``None``。
        """
        if self._profile is not None:
            return self._profile
        if self._chroma_client is not None:
            self._profile = self._chroma_client.get_or_create_collection("user_profile")
            return self._profile
        try:
            import chromadb
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            logger.warning("Chroma SDK 不可用，无法初始化用户画像: %s", exc)
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
            self._profile = client.get_or_create_collection("user_profile")
        except Exception as exc:
            logger.warning("Chroma 用户画像集合不可用: %s", exc)
            return None
        return self._profile

    async def _resolve_profile_collection(self) -> Any:
        """在线程中解析用户画像集合，避免同步 Chroma 阻塞事件循环。

        Returns:
            ``user_profile`` 集合或 ``None``。
        """
        return await asyncio.to_thread(self._get_profile_collection)

    @staticmethod
    def _validate_profile_data(data: Any) -> Dict[str, Any]:
        """校验并复制画像 JSON，阻止畸形数据覆盖已有画像。

        Args:
            data: LLM 返回或 Chroma 读取的候选画像。

        Returns:
            只包含字符串列表的 ``preferences`` 与 ``entities`` 字典。

        Raises:
            ValueError: 画像缺少必需字段或字段类型不符合契约。
        """
        if not isinstance(data, dict):
            raise ValueError("用户画像必须是 JSON 对象")
        preferences = data.get("preferences")
        entities = data.get("entities")
        if not isinstance(preferences, list) or any(
            not isinstance(item, str) for item in preferences
        ):
            raise ValueError("用户画像 preferences 必须是字符串数组")
        if not isinstance(entities, dict):
            raise ValueError("用户画像 entities 必须是对象")
        for key, values in entities.items():
            if not isinstance(key, str) or not isinstance(values, list) or any(
                not isinstance(item, str) for item in values
            ):
                raise ValueError("用户画像 entities 的值必须是字符串数组")
        # 只复制已校验的 JSON 结构，避免调用方在持久化后通过可变对象篡改画像。
        return {
            "preferences": list(preferences),
            "entities": {key: list(values) for key, values in entities.items()},
        }

    @staticmethod
    def _profile_prompt(messages: List[Message]) -> str:
        """构造用户画像抽取提示词，要求模型只返回严格 JSON。

        Args:
            messages: 最近的工作记忆消息。

        Returns:
            包含固定画像 schema 和消息文本的提示词。
        """
        return (
            "请从以下最近对话中提取用户画像，只返回严格 JSON，不要添加 Markdown 或解释。\n"
            "JSON 结构必须为：{\"preferences\":[\"...\"],\"entities\":{"
            "\"order_id\":[],\"product\":[],\"date\":[],\"amount\":[],"
            "\"error_code\":[]}}\n"
            "只记录用户明确表达的偏好和已出现的实体；无法确认的字段使用空数组。\n"
            "最近对话：\n"
            + MemoryManager._format_messages(messages)
        )

    async def update_profile(
        self,
        user_id: str,
        conv_id: str,
        llm_client: Any = None,
    ) -> None:
        """异步提取并持久化用户画像，适合由 ``asyncio.create_task`` 后台触发。

        Args:
            user_id: 用户标识。
            conv_id: 当前会话标识。
            llm_client: 可选的画像抽取 LLM 客户端。

        Returns:
            无返回值；任何异常都会在方法内部隔离。
        """
        try:
            messages = await self._get_working_memory(user_id, conv_id, limit=10)
            if not messages:
                return

            client = llm_client or await self._get_llm_client()
            response = await client.messages.create(
                model=self._model,
                max_tokens=512,
                temperature=0,
                messages=[
                    {"role": "user", "content": self._profile_prompt(messages)}
                ],
            )
            raw_text = self._extract_llm_text(response).strip()
            profile = self._validate_profile_data(json.loads(raw_text))
            collection = await self._resolve_profile_collection()
            if collection is None:
                return

            document_id = f"{PROFILE_DOC_PREFIX}{user_id}"
            document = json.dumps(profile, ensure_ascii=False)
            metadata = {
                "user_id": str(user_id),
                "conv_id": str(conv_id),
                "updated_at": datetime.now().isoformat(),
            }
            values = {
                "ids": [document_id],
                "documents": [document],
                "metadatas": [metadata],
            }
            upsert = getattr(collection, "upsert", None)
            if upsert is not None:
                await asyncio.to_thread(upsert, **values)
            else:
                delete = getattr(collection, "delete", None)
                if delete is not None:
                    await asyncio.to_thread(delete, ids=[document_id])
                await asyncio.to_thread(collection.add, **values)
        except Exception as exc:
            # 该协程由 asyncio.create_task 后台执行；吞掉异常是为了防止画像失败冒泡导致用户白屏。
            logger.warning("用户画像异步更新失败: %s", exc)

    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """读取用户画像并在缺失或异常时返回空结构。

        Args:
            user_id: 用户标识。

        Returns:
            包含 ``preferences`` 和 ``entities`` 的画像字典。
        """
        default = {"preferences": [], "entities": {}}
        try:
            collection = await self._resolve_profile_collection()
            if collection is None:
                return default
            result = await asyncio.to_thread(
                collection.get,
                ids=[f"{PROFILE_DOC_PREFIX}{user_id}"],
            )
            documents = result.get("documents", []) if isinstance(result, dict) else []
            if documents and isinstance(documents[0], list):
                documents = documents[0]
            if not documents:
                return default
            raw_document = documents[0]
            parsed = json.loads(raw_document) if isinstance(raw_document, str) else raw_document
            return self._validate_profile_data(parsed)
        except Exception as exc:
            logger.warning("用户画像读取失败: %s", exc)
            return default

    async def _store_episodic(
        self,
        user_id: str,
        conv_id: str,
        summary: str,
        full_text: str,
    ) -> Optional[str]:
        """将摘要及前序对话片段写入 ``episodic`` 集合。

        Args:
            user_id: 用户标识。
            conv_id: 会话标识。
            summary: 压缩后的摘要文本。
            full_text: 摘要对应的原始对话文本。

        Returns:
            新建文档 ID；集合不可用时返回 ``None``。
        """
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
        """把 Chroma 的嵌套或扁平返回统一为记录列表。

        Args:
            result: Chroma ``query`` 或 ``get`` 返回值。

        Returns:
            每条记录包含 id、document、metadata 和 distance 的列表。
        """
        if not isinstance(result, dict):
            return []

        def first(value: Any) -> List[Any]:
            """展开 Chroma 单查询结果常见的二维外层列表。"""
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
        """在线程中执行一次带过滤条件的情景记忆查询。

        Args:
            collection: Chroma 情景集合。
            where: 用户或会话过滤条件。
            query: 可选语义检索文本。
            limit: 本次最多召回条数。

        Returns:
            Chroma 原始查询结果。
        """
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
        """按同会话优先、同用户补足的顺序召回情景记忆。

        Args:
            user_id: 用户标识。
            conv_id: 当前会话标识。
            query: 可选的语义检索文本。
            limit: 返回条数，最终受 5 条上限约束。

        Returns:
            去重后的情景记忆记录；Chroma 异常时返回空列表。
        """
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
        """关闭 Redis 客户端（兼容 ``aclose`` 与 ``close`` 两种 API）。"""
        close_method = getattr(self._redis, "aclose", None)
        if close_method is None:
            close_method = getattr(self._redis, "close", None)
        if close_method is not None:
            result = close_method()
            if inspect.isawaitable(result):
                await result
