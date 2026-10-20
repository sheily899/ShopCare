"""面向消费者的语气边界。

客服答复由对应 Agent 基于消息、知识证据和工具事实生成。本模块不添加开场白，
也不重排或改写其事实内容，避免固定模板覆盖模型答复。
"""

from __future__ import annotations

from typing import Any


def customer_friendly_response(text: str, *, intent: Any = None, emotion: str = "neutral") -> str:
    """保留 Agent 已生成的消费者答复；情绪和意图只用于模型提示/客服摘要。"""
    del intent, emotion
    return str(text or "").strip()
