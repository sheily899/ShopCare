"""ShopCare 公共 API 边界。

本模块只负责 FastAPI 应用壳与 Pydantic 请求/响应契约；业务编排、记忆
访问和外部工具调用均留在该边界之外。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator


app = FastAPI(title="ShopCare", version="0.1.0")


class ChatRequest(BaseModel):
    """单次 ShopCare 会话请求的边界校验契约。

    本模型只做 API 边界校验，不负责意图识别或 Agent 执行。

    Attributes:
        message: 必须包含非空白字符的用户消息。
        user_id: 稳定的用户标识，默认值为 ``anonymous``。
        conv_id: 调用方可选传入的会话标识。
    """

    message: str
    user_id: str = "anonymous"
    conv_id: Optional[str] = None

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        """在消息进入核心链路前拒绝纯空白文本。

        Args:
            value: API 载荷中的原始消息。

        Returns:
            包含非空白内容时返回原始消息。

        Raises:
            ValueError: 去除首尾空白后消息为空时抛出。
        """
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class ChatResponse(BaseModel):
    """公共 API 边界对外序列化的响应契约。

    字段映射内部路由结果，但不暴露 dataclass 实例或客户端句柄等实现细节。

    Attributes:
        conv_id: 响应所属的会话标识。
        request_id: 请求追踪标识，缺省为空字符串。
        response: 面向用户的响应文本。
        intent: 路由层选定的主意图值。
        secondary_intents: 跨域辅助意图值列表。
        intent_group: 主意图所属的粗粒度业务分组。
        agent_type: 主响应 Agent 类型。
        agent_types: 参与响应的 Agent 类型列表。
        primary_agent: 主 Agent 名称。
        supporting_agents: 辅助 Agent 名称列表。
        tools_used: 本次请求使用过的工具名。
        routing_reason: 路由决策说明。
        routing_confidence: 路由置信度，范围为 0 到 1。
        escalated: 安全策略是否将请求标记为升级建议。
        agent_recommended_handoff: Agent 是否建议转人工。
        latency_ms: 端到端耗时，单位为毫秒。
        knowledge_used: 是否使用知识库内容。
        entities: 从用户消息中抽取的实体。
        intent_confidence: 主意图置信度，范围为 0 到 1。
        intent_source_scores: 各识别来源的置信度。
        intent_diagnostics: 意图识别与路由诊断信息。
    """

    conv_id: str
    request_id: str = ""
    response: str
    intent: str
    secondary_intents: List[str] = Field(default_factory=list)
    intent_group: str = "other"
    agent_type: str
    agent_types: List[str] = Field(default_factory=list)
    primary_agent: str = ""
    supporting_agents: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    escalated: bool
    agent_recommended_handoff: bool = False
    latency_ms: float = Field(..., ge=0.0)
    knowledge_used: bool = False
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    intent_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    intent_source_scores: Dict[str, float] = Field(default_factory=dict)
    intent_diagnostics: Dict[str, Any] = Field(default_factory=dict)
