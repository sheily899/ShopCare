from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator


app = FastAPI(title="ShopCare", version="0.1.0")


class ChatRequest(BaseModel):
    """The validated request shape at the public chat API boundary."""

    message: str
    user_id: str = "anonymous"
    conv_id: Optional[str] = None

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class ChatResponse(BaseModel):
    """The response contract exposed by the public chat API boundary."""

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
