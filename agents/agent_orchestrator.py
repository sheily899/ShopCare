from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.intent_recognizer import IntentCategory, UrgencyLevel


class AgentType(str, Enum):
    GENERAL = "general"
    TECHNICAL = "technical"
    BILLING = "billing"
    ESCALATION = "escalation"


@dataclass(frozen=True)
class AgentProfile:
    role: str
    mission: str
    workflow: Tuple[str, ...]
    input_contract: Tuple[str, ...]
    output_contract: Tuple[str, ...]
    handoff_conditions: Tuple[str, ...] = ()
    tool_scope: Tuple[str, ...] = ()
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1024


@dataclass
class AgentResponse:
    agent_type: AgentType
    content: str
    success: bool
    confidence: float = 1.0
    latency_ms: float = 0.0
    escalate: bool = False
    tools_used: List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Request:
    message: str
    user_id: str = "anonymous"
    conv_id: Optional[str] = None
    context: str = ""
    history: Optional[List[Any]] = None
    entities: Dict[str, List[str]] = field(default_factory=dict)
    intent: Optional[IntentCategory] = None
    intent_group: Optional[str] = None
    urgency: Optional[UrgencyLevel] = None
    intent_confidence: float = 1.0
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    intent_diagnostics: Dict[str, Any] = field(default_factory=dict)
    secondary_intents: List[IntentCategory] = field(default_factory=list)
    knowledge_required: bool = False


@dataclass
class RoutingDecision:
    primary_agent: AgentType
    supporting_agents: List[AgentType] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0


@dataclass
class OrchestratorResult:
    request_id: str
    response: str
    agent_type: AgentType
    intent: Optional[IntentCategory]
    escalated: bool = False
    agent_recommended_handoff: bool = False
    latency_ms: float = 0.0
    agent_types: List[AgentType] = field(default_factory=list)
    primary_agent: Optional[AgentType] = None
    supporting_agents: List[AgentType] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0
    intent_diagnostics: Dict[str, Any] = field(default_factory=dict)
    secondary_intents: List[IntentCategory] = field(default_factory=list)
    knowledge_required: bool = False
