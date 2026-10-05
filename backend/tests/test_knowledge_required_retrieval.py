"""Step 6.2 知识前置检索与 Agent 工具隔离测试。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from agents.agent_orchestrator import (
    AgentOrchestrator,
    AgentResponse,
    AgentType,
    Request,
    RoutingDecision,
    TechnicalAgent,
)
from intent.intent_recognizer import IntentCategory


class _FakeClient:
    """记录 Anthropic 请求参数的异步客户端替身。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    async def create(self, **kwargs: Any) -> Any:
        """记录调用并返回确定性文本。"""

        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="技术答复")])


def _run(coroutine: Any) -> Any:
    """使用标准库事件循环执行异步用例。"""

    return asyncio.run(coroutine)


def test_business_agent_prefetches_once_and_removes_search_tool() -> None:
    """业务请求每个 Agent 只预检索一次，LLM 后续不可再看到 search 工具。"""

    client = _FakeClient()
    agent = TechnicalAgent(client=client)
    search_calls: list[tuple[str, int]] = []

    async def search(query: str, top_k: int) -> list[dict[str, Any]]:
        search_calls.append((query, top_k))
        return [
            {
                "title": "401 排查",
                "content": "请检查登录凭据和网络连接。",
                "score": 0.91,
                "source_chunk_id": "technical_chunk_401",
            }
        ]

    orchestrator = AgentOrchestrator(
        agent_pool={AgentType.TECHNICAL: agent},
        knowledge_search=search,
    )
    request = Request(
        message="登录失败",
        intent=IntentCategory.TECHNICAL_LOGIN,
        knowledge_required=True,
    )
    result = _run(
        orchestrator.run_parallel(
            request,
            RoutingDecision(primary_agent=AgentType.TECHNICAL),
        )
    )

    assert result.response == "技术答复"
    assert search_calls == [("登录失败", 5)]
    assert len(client.calls) == 1
    visible_tools = {tool["name"] for tool in client.calls[0]["tools"]}
    assert "search_knowledge_base" not in visible_tools
    assert "lookup_error_code" in visible_tools
    assert "401 排查" in client.calls[0]["messages"][0]["content"]


def test_each_parallel_agent_gets_one_independent_retrieval() -> None:
    """主 Agent 与辅助 Agent 各自独立检索一次，且都使用 top_k=5。"""

    calls: list[tuple[str, int]] = []

    class _Agent:
        def __init__(self, agent_type: AgentType) -> None:
            self.agent_type = agent_type

        async def handle(self, request: Request) -> AgentResponse:
            """返回带工具轨迹的成功结果。"""

            return AgentResponse(
                agent_type=self.agent_type,
                content=self.agent_type.value,
                success=True,
            )

    async def search(query: str, top_k: int) -> list[dict[str, Any]]:
        calls.append((query, top_k))
        return []

    orchestrator = AgentOrchestrator(
        agent_pool={
            AgentType.TECHNICAL: _Agent(AgentType.TECHNICAL),
            AgentType.BILLING: _Agent(AgentType.BILLING),
        },
        knowledge_search=search,
    )
    request = Request(message="技术和账单", knowledge_required=True)
    result = _run(
        orchestrator.run_parallel(
            request,
            RoutingDecision(
                primary_agent=AgentType.TECHNICAL,
                supporting_agents=[AgentType.BILLING],
            ),
        )
    )

    assert sorted(calls) == [("技术和账单", 5), ("技术和账单", 5)]
    assert result.agent_types == [AgentType.TECHNICAL, AgentType.BILLING]


def test_domain_scoped_retrieval_is_forwarded_without_breaking_legacy_searchers() -> None:
    """多业务域请求应把 Agent 域传给 KnowledgeBase，旧替身仍可兼容。"""

    calls: list[tuple[str, int, str | None]] = []

    class _DomainSearcher:
        def search(self, query: str, top_k: int = 5, domain: str | None = None) -> list[dict[str, Any]]:
            calls.append((query, top_k, domain))
            return []

    class _Agent:
        def __init__(self, agent_type: AgentType) -> None:
            self.agent_type = agent_type

        async def handle(self, request: Request) -> AgentResponse:
            return AgentResponse(agent_type=self.agent_type, content="ok", success=True)

    orchestrator = AgentOrchestrator(
        agent_pool={
            AgentType.TECHNICAL: _Agent(AgentType.TECHNICAL),
            AgentType.BILLING: _Agent(AgentType.BILLING),
        },
        knowledge_search=_DomainSearcher(),
    )
    request = Request(message="设备报错且退款", knowledge_required=True)
    result = _run(
        orchestrator.run_parallel(
            request,
            RoutingDecision(
                primary_agent=AgentType.TECHNICAL,
                supporting_agents=[AgentType.BILLING],
            ),
        )
    )

    assert sorted(calls, key=lambda item: item[2] or "") == [
        ("设备报错且退款", 5, "billing"),
        ("设备报错且退款", 5, "technical"),
    ]
    assert {trace["input"].get("domain") for trace in result.tool_traces} == {
        "technical",
        "billing",
    }
