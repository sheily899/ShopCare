"""Step 6.2 知识前置检索与 Agent 工具隔离测试。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from agents.agent_orchestrator import (
    AgentOrchestrator,
    AgentResponse,
    AgentType,
    BillingAgent,
    GeneralAgent,
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
    agent = GeneralAgent(client=client)
    search_calls: list[tuple[str, int]] = []

    async def search(query: str, top_k: int) -> list[dict[str, Any]]:
        search_calls.append((query, top_k))
        return [
            {
                "title": "澄润精华水成分说明",
                "content": "主要成分包括甘油、泛醇和神经酰胺 NP。",
                "score": 0.91,
                "source_chunk_id": "serum_ingredients_01",
            }
        ]

    orchestrator = AgentOrchestrator(
        agent_pool={AgentType.INQUIRY: agent},
        knowledge_search=search,
    )
    request = Request(
        message="澄润精华水有哪些主要成分？",
        intent=IntentCategory.PRODUCT_INFO,
        knowledge_required=True,
    )
    result = _run(
        orchestrator.run_parallel(
            request,
            RoutingDecision(primary_agent=AgentType.INQUIRY),
        )
    )

    assert result.response == "技术答复"
    assert search_calls == [("澄润精华水有哪些主要成分？", 5)]
    assert len(client.calls) == 1
    visible_tools = {tool["name"] for tool in client.calls[0]["tools"]}
    assert "search_knowledge_base" not in visible_tools
    assert "inspect_request_context" in visible_tools
    assert "神经酰胺 NP" in client.calls[0]["messages"][0]["content"]


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
        return [{"title": "产品规则", "content": "该商品相关规则说明。", "score": 0.9}]

    orchestrator = AgentOrchestrator(
        agent_pool={
            AgentType.INQUIRY: _Agent(AgentType.INQUIRY),
            AgentType.REFUND_RISK: _Agent(AgentType.REFUND_RISK),
        },
        knowledge_search=search,
    )
    request = Request(
        message="精华水成分和退款规则",
        intent=IntentCategory.PRODUCT_INFO,
        secondary_intents=[IntentCategory.RETURN_REFUND],
        knowledge_required=True,
    )
    result = _run(
        orchestrator.run_parallel(
            request,
            RoutingDecision(
                primary_agent=AgentType.INQUIRY,
                supporting_agents=[AgentType.REFUND_RISK],
            ),
        )
    )

    assert sorted(calls) == [("精华水成分和退款规则", 5), ("精华水成分和退款规则", 5)]
    assert result.agent_types == [AgentType.INQUIRY, AgentType.REFUND_RISK]


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
            AgentType.INQUIRY: _Agent(AgentType.INQUIRY),
            AgentType.REFUND_RISK: _Agent(AgentType.REFUND_RISK),
        },
        knowledge_search=_DomainSearcher(),
    )
    request = Request(
        message="敏感肌精华成分和退货退款规则",
        intent=IntentCategory.INGREDIENT_FIT,
        secondary_intents=[IntentCategory.RETURN_REFUND],
        knowledge_required=True,
    )
    result = _run(
        orchestrator.run_parallel(
            request,
            RoutingDecision(
                primary_agent=AgentType.INQUIRY,
                supporting_agents=[AgentType.REFUND_RISK],
            ),
        )
    )

    assert sorted(calls, key=lambda item: item[2] or "") == [
        ("敏感肌精华成分和退货退款规则", 5, "after_sales"),
        ("敏感肌精华成分和退货退款规则", 5, "consult"),
    ]
    assert {
        trace["input"].get("domain")
        for trace in result.tool_traces
        if isinstance(trace.get("input"), dict)
    } == {
        "consult",
        "after_sales",
    }


def test_required_retrieval_empty_blocks_agent_model_call() -> None:
    """必需知识检索为空时不允许模型基于空上下文生成产品结论。"""

    client = _FakeClient()
    agent = GeneralAgent(client=client)

    async def search(query: str, top_k: int) -> list[dict[str, Any]]:
        return []

    orchestrator = AgentOrchestrator(
        agent_pool={AgentType.INQUIRY: agent},
        knowledge_search=search,
    )
    result = _run(
        orchestrator.run_parallel(
            Request(
                message="澄润精华水有哪些主要成分？",
                intent=IntentCategory.PRODUCT_INFO,
                knowledge_required=True,
            ),
            RoutingDecision(primary_agent=AgentType.INQUIRY),
        )
    )

    assert client.calls == []
    assert not result.agent_types
    assert "不凭猜测" in result.response
    assert result.tool_traces[0]["status"] == "empty"
    assert result.tool_traces[0]["knowledge_required"] is True


def test_required_retrieval_exception_blocks_agent_model_call() -> None:
    """必需知识检索异常时不允许模型绕过检索继续回答。"""

    client = _FakeClient()
    agent = GeneralAgent(client=client)

    async def search(query: str, top_k: int) -> list[dict[str, Any]]:
        raise RuntimeError("embedding backend unavailable")

    orchestrator = AgentOrchestrator(
        agent_pool={AgentType.INQUIRY: agent},
        knowledge_search=search,
    )
    result = _run(
        orchestrator.run_parallel(
            Request(
                message="澄润精华水有哪些主要成分？",
                intent=IntentCategory.PRODUCT_INFO,
                knowledge_required=True,
            ),
            RoutingDecision(primary_agent=AgentType.INQUIRY),
        )
    )

    assert client.calls == []
    assert "不凭猜测" in result.response
    assert result.tool_traces[0]["status"] == "failed"
    assert result.tool_traces[0]["knowledge_required"] is True


def test_agent_chosen_required_search_with_no_results_aborts_before_final_answer() -> None:
    """条件检索一旦被 Agent 选中且空召回，必须停止后续模型润色。"""

    class ToolChoosingClient:
        def __init__(self) -> None:
            self.messages = self
            self.calls: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return SimpleNamespace(content=[SimpleNamespace(
                    type="tool_use",
                    id="toolu-search-1",
                    name="search_knowledge_base",
                    input={"query": "物流停滞处理规则"},
                )])
            return SimpleNamespace(content=[SimpleNamespace(
                type="text", text="这是未检索到依据的猜测答复。"
            )])

    client = ToolChoosingClient()
    agent = TechnicalAgent(client=client)

    async def search(query: str, top_k: int) -> list[dict[str, Any]]:
        return []

    orchestrator = AgentOrchestrator(
            agent_pool={AgentType.FULFILLMENT: agent},
        knowledge_search=search,
    )
    result = _run(
        orchestrator.run_parallel(
            Request(
                message="物流停滞几天了",
                intent=IntentCategory.LOGISTICS_STALLED,
            ),
            RoutingDecision(primary_agent=AgentType.FULFILLMENT),
        )
    )

    assert len(client.calls) == 1
    assert "这是未检索到依据的猜测答复" not in result.response
    assert "不凭猜测" in result.response
    assert any(
        trace.get("retrieval_stage") == "knowledge_tool"
        and trace.get("status") == "empty"
        and trace.get("knowledge_required") is True
        for trace in result.tool_traces
    )
