"""Step 6.2 Fan-out 异常隔离与 Trace 聚合测试。"""

from __future__ import annotations

import asyncio
from typing import Any

import agents.agent_orchestrator as orchestrator_module
from agents.agent_orchestrator import (
    AgentOrchestrator,
    AgentResponse,
    AgentType,
    Request,
    RoutingDecision,
)
from intent.intent_recognizer import IntentCategory


def _run(coroutine: Any) -> Any:
    """使用标准库事件循环执行异步用例。"""

    return asyncio.run(coroutine)


class _TraceAgent:
    """返回预置工具使用记录的 Agent 替身。"""

    def __init__(self, agent_type: AgentType, error: Exception | None = None) -> None:
        self.agent_type = agent_type
        self.error = error

    async def handle(self, request: Request) -> AgentResponse:
        """返回成功响应或抛出未捕获异常。"""

        if self.error is not None:
            raise self.error
        return AgentResponse(
            agent_type=self.agent_type,
            content=f"{self.agent_type.value} 成功",
            success=True,
            tools_used=["inspect_request_context", "search_knowledge_base"],
            tool_traces=[
                {
                    "tool_name": "inspect_request_context",
                    "input": {"intent": "ingredient_fit"},
                    "latency_ms": 1.2,
                    "status": "success",
                }
            ],
        )


def test_gather_isolates_auxiliary_exception_and_keeps_primary(monkeypatch: Any) -> None:
    """辅助协程异常不得覆盖主 Agent 成功结果。"""

    async def fake_execute(request: Request, agent_type: AgentType, **kwargs: Any) -> AgentResponse:
        """模拟一个成功主分支和一个崩溃辅助分支。"""

        if agent_type is AgentType.REFUND_RISK:
            raise RuntimeError("after-sales network down")
        return AgentResponse(
            agent_type=AgentType.FULFILLMENT,
            content="主 Agent 输出",
            success=True,
            tools_used=["lookup_order"],
            tool_traces=[
                {
                    "tool_name": "lookup_order",
                    "input": {"order_id": "SC-DEMO-4821"},
                    "latency_ms": 2.0,
                    "status": "success",
                }
            ],
        )

    monkeypatch.setattr(orchestrator_module, "_execute", fake_execute)
    orchestrator = AgentOrchestrator(agent_pool={})
    result = _run(
        orchestrator.run_parallel(
            Request(message="复合问题"),
            RoutingDecision(
                primary_agent=AgentType.FULFILLMENT,
                supporting_agents=[AgentType.REFUND_RISK],
            ),
        )
    )

    assert result.response == "主 Agent 输出"
    assert result.agent_types == [AgentType.FULFILLMENT]
    assert result.tools_used == ["lookup_order"]
    assert result.tool_traces[0]["tool_name"] == "lookup_order"


def test_trace_aggregation_contains_retrieval_stage_and_deduplicated_tools() -> None:
    """并发结果应聚合工具调用、检索摘要和每个 Agent 的耗时状态。"""

    async def search(query: str, top_k: int) -> list[dict[str, Any]]:
        """返回一个可追踪的知识切片。"""

        return [
            {
                "title": "精华成分资料",
                "content": "产品主要成分及其护肤用途",
                "score": 0.88,
                "source_chunk_id": "product_ingredient_chunk_1",
            }
        ]

    orchestrator = AgentOrchestrator(
        agent_pool={AgentType.INQUIRY: _TraceAgent(AgentType.INQUIRY)},
        knowledge_search=search,
    )
    result = _run(
        orchestrator.run_parallel(
            Request(
                message="这款精华有哪些成分？",
                intent=IntentCategory.INGREDIENT_FIT,
                knowledge_required=True,
            ),
            RoutingDecision(primary_agent=AgentType.INQUIRY),
        )
    )

    assert result.tools_used == ["search_knowledge_base", "inspect_request_context"]
    assert result.tool_traces[0]["tool_name"] == "search_knowledge_base"
    retrieval_trace = result.tool_traces[0]
    assert retrieval_trace["retrieval_stage"] == "knowledge_prefetch"
    assert retrieval_trace["status"] == "success"
    assert retrieval_trace["latency_ms"] >= 0.0
    assert retrieval_trace["retrieved_chunks"][0]["source_chunk_id"] == "product_ingredient_chunk_1"
    assert result.intent_diagnostics["retrieval_stage"][0]["agent_type"] == "consult"
