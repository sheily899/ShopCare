"""复合业务请求必须按子句分发，并在各自知识域完成完整 RAG 流程。"""

from __future__ import annotations

import asyncio
from typing import Any

from agents.agent_orchestrator import (
    AgentOrchestrator,
    AgentResponse,
    AgentType,
    Request,
    RoutingDecision,
)
from intent.intent_recognizer import (
    IntentCategory,
    build_scoped_retrieval_queries,
    detect_compound_intents,
)
from mcp.knowledge_base import KnowledgeBase
from mcp.tool_manager import MCPToolManager, Tool, ToolResult


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def test_compound_request_splits_agents_and_runs_four_domain_scoped_recalls_each():
    """两个领域 Agent 各收本域子句，并各跑原句+3改写、域内召回和重排。"""

    message = "这款精华适合油皮吗？而且我想申请退款"
    secondary, _, diagnostics = detect_compound_intents(
        message,
        IntentCategory.INGREDIENT_FIT,
    )
    scoped_queries = build_scoped_retrieval_queries(
        message,
        secondary,
        diagnostics["clause_matches"],
    )
    request = Request(
        message=message,
        intent=IntentCategory.INGREDIENT_FIT,
        secondary_intents=secondary,
        sub_requests=diagnostics["sub_requests"],
        intent_diagnostics={"scoped_retrieval_queries": scoped_queries},
        knowledge_required=True,
    )

    retrieval_calls: list[tuple[str, str]] = []
    embedding_inputs: list[str] = []
    vector_queries: list[dict[str, Any]] = []
    rewrite_calls: list[str] = []
    rerank_calls: list[tuple[str, list[str], bool]] = []
    agent_inputs: dict[str, str] = {}

    class DomainCollection:
        def query(self, **kwargs):
            vector_queries.append(kwargs)
            domain = kwargs["where"]["domain"]
            query_index = len(vector_queries)
            return {
                "ids": [[f"{domain}:{query_index}"]],
                "documents": [[f"来自 {domain} 业务域的依据。"]],
                "metadatas": [[{
                    "title": f"{domain} 规则",
                    "source_chunk_id": f"{domain}:{query_index}",
                    "domain": domain,
                }]],
                "distances": [[0.2]],
            }

    class DomainClient:
        def get_or_create_collection(self, **kwargs):
            return DomainCollection()

    kb = KnowledgeBase(
        client=DomainClient(),
        embedding_function=lambda texts: [
            embedding_inputs.append(text) or [0.1, 0.2, 0.3] for text in texts
        ],
    )

    async def knowledge_handler(params, context=None):
        retrieval_calls.append((params["query"], params["domain"]))
        return kb.search(params["query"], top_k=params["top_k"], domain=params["domain"])

    manager = MCPToolManager()
    manager.register(
        Tool(
            name="search_knowledge_base",
            description="测试用域内知识检索",
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                    "domain": {"type": "string", "enum": ["consult", "order", "after_sales"]},
                },
                "required": ["query", "domain"],
                "additionalProperties": False,
            },
            handler=knowledge_handler,
        )
    )

    async def rewrite(query: str, n: int = 3) -> list[str]:
        rewrite_calls.append(query)
        assert n == 3
        return [query, f"{query} 改写一", f"{query} 改写二", f"{query} 改写三"]

    async def rerank(query, candidates, top_k=5, *, force=False, **kwargs):
        domains = [str(item["domain"]) for item in candidates]
        rerank_calls.append((query, domains, force))
        return ToolResult(
            name="rerank_results",
            success=True,
            data=list(reversed(candidates))[:top_k],
            metadata={"candidate_count": len(candidates)},
            reranked=True,
        )

    manager.rewrite_query = rewrite
    manager.rerank_results = rerank

    class CapturingAgent:
        def __init__(self, agent_type: AgentType) -> None:
            self.agent_type = agent_type

        async def handle(self, request: Request) -> AgentResponse:
            agent_inputs[self.agent_type.value] = request.message
            return AgentResponse(
                agent_type=self.agent_type,
                content=f"{self.agent_type.value} handled",
                success=True,
            )

    orchestrator = AgentOrchestrator(
        agent_pool={
            AgentType.INQUIRY: CapturingAgent(AgentType.INQUIRY),
            AgentType.REFUND_RISK: CapturingAgent(AgentType.REFUND_RISK),
        },
        knowledge_search=manager,
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

    assert set(agent_inputs) == {"consult", "after_sales"}
    assert agent_inputs == scoped_queries
    assert "申请退款" not in agent_inputs["consult"]
    assert "精华适合油皮" not in agent_inputs["after_sales"]

    assert len(rewrite_calls) == 2
    assert sorted(domain for _, domain in retrieval_calls) == [
        "after_sales", "after_sales", "after_sales", "after_sales",
        "consult", "consult", "consult", "consult",
    ]
    for domain in ("consult", "after_sales"):
        domain_calls = [query for query, item_domain in retrieval_calls if item_domain == domain]
        assert domain_calls == [
            scoped_queries[domain],
            f"{scoped_queries[domain]} 改写一",
            f"{scoped_queries[domain]} 改写二",
            f"{scoped_queries[domain]} 改写三",
        ]

    assert len(rerank_calls) == 2
    assert {query for query, _, _ in rerank_calls} == set(scoped_queries.values())
    domain_for_query = {query: domain for domain, query in scoped_queries.items()}
    assert all(
        len(domains) == 4 and set(domains) == {domain_for_query[query]}
        for query, domains, _ in rerank_calls
    )
    assert all(force for _, _, force in rerank_calls)

    assert len(embedding_inputs) == 8
    assert {query for query, _ in retrieval_calls} == set(embedding_inputs)
    assert len(vector_queries) == 8
    assert all("where" in query for query in vector_queries)
    assert sorted(query["where"]["domain"] for query in vector_queries) == [
        "after_sales", "after_sales", "after_sales", "after_sales",
        "consult", "consult", "consult", "consult",
    ]

    retrieval_traces = [
        trace for trace in result.tool_traces
        if trace.get("retrieval_stage") == "knowledge_prefetch"
    ]
    assert len(retrieval_traces) == 2
    assert {trace["input"]["domain"] for trace in retrieval_traces} == set(scoped_queries)
    assert all(len(trace["query_variants"]) == 4 for trace in retrieval_traces)
    assert all(trace["reranked"] is True for trace in retrieval_traces)
    assert all(
        {chunk["domain"] for chunk in trace["retrieved_chunks"]} == {trace["input"]["domain"]}
        for trace in retrieval_traces
    )
