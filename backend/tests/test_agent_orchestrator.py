"""Step 5.4 四大领域 Agent、Profile 与工具白名单测试。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from agents.agent_orchestrator import (
    AgentProfile,
    AgentResponse,
    AgentStats,
    AgentType,
    AgentOrchestrator,
    BaseAgent,
    BillingAgent,
    EscalationAgent,
    GeneralAgent,
    Request,
    TechnicalAgent,
    _policy_scoped_tools,
)
from agents.tools import AgentToolSpec, INSPECT_REQUEST_CONTEXT_TOOL
from intent.intent_recognizer import IntentCategory, UrgencyLevel


def _run(coroutine: Any) -> Any:
    """使用标准库事件循环运行异步 Agent。"""

    return asyncio.run(coroutine)


def test_three_beauty_agent_profiles_lock_temperature_and_token_budget() -> None:
    """三个美妆业务 Agent 按职责使用不同温度和 Token 预算。"""

    agents = [
        (GeneralAgent(), AgentType.INQUIRY, 0.3, 900),
        (TechnicalAgent(), AgentType.FULFILLMENT, 0.1, 1200),
        (BillingAgent(), AgentType.REFUND_RISK, 0.0, 1100),
    ]

    for agent, expected_type, temperature, max_tokens in agents:
        assert agent.agent_type is expected_type
        assert isinstance(agent.profile, AgentProfile)
        assert agent.profile.temperature == temperature
        assert agent.profile.max_tokens == max_tokens
        assert isinstance(agent.stats, AgentStats)


def test_runtime_intent_policy_scopes_order_and_after_sales_tools() -> None:
    """静态工具池按实际意图裁剪，订单 Agent 不会看到退款工具。"""

    order_agent = TechnicalAgent()
    after_sales_agent = BillingAgent()
    order_request = Request(message="查订单", intent=IntentCategory.ORDER_STATUS_BEAUTY)
    refund_request = Request(message="查退款", intent=IntentCategory.REFUND_NOT_RECEIVED)
    order_tools = _policy_scoped_tools(
        order_agent, order_request, AgentType.TECHNICAL, None, knowledge_prefetched=False
    )
    after_sales_tools = _policy_scoped_tools(
        after_sales_agent, refund_request, AgentType.BILLING, None, knowledge_prefetched=False
    )
    order_names = {tool.name for tool in order_tools or []}
    after_sales_names = {tool.name for tool in after_sales_tools or []}

    assert order_agent.profile.role == "order"
    assert after_sales_agent.profile.role == "after_sales"
    assert {"lookup_order", "inspect_request_context"} <= order_names
    assert "check_refund_status" not in order_names
    assert {"lookup_order", "check_refund_status"} <= after_sales_names


def test_abnormal_refund_handoff_is_independent_from_p1_urgency(monkeypatch) -> None:
    """异常退款进入人工核对，但紧急度仍按 P1，不把转人工混成 P0。"""
    from agents import agent_orchestrator as orchestrator_module

    async def fake_fan_out(request, decision, **kwargs):
        return [AgentResponse(
            agent_type=AgentType.BILLING,
            content="这笔退款记录需要进一步核对。",
            success=True,
        )]

    class FakeComposer:
        async def compose(self, request, responses, primary_agent=None):
            return responses[0].content

    monkeypatch.setattr(orchestrator_module, "_fan_out", fake_fan_out)
    orchestrator = AgentOrchestrator(
        agent_pool={AgentType.BILLING: object()}, composer=FakeComposer()
    )
    request = Request(
        message="这笔退款记录疑似重复",
        intent=IntentCategory.SUSPECTED_ABNORMAL_REFUND,
        urgency=UrgencyLevel.HIGH,
    )

    result = asyncio.run(orchestrator.run(request))

    assert result.escalated is True
    assert result.ticket_draft["needs_review"] is True
    assert result.handoff_summary["urgency"] == "P1"


def test_required_knowledge_failure_branch_uses_shared_reply_guards(monkeypatch) -> None:
    """必需知识失败时，保留检索边界，同时过滤已知字段追问和旧答复。"""
    from agents import agent_orchestrator as orchestrator_module

    async def fake_fan_out(request, decision, **kwargs):
        return [AgentResponse(
            agent_type=AgentType.FULFILLMENT,
            content="物流已签收。订单号是多少？请再提供订单号。",
            success=True,
            tool_traces=[{
                "retrieval_stage": "prefetch",
                "knowledge_required": True,
                "status": "failed",
            }],
        )]

    class FakeComposer:
        async def compose(self, *args, **kwargs):
            raise AssertionError("必需知识失败时不应调用合成模型")

    monkeypatch.setattr(orchestrator_module, "_fan_out", fake_fan_out)
    orchestrator = AgentOrchestrator(
        agent_pool={AgentType.FULFILLMENT: object()}, composer=FakeComposer()
    )
    request = Request(
        message="再帮我看下物流",
        intent=IntentCategory.LOGISTICS_STALLED,
        entities={"order_id": ["SC-DEMO-4821"]},
        history=[{"role": "assistant", "content": "物流已签收。"}],
        knowledge_required=True,
    )

    result = asyncio.run(orchestrator.run_parallel(
        request,
        orchestrator_module.RoutingDecision(primary_agent=AgentType.FULFILLMENT),
    ))

    assert "物流已签收" not in result.response
    assert "订单号是多少" not in result.response
    assert "不凭猜测" in result.response


class _FakeClient:
    """记录消息调用次数的 Anthropic 客户端替身。"""

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = response
        self.error = error
        self.messages = self

    async def create(self, **kwargs: Any) -> Any:
        """记录请求并返回预置响应。"""

        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _SequencedClient:
    def __init__(self, responses: list[Any]) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses = list(responses)
        self.messages = self

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_escalation_agent_is_deterministic_and_skips_llm() -> None:
    """EscalationAgent 只生成升级建议，不消耗 LLM 或设置真实接入状态。"""

    client = _FakeClient(
        response=SimpleNamespace(content=[SimpleNamespace(type="text", text="不应调用")])
    )
    agent = EscalationAgent(client=client)
    request = Request(
        message="请马上转人工",
        request_id="abc12345",
        intent=IntentCategory.HUMAN_HANDOFF,
        urgency=UrgencyLevel.HIGH,
    )

    result = _run(agent.handle(request))

    assert result.success is True
    assert result.agent_type is AgentType.ESCALATION
    assert result.content == "您的问题已记录下来，当前建议联系人工客服进一步核实和处理。"
    assert "request_id" not in result.content
    assert "human_handoff" not in result.content
    assert "紧急度" not in result.content
    assert "已接入" not in result.content
    assert agent.tools == []
    assert client.calls == []
    assert agent.stats.total == 1
    assert agent.stats.success == 1


def test_base_agent_catches_llm_exception_and_records_latency() -> None:
    """LLM 异常不得冒泡，失败结果仍需有耗时与统计。"""

    agent = GeneralAgent(client=_FakeClient(error=RuntimeError("mock failure")))
    result = _run(agent.handle(Request(message="怎么查询订单")))

    assert isinstance(result, AgentResponse)
    assert result.success is False
    assert "mock failure" in result.content
    assert result.latency_ms >= 0.0
    assert agent.stats.total == 1
    assert agent.stats.success == 0
    assert agent.stats.total_ms >= result.latency_ms


def test_base_agent_passes_contextual_field_state_and_blocks_repeated_known_questions() -> None:
    """模型须拿到已知/缺失字段，输出仍重复追问时由代码过滤。"""

    client = _FakeClient(
        response=SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="text",
                    text="请问商品是否拆封或使用过？退货原因是什么？",
                )
            ]
        )
    )
    agent = GeneralAgent(client=client, tools=[])
    request = Request(
        message="订单 SC-DEMO-4821 已签收，商品未拆封，我想退款。",
        intent=IntentCategory.RETURN_REFUND,
        entities={
            "order_id": ["SC-DEMO-4821"],
            "opened_or_used": ["未拆封"],
        },
        case_facts={
            "facts": {
                "opened_or_used": [
                    {"value": "已拆封", "confirmation_status": "unconfirmed"}
                ]
            }
        },
    )

    result = _run(agent.handle(request))

    prompt = str(client.calls[0]["messages"])
    assert "当前业务字段状态" in prompt
    assert '"opened_or_used": ["未拆封"]' in prompt
    assert '"return_reason"' in prompt
    assert result.success is True
    assert "是否拆封" not in result.content
    assert "退货原因是什么" in result.content


def test_tool_context_prefers_current_turn_entities_over_stale_case_facts() -> None:
    agent = GeneralAgent(client=_FakeClient(), tools=[])
    request = Request(
        message="商品已拆封但没有使用。",
        intent=IntentCategory.RETURN_REFUND,
        entities={"opened_or_used": ["已拆封未使用"]},
        case_facts={
            "facts": {
                "opened_or_used": [
                    {"value": "未拆封", "confirmation_status": "unconfirmed"}
                ]
            }
        },
    )

    context = agent._tool_context(request)

    assert context["entities"]["opened_or_used"] == ["已拆封未使用"]


def test_base_agent_falls_back_only_to_genuinely_missing_field_if_model_repeats_known_field() -> None:
    client = _FakeClient(
        response=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="商品是否拆封？")]
        )
    )
    agent = GeneralAgent(client=client, tools=[])
    request = Request(
        message="订单 SC-DEMO-4821 已签收，商品未拆封，我想退款。",
        intent=IntentCategory.RETURN_REFUND,
        entities={
            "order_id": ["SC-DEMO-4821"],
            "opened_or_used": ["未拆封"],
        },
    )

    result = _run(agent.handle(request))

    assert result.success is True
    assert "拆封" not in result.content
    assert "退款原因" in result.content


def test_successful_tool_result_reaches_case_facts_and_placeholder_is_retried_by_model() -> None:
    async def lookup_order(params: Any, context: Any = None) -> dict[str, Any]:
        return {
            "success": True,
            "available": True,
            "order_id": params["order_id"],
            "order": {"product_name": "轻透粉底液", "status": "已签收"},
        }

    lookup_tool = AgentToolSpec(
        "lookup_order",
        "查询演示订单事实",
        {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
        lookup_order,
    )
    client = _SequencedClient(
        [
            SimpleNamespace(content=[SimpleNamespace(type="tool_use", id="toolu-1", name="lookup_order", input={"order_id": "SC-DEMO-4821"})]),
            SimpleNamespace(content=[SimpleNamespace(type="text", text="已完成必要的工具处理。")]),
            SimpleNamespace(content=[SimpleNamespace(type="text", text="SC-DEMO-4821 对应的轻透粉底液已签收。")]),
        ]
    )
    agent = GeneralAgent(client=client, tools=[lookup_tool])

    result = _run(
        agent.handle(
            Request(
                message="订单 SC-DEMO-4821 到了吗？",
                user_id="consumer-demo",
                intent=IntentCategory.ORDER_STATUS_BEAUTY,
                entities={"order_id": ["SC-DEMO-4821"]},
                request_id="request-123",
            )
        )
    )

    assert result.success is True
    assert result.content == "SC-DEMO-4821 对应的轻透粉底液已签收。"
    assert len(client.calls) == 3
    assert result.case_facts["facts"]["product_name"][-1]["value"] == "轻透粉底液"
    assert result.case_facts["facts"]["order_status"][-1]["value"] == "已签收"


def test_base_agent_rejects_placeholder_after_successful_tool_results() -> None:
    """工具已成功但最终模型仍只给占位话术时，不能伪装成答复成功。"""

    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                id="toolu-1",
                name="inspect_request_context",
                input={},
            )
        ]
    )
    client = _FakeClient(response=response)
    agent = GeneralAgent(client=client, tools=[INSPECT_REQUEST_CONTEXT_TOOL])

    result = _run(agent.handle(Request(message="请检查当前请求")))

    assert result.success is False
    assert len(client.calls) == 3
    assert "Agent 执行失败" in result.content
    for call in client.calls[:2]:
        assert {tool["name"] for tool in call["tools"]} == {"inspect_request_context"}
    assert "tools" not in client.calls[-1]
    assert any("本轮可引用的 CaseFacts" in str(item) for item in client.calls[-1]["messages"])


def test_compound_request_fans_out_domain_scoped_subrequests_and_merges_once() -> None:
    class CapturingAgent:
        def __init__(self, agent_type: AgentType) -> None:
            self.agent_type = agent_type
            self.messages: list[str] = []

        async def handle(self, request: Request) -> AgentResponse:
            self.messages.append(request.message)
            await asyncio.sleep(0)
            return AgentResponse(
                agent_type=self.agent_type,
                content=f"{self.agent_type.value}已核实。",
                success=True,
            )

    class JoiningComposer:
        async def compose(self, _request, responses, primary_agent=None):
            del primary_agent
            return "；".join(response.content for response in responses)

    async def knowledge_search(
        query: str,
        top_k: int,
        domain: str | None = None,
    ) -> list[dict[str, Any]]:
        assert top_k == 5
        assert domain in {"order", "after_sales"}
        return [{"title": "测试业务规则", "content": f"与“{query}”相关的已确认规则。", "score": 1.0}]

    order_agent = CapturingAgent(AgentType.FULFILLMENT)
    after_sales_agent = CapturingAgent(AgentType.REFUND_RISK)
    orchestrator = AgentOrchestrator(
        agent_pool={
            AgentType.FULFILLMENT: order_agent,
            AgentType.REFUND_RISK: after_sales_agent,
        },
        composer=JoiningComposer(),
        knowledge_search=knowledge_search,
    )

    result = _run(
        orchestrator.run(
            Request(
                message="订单里漏发了赠品，同时我要申请退款。",
                user_id="consumer-demo",
                intent=IntentCategory.MISSING_ITEM,
                secondary_intents=[IntentCategory.RETURN_REFUND],
                sub_requests=[
                    {"id": "order-1", "domain": "order", "intent": "missing_item", "text": "订单里漏发了赠品"},
                    {"id": "refund-1", "domain": "after_sales", "intent": "return_refund", "text": "同时我要申请退款"},
                ],
                intent_diagnostics={
                    "scoped_retrieval_queries": {
                        "order": "订单里漏发了赠品",
                        "after_sales": "同时我要申请退款",
                    }
                },
            )
        )
    )

    assert order_agent.messages == ["订单里漏发了赠品"]
    assert after_sales_agent.messages == ["同时我要申请退款"]
    assert result.response == "order已核实。；after_sales已核实。"
    assert set(result.agent_types) == {AgentType.FULFILLMENT, AgentType.REFUND_RISK}
    fanout = [trace for trace in result.tool_traces if trace.get("stage") == "agent_fanout"]
    assert {trace["agent_type"] for trace in fanout} == {"order", "after_sales"}
    assert {tuple(trace["sub_request_ids"]) for trace in fanout} == {("order-1",), ("refund-1",)}
