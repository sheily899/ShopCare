import asyncio
from functools import wraps
from types import SimpleNamespace

from evaluation.metrics import _forbidden_claim_detected, _sensitive_request_detected


def _run_async(test_function):
    @wraps(test_function)
    def wrapper(*args, **kwargs):
        return asyncio.run(test_function(*args, **kwargs))

    return wrapper


def test_safety_refusal_is_not_marked_as_sensitive_request():
    response = "请不要在公开渠道粘贴完整密码、验证码或 API Key；错误码可以发我，我会先给出排查建议。"

    assert not _sensitive_request_detected("密码", response)
    assert not _sensitive_request_detected("验证码", response)


def test_safety_metric_detects_explicit_sensitive_request():
    assert _sensitive_request_detected("密码", "请提供登录密码后我再继续处理。")
    assert _sensitive_request_detected("验证码", "请把短信验证码发我。")


def test_forbidden_claim_negation_is_not_treated_as_claim():
    response = "我不能补全未验证的企业政策，也不能声称退款已经成功；请以官方页面为准。"

    assert not _forbidden_claim_detected("未验证的企业政策", response)
    assert not _forbidden_claim_detected("退款已经成功", response)


@_run_async
async def test_agent_response_exposes_provider_usage():
    from agents.agent_orchestrator import AgentType, GeneralAgent, Request

    class FakeMessages:
        async def create(self, **kwargs):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="已处理")],
                usage=SimpleNamespace(input_tokens=12, output_tokens=8),
            )

    agent = GeneralAgent(client=SimpleNamespace(messages=FakeMessages()))
    result = await agent.handle(Request(message="你好"))

    assert result.agent_type is AgentType.GENERAL
    assert result.success is True
    assert result.usage == {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}


@_run_async
async def test_composer_usage_is_scoped_to_request():
    from agents.agent_orchestrator import AgentResponse, AgentType, Request, ResponseComposer

    class FakeMessages:
        async def create(self, **kwargs):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="已合并")],
                usage=SimpleNamespace(input_tokens=30, output_tokens=10),
            )

    request = Request(message="技术问题和账单问题", request_id="usage-1")
    composer = ResponseComposer(client=SimpleNamespace(messages=FakeMessages()))
    await composer.compose(
        request,
        [
            AgentResponse(AgentType.TECHNICAL, "技术结论", True),
            AgentResponse(AgentType.BILLING, "账单结论", True),
        ],
        primary_agent=AgentType.TECHNICAL,
    )

    assert composer.usage_for(request.request_id) == {
        "input_tokens": 30,
        "output_tokens": 10,
        "total_tokens": 40,
    }
    assert composer.usage_for(request.request_id) == {}


@_run_async
async def test_orchestrator_breaks_usage_down_by_stage_and_keeps_flat_total():
    from agents.agent_orchestrator import (
        AgentOrchestrator,
        AgentResponse,
        AgentType,
        Request,
        ResponseComposer,
        RoutingDecision,
    )
    from intent.intent_recognizer import IntentCategory

    class FakeAgent:
        def __init__(self, agent_type, usage):
            self.agent_type = agent_type
            self.usage = usage

        async def handle(self, request, **kwargs):
            return AgentResponse(
                agent_type=self.agent_type,
                content=f"{self.agent_type.value} result",
                success=True,
                usage=self.usage,
            )

    class FakeComposer(ResponseComposer):
        async def compose(self, request, responses, primary_agent=None):
            self._usage_by_request[request.request_id] = {
                "input_tokens": 5,
                "output_tokens": 3,
                "total_tokens": 8,
            }
            return "merged result"

    intent_usage = {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9}
    general_usage = {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
    billing_usage = {"input_tokens": 11, "output_tokens": 5, "total_tokens": 16}
    request = Request(
        message="综合问题",
        request_id="stage-usage",
        intent=IntentCategory.RETURN_REFUND,
        intent_diagnostics={"usage": intent_usage},
    )

    async def knowledge_search(query, top_k, domain=None):
        assert top_k == 5
        assert domain == "after_sales"
        return [{"title": "规则", "content": f"已检索到与{query}相关的规则。", "score": 1.0}]

    orchestrator = AgentOrchestrator(
        agent_pool={
            AgentType.GENERAL: FakeAgent(AgentType.GENERAL, general_usage),
            AgentType.BILLING: FakeAgent(AgentType.BILLING, billing_usage),
        },
        composer=FakeComposer(),
        knowledge_search=knowledge_search,
    )

    result = await orchestrator.run_parallel(
        request,
        RoutingDecision(
            primary_agent=AgentType.BILLING,
            supporting_agents=[AgentType.GENERAL],
        ),
    )

    assert result.usage == {
        "input_tokens": 33,
        "output_tokens": 14,
        "total_tokens": 47,
    }
    assert result.usage_by_stage == {
        "intent": intent_usage,
        "agent_after_sales": billing_usage,
        "agent_consult": general_usage,
        "composer": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
    }
