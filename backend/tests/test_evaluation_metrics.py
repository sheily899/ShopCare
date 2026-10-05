"""评测核心指标的离线确定性测试。"""

from evaluation.metrics import evaluate_case, percentile, summarize


def _case(**overrides):
    value = {
        "case_id": "case-1",
        "expected_intent": "refund",
        "acceptable_intents": ["refund"],
        "expected_agent": "billing",
        "knowledge_required": True,
        "expected_retrieval_chunks": ["refund#chunk_001"],
        "required_facts": ["原路退回", "不能承诺固定到账日期"],
        "forbidden_claims": ["一定到账"],
        "safety_constraints": {"must_not_request": ["password"]},
        "expected_escalated": False,
        "compound": {"is_compound": False},
    }
    value.update(overrides)
    return value


def _result(content="退款原则上原路退回，不能承诺固定到账日期。", **overrides):
    value = {
        "status": "completed",
        "http_status": 200,
        "round_trip_ms": 120.0,
        "response": {
            "response": content,
            "intent": "refund",
            "agent_type": "billing",
            "knowledge_used": True,
            "escalated": False,
            "supporting_agents": [],
        },
        "trace": {"trace": {"tool_traces": [{
            "retrieval_stage": "knowledge_prefetch",
            "retrieved_chunks": [{"source_chunk_id": "refund#chunk_001"}],
        }]}},
    }
    value.update(overrides)
    return value


def test_case_metrics_cover_route_knowledge_and_safety():
    item = evaluate_case(_case(), _result())
    assert item.endpoint_success is True
    assert item.intent_route_correct is True
    assert item.retrieval_hit is True
    assert item.knowledge_effective is True
    assert item.safety_pass is True


def test_infrastructure_failure_is_not_counted_as_business_failure():
    item = evaluate_case(_case(), {"status": "infrastructure_failure", "round_trip_ms": 99})
    summary = summarize([item], [{"status": "infrastructure_failure"}])
    assert item.intent_route_correct is None
    assert summary["infrastructure_failures"] == 1
    assert summary["business_failures"] == 0


def test_compound_and_explicit_cross_domain_denominators_are_separate():
    case = _case(
        compound={"is_compound": True},
        expected_supporting_agents=["technical"],
        expected_secondary_intents=["technical_login"],
    )
    result = _result()
    result["response"]["supporting_agents"] = ["technical"]
    item = evaluate_case(case, result)
    assert item.compound_resolved is True
    assert item.explicit_cross_domain is True
    summary = summarize([item], [result])
    assert summary["explicit_cross_domain_denominator"] == 1


def test_percentiles_are_client_round_trip_values():
    assert percentile([10, 20, 30, 40], 0.5) == 20.0
    assert percentile([10, 20, 30, 40], 0.95) == 40.0
    assert percentile([], 0.95) is None


def test_cost_missing_usage_is_unavailable_not_zero():
    item = evaluate_case(_case(), _result())
    assert summarize([item], [_result()])["cost"]["status"] == "unavailable"
