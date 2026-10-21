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
    assert item.intent_correct is True
    assert item.route_correct is True
    assert item.knowledge_cited is True
    assert item.retrieval_hit is True
    assert item.retrieval_recall == 1.0
    assert item.retrieval_precision == 1.0
    assert item.retrieval_precision_at_1 == 1.0
    assert item.retrieval_precision_at_3 == 1.0
    assert item.retrieval_precision_at_5 == 1.0
    assert item.retrieval_recall_at_5 == 1.0
    assert item.retrieval_f1 == 1.0
    assert item.retrieval_mrr == 1.0
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
    assert item.compound_route_correct is True
    assert item.explicit_cross_domain is True
    summary = summarize([item], [result])
    assert summary["explicit_cross_domain_denominator"] == 1


def test_domain_intent_accepts_more_specific_sub_intent():
    result = _result(content="排查页面错误")
    result["response"]["intent"] = "technical_crash"
    result["response"]["agent_type"] = "technical"
    item = evaluate_case(
        _case(expected_intent="technical", acceptable_intents=["technical"], expected_agent="technical"),
        result,
    )
    assert item.intent_correct is True
    assert item.route_correct is True


def test_percentiles_are_client_round_trip_values():
    assert percentile([10, 20, 30, 40], 0.5) == 20.0
    assert percentile([10, 20, 30, 40], 0.95) == 40.0
    assert percentile([], 0.95) is None


def test_cost_missing_usage_is_unavailable_not_zero():
    item = evaluate_case(_case(), _result())
    assert summarize([item], [_result()])["cost"]["status"] == "unavailable"


def test_cost_summary_breaks_down_stage_usage_without_changing_total_cost():
    item = evaluate_case(_case(), _result())
    result = _result()
    result["usage"] = {
        "input_tokens": 30,
        "output_tokens": 10,
        "total_tokens": 40,
    }
    result["response"]["usage_by_stage"] = {
        "intent": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        "agent_billing": {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
    }

    summary = summarize([item], [result])

    assert summary["cost"]["total_tokens"] == 40.0
    assert summary["cost"]["by_stage"] == {
        "intent": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12, "requests": 1},
        "agent_billing": {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28, "requests": 1},
    }


def test_summary_separates_intent_route_citation_and_latency_percentiles():
    item = evaluate_case(_case(), _result())
    summary = summarize([item], [_result()])
    assert summary["intent_accuracy"] == 1.0
    assert summary["route_accuracy"] == 1.0
    assert summary["knowledge_citation_rate"] == 1.0
    assert summary["retrieval_recall"] == 1.0
    assert summary["retrieval_precision"] == 1.0
    assert summary["retrieval_precision_at_1"] == 1.0
    assert summary["retrieval_precision_at_3"] == 1.0
    assert summary["retrieval_precision_at_5"] == 1.0
    assert summary["retrieval_recall_at_5"] == 1.0
    assert summary["retrieval_f1"] == 1.0
    assert summary["retrieval_mrr"] == 1.0
    assert summary["client_round_trip_ms"]["p99"] == 120.0


def test_e2e_contract_accuracy_requires_all_business_gates():
    """端到端契约通过必须同时满足主链路、知识和安全门槛。"""

    passing = evaluate_case(_case(), _result())
    assert passing.e2e_contract_pass is True
    assert passing.e2e_failure_reasons == ()

    missing_retrieval = _result()
    missing_retrieval["trace"] = {"trace": {"tool_traces": []}}
    failed = evaluate_case(_case(), missing_retrieval)
    assert failed.e2e_contract_pass is False
    assert "knowledge_retrieval" in failed.e2e_failure_reasons

    summary = summarize(
        [passing, failed],
        [_result(), missing_retrieval],
    )
    assert summary["e2e_contract_accuracy"] == 0.5
    assert summary["e2e_contract_passed"] == 1
    assert summary["e2e_contract_denominator"] == 2
    assert summary["e2e_semantic_accuracy"]["status"] == "unavailable"
