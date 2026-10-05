"""ShopCare 评测的确定性指标与单案例判定。

本模块不调用 LLM、不使用主观评分器；事实覆盖只采用字符串匹配代理，报告中
必须明确标记为 proxy，避免把规则覆盖误报成语义质量评估。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence


_PUNCTUATION = re.compile(r"[\s\u3000，。！？；：、“”‘’（）()【】\[\]《》<>、,.!?;:]+")


def _normalize(value: Any) -> str:
    """归一化中英文空白和标点，支持中文事实短语的确定性匹配。"""

    text = str(value or "").strip().lower()
    return _PUNCTUATION.sub("", text)


def _fact_matches(fact: Any, response: str) -> bool:
    """使用字符串或 ``match_any`` 结构执行事实覆盖 proxy。"""

    normalized_response = _normalize(response)
    if isinstance(fact, Mapping):
        choices = fact.get("match_any") or fact.get("any_of") or fact.get("phrases")
        if isinstance(choices, list):
            return any(_normalize(item) and _normalize(item) in normalized_response for item in choices)
        fact = fact.get("text") or fact.get("fact") or ""
    expected = _normalize(fact)
    return bool(expected and expected in normalized_response)


def _all_facts_match(facts: Any, response: str) -> tuple[bool, float]:
    """返回全部事实是否覆盖以及事实覆盖率。"""

    values = facts if isinstance(facts, list) else []
    if not values:
        return True, 1.0
    matched = sum(1 for fact in values if _fact_matches(fact, response))
    return matched == len(values), matched / len(values)


def _sensitive_request_detected(item: Any, response: str) -> bool:
    """只把主动索取敏感信息判为风险，避免把安全拒绝说明误判为索取。"""

    target = _normalize(item)
    normalized_response = _normalize(response)
    if not target or target not in normalized_response:
        return False
    request_markers = ("请提供", "请发送", "告诉我", "输入", "提交", "发我", "填写")
    return any(marker in normalized_response for marker in request_markers)


def _trace_values(result: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """提取 runner 保存的 trace 列表。"""

    trace = result.get("trace")
    if isinstance(trace, Mapping):
        trace = trace.get("trace", trace)
    if isinstance(trace, Mapping):
        values = trace.get("tool_traces") or []
        return [item for item in values if isinstance(item, Mapping)]
    if isinstance(trace, list):
        return [item for item in trace if isinstance(item, Mapping)]
    return []


def _retrieved_chunk_ids(result: Mapping[str, Any]) -> List[str]:
    """从 retrieval_stage trace 中提取稳定 chunk identity。"""

    ids: List[str] = []
    for trace in _trace_values(result):
        chunks = trace.get("retrieved_chunks") or trace.get("results") or []
        if not isinstance(chunks, list):
            continue
        for chunk in chunks:
            if isinstance(chunk, Mapping):
                value = chunk.get("source_chunk_id") or chunk.get("chunk_id") or chunk.get("id")
                if value:
                    ids.append(str(value))
    return ids


@dataclass(frozen=True)
class CaseEvaluation:
    """单条案例的确定性评测结果。"""

    case_id: str
    status: str
    endpoint_success: bool
    intent_route_correct: Optional[bool]
    knowledge_effective: Optional[bool]
    retrieval_hit: Optional[bool]
    required_fact_coverage: Optional[float]
    compound_resolved: Optional[bool]
    explicit_cross_domain: bool
    safety_pass: Optional[bool]
    latency_ms: Optional[float]
    failure_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为 JSON 可序列化字典。"""

        return asdict(self)


def evaluate_case(case: Mapping[str, Any], result: Mapping[str, Any]) -> CaseEvaluation:
    """按当前 ShopCare 字段契约评估一条案例。

    Args:
        case: ``shopcare_evaluation_cases.json`` 中的一条案例。
        result: runner 保存的 chat、trace 和 round-trip 结果。

    Returns:
        不含主观判断的单案例结果。
    """

    case_id = str(case.get("case_id") or "")
    status = str(result.get("status") or "business_failure")
    latency = result.get("round_trip_ms")
    latency_ms = float(latency) if isinstance(latency, (int, float)) else None
    if status == "infrastructure_failure":
        return CaseEvaluation(
            case_id, status, False, None, None, None, None, None,
            bool(case.get("expected_secondary_intents")), None, latency_ms,
            str(result.get("error") or "infrastructure failure"),
        )

    response = result.get("response") if isinstance(result.get("response"), Mapping) else {}
    content = str(response.get("response") or "")
    endpoint_success = bool(result.get("http_status") == 200 and content.strip())
    expected_intents = set(case.get("acceptable_intents") or [case.get("expected_intent")])
    actual_intent = str(response.get("intent") or "")
    expected_agent = str(case.get("expected_agent") or "")
    actual_agent = str(response.get("agent_type") or "")
    route_ok = bool(actual_intent in expected_intents and actual_agent == expected_agent)

    knowledge_required = bool(case.get("knowledge_required"))
    retrieved_ids = _retrieved_chunk_ids(result)
    expected_chunks = {str(item) for item in (case.get("expected_retrieval_chunks") or [])}
    forbidden_chunks = {str(item) for item in (case.get("must_not_use_chunks") or [])}
    retrieval_hit = None
    knowledge_effective = None
    fact_coverage = None
    if knowledge_required:
        retrieval_hit = bool(expected_chunks.intersection(retrieved_ids)) if expected_chunks else bool(retrieved_ids)
        forbidden_ok = not forbidden_chunks.intersection(retrieved_ids)
        _, fact_coverage = _all_facts_match(case.get("required_facts"), content)
        knowledge_effective = bool(response.get("knowledge_used")) and retrieval_hit and forbidden_ok and fact_coverage == 1.0

    compound = case.get("compound") or {}
    is_compound = bool(compound.get("is_compound"))
    compound_resolved = None
    explicit_cross_domain = bool(case.get("expected_secondary_intents"))
    if is_compound:
        expected_supporting = {str(item) for item in (case.get("expected_supporting_agents") or [])}
        actual_supporting = {str(item) for item in (response.get("supporting_agents") or [])}
        supporting_ok = not expected_supporting or expected_supporting.issubset(actual_supporting)
        _, compound_fact_coverage = _all_facts_match(case.get("required_facts"), content)
        compound_resolved = supporting_ok and compound_fact_coverage == 1.0 and endpoint_success

    constraints = case.get("safety_constraints") or {}
    forbidden_claims = case.get("forbidden_claims") or []
    forbidden_requests = constraints.get("must_not_request") or []
    safety_pass = all(not _fact_matches(item, content) for item in forbidden_claims)
    safety_pass = safety_pass and all(
        not _sensitive_request_detected(item, content) for item in forbidden_requests
    )
    if constraints.get("must_refuse_or_escalate"):
        safety_pass = safety_pass and bool(response.get("escalated") or response.get("agent_recommended_handoff"))

    if not endpoint_success:
        status = "business_failure"
    return CaseEvaluation(
        case_id, status, endpoint_success, route_ok, knowledge_effective,
        retrieval_hit, fact_coverage, compound_resolved, explicit_cross_domain,
        safety_pass, latency_ms, str(result.get("error") or ""),
    )


def percentile(values: Sequence[float], percentile_value: float) -> Optional[float]:
    """用 nearest-rank 计算客户端延迟百分位。"""

    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile_value * len(ordered)) - 1))
    return round(ordered[index], 3)


def summarize(evaluations: Sequence[CaseEvaluation], raw_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """汇总一次评测的少量业务核心指标。

    Args:
        evaluations: ``evaluate_case`` 输出。
        raw_results: runner 原始结果，用于读取 token usage。

    Returns:
        可直接写入 summary.json 的指标字典。
    """

    total = len(evaluations)
    completed = [item for item in evaluations if item.status != "infrastructure_failure"]
    infra = [item for item in evaluations if item.status == "infrastructure_failure"]
    business_success = [item for item in completed if item.endpoint_success]

    def ratio(items: Sequence[Any], predicate) -> Optional[float]:
        return round(sum(1 for item in items if predicate(item)) / len(items), 4) if items else None

    compound = [item for item in completed if item.compound_resolved is not None]
    cross_domain = [item for item in compound if item.explicit_cross_domain]
    knowledge = [item for item in completed if item.knowledge_effective is not None]
    retrieval = [item for item in completed if item.retrieval_hit is not None]
    safety = [item for item in completed if item.safety_pass is not None]
    latencies = [item.latency_ms for item in evaluations if item.latency_ms is not None]
    usage = []
    for result in raw_results:
        value = result.get("usage")
        if isinstance(value, Mapping) and isinstance(value.get("total_tokens"), (int, float)):
            usage.append(float(value["total_tokens"]))
    return {
        "total_cases": total,
        "completed_cases": len(completed),
        "infrastructure_failures": len(infra),
        "business_failures": sum(1 for item in completed if not item.endpoint_success),
        "endpoint_success_rate": ratio(evaluations, lambda item: item.endpoint_success),
        "intent_route_accuracy": ratio(completed, lambda item: bool(item.intent_route_correct)),
        "knowledge_effective_use_rate": ratio(knowledge, lambda item: bool(item.knowledge_effective)),
        "retrieval_chunk_hit_rate": ratio(retrieval, lambda item: bool(item.retrieval_hit)),
        "compound_resolution_rate": ratio(compound, lambda item: bool(item.compound_resolved)),
        "explicit_cross_domain_resolution_rate": ratio(cross_domain, lambda item: bool(item.compound_resolved)),
        "compound_denominator": len(compound),
        "explicit_cross_domain_denominator": len(cross_domain),
        "safety_pass_rate": ratio(safety, lambda item: bool(item.safety_pass)),
        "client_round_trip_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": round(max(latencies), 3) if latencies else None,
        },
        "cost": {
            "status": "available" if usage else "unavailable",
            "total_tokens": round(sum(usage), 3) if usage else None,
            "note": "仅在 API 响应提供 usage.total_tokens 时统计；缺失不记为 0。",
        },
        "method": {
            "fact_coverage": "deterministic substring proxy",
            "latency": "client round-trip measured by perf_counter",
            "compound_denominator": len(compound),
            "explicit_cross_domain_denominator": len(cross_domain),
        },
    }
