"""生成一次评测的逐案例错误分析报告，不调用模型或后端。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.metrics import _retrieved_chunk_ids, evaluate_case


def analyze(report_dir: str | Path) -> Path:
    directory = Path(report_dir)
    cases = json.loads(Path("evaluation/datasets/shopcare_evaluation_cases.json").read_text(encoding="utf-8"))["cases"]
    raw = [json.loads(line) for line in (directory / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for case, result in zip(cases, raw):
        evaluation = evaluate_case(case, result)
        response = result.get("response") or {}
        ids = list(dict.fromkeys(_retrieved_chunk_ids(result)))
        expected = [str(item) for item in (case.get("expected_retrieval_chunks") or [])]
        ranks = [ids.index(item) + 1 for item in expected if item in ids]
        failures = []
        if evaluation.intent_correct is False:
            failures.append(f"intent: expected={case.get('acceptable_intents') or [case.get('expected_intent')]} actual={response.get('intent')}")
        if evaluation.route_correct is False:
            failures.append(f"route: expected={case.get('expected_agent')} actual={response.get('agent_type')}")
        if evaluation.knowledge_cited is False:
            failures.append("knowledge_citation: knowledge_used=false")
        if evaluation.retrieval_recall_at_5 is not None and evaluation.retrieval_recall_at_5 < 1:
            failures.append(f"retrieval_recall@5: {evaluation.retrieval_recall_at_5:.3f}, expected={expected}, ranks={ranks or 'miss'}")
        if evaluation.safety_pass is False:
            failures.append("safety: failed deterministic policy check")
        if evaluation.explicit_cross_domain and evaluation.compound_resolved is False:
            failures.append("cross_domain: compound resolution check failed")
        if failures:
            rows.append((case, result, evaluation, ids, failures))

    by_metric = {
        "intent": [row for row in rows if row[1].get("response", {}).get("intent") not in (row[0].get("acceptable_intents") or [row[0].get("expected_intent")])],
        "route": [row for row in rows if row[2].route_correct is False],
        "citation": [row for row in rows if row[2].knowledge_cited is False],
        "retrieval": [row for row in rows if row[2].retrieval_recall_at_5 is not None and row[2].retrieval_recall_at_5 < 1],
        "safety": [row for row in rows if row[2].safety_pass is False],
        "cross_domain": [row for row in rows if row[2].explicit_cross_domain and row[2].compound_resolved is False],
    }
    compound_cases = [evaluate_case(c, r) for c, r in zip(cases, raw) if (c.get("compound") or {}).get("is_compound")]
    explicit_cross_domain = [item for item in compound_cases if item.explicit_cross_domain]
    slow = sorted((row for row in [(c, r, evaluate_case(c, r)) for c, r in zip(cases, raw)] if row[2].latency_ms is not None), key=lambda row: row[2].latency_ms or 0, reverse=True)[:10]

    lines = [
        "# ShopCare 评测逐案例错误分析",
        "",
        f"来源：`{directory}`；仅使用已有 cases.jsonl，未重新调用后端或 LLM。",
        "",
        "## 指标失败数量",
        "",
        *(f"- {name}: {len(items)} 条" for name, items in by_metric.items()),
        f"- compound_routing_accuracy: {sum(bool(item.compound_route_correct) for item in compound_cases)}/{len(compound_cases)}",
        f"- explicit_cross_domain_routing_accuracy: {sum(bool(item.compound_route_correct) for item in explicit_cross_domain)}/{len(explicit_cross_domain)}",
        "",
    ]
    labels = {"intent": "意图识别", "route": "路由", "citation": "知识库引用", "retrieval": "知识库召回", "safety": "安全", "cross_domain": "明确跨域"}
    for name, items in by_metric.items():
        lines += [f"## {labels[name]}逐案例分析", ""]
        if not items:
            lines.append("无失败案例。\n")
            continue
        for case, result, evaluation, ids, failures in items:
            lines += [
                f"### `{case.get('case_id')}`",
                f"- 用户问题：{case.get('message')}",
                f"- 实际回答：{str((result.get('response') or {}).get('response') or '').replace(chr(10), ' ')[:240]}",
                f"- 诊断：{'；'.join(failures)}",
                f"- 前五条切片：{ids[:5]}",
                "",
            ]
    lines += ["## 延迟最高案例（前 10）", ""]
    for case, result, evaluation in slow:
        lines.append(f"- `{case.get('case_id')}`：{evaluation.latency_ms} ms，intent={((result.get('response') or {}).get('intent'))}，agent={((result.get('response') or {}).get('agent_type'))}")
    lines += ["", "## 结论", "", "- 本报告把评测失败按指标拆开；同一案例可能同时出现在多个章节。", "- 事实覆盖字符串 proxy 未作为主失败条件，避免把表达差异误判为知识库未使用。", "- 需要结合每条 trace 判断是检索、排序、路由、生成还是评测规则问题。", ""]
    output = directory / "error_analysis.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"error analysis written: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 ShopCare 逐案例评测错误分析")
    parser.add_argument("report_dir")
    args = parser.parse_args()
    analyze(args.report_dir)


if __name__ == "__main__":
    main()
