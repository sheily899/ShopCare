"""ShopCare 100 案例单次评测运行器与报告输出。"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import httpx

from evaluation.corpus import EVALUATION_COLLECTION_NAME, load_cases
from evaluation.metrics import CaseEvaluation, evaluate_case, summarize


class EvaluationRunner:
    """顺序发送 100 条案例，严格区分网络故障和业务失败。

    Args:
        base_url: FastAPI 地址。
        collection_name: 预期的独立评测 collection。
        client: 可注入的异步 HTTP client，测试时不访问公网。
    """

    def __init__(self, base_url: str, collection_name: str, client: Any) -> None:
        self.base_url = base_url.rstrip("/")
        self.collection_name = collection_name
        self.client = client

    async def _preflight(self) -> Mapping[str, Any]:
        """检查服务健康状态和评测 collection，失败立即停止。"""

        response = await self.client.get(f"{self.base_url}/health")
        if int(getattr(response, "status_code", 0)) != 200:
            raise RuntimeError(f"/health 不可用: status={getattr(response, 'status_code', None)}")
        payload = response.json()
        actual = str(((payload.get("knowledge_base") or {}).get("collection_name")) or "")
        if actual != self.collection_name:
            raise RuntimeError(
                f"评测 collection 不匹配: expected={self.collection_name} actual={actual}；"
                "请先准备 evaluation_benchmark，禁止把产品库当评测库。"
            )
        return payload

    async def run(self, cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """运行案例并返回原始结果与确定性指标。

        Args:
            cases: 按文件顺序排列的评测案例。

        Returns:
            包含 preflight、cases、evaluations 和 summary 的报告对象。
        """

        health = await self._preflight()
        raw_results = []
        evaluations = []
        for case in cases:
            raw = await self._run_one(case)
            raw_results.append(raw)
            evaluations.append(evaluate_case(case, raw))
        return {
            "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6],
            "collection_name": self.collection_name,
            "preflight": health,
            "cases": raw_results,
            "evaluations": [evaluation.to_dict() for evaluation in evaluations],
            "summary": summarize(evaluations, raw_results),
        }

    async def _run_one(self, case: Mapping[str, Any]) -> Dict[str, Any]:
        """执行一条 chat 和一次 trace 读取，不自动重试。"""

        started = time.perf_counter()
        case_id = str(case.get("case_id") or "")
        payload = {
            "message": str(case.get("message") or ""),
            "user_id": str(case.get("user_id") or f"eval-{case_id}"),
            "conv_id": str(case.get("conv_id") or f"eval-{case_id}"),
        }
        try:
            response = await self.client.post(f"{self.base_url}/chat", json=payload)
        except Exception as exc:
            return {
                "case_id": case_id,
                "status": "infrastructure_failure",
                "error": f"chat request failed: {exc}",
                "round_trip_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        status_code = int(getattr(response, "status_code", 0))
        raw: Dict[str, Any] = {
            "case_id": case_id,
            "status": "completed" if status_code == 200 else "business_failure",
            "http_status": status_code,
            "round_trip_ms": elapsed,
        }
        try:
            body = response.json()
        except Exception as exc:
            raw.update({"status": "infrastructure_failure", "error": f"invalid chat JSON: {exc}"})
            return raw
        if status_code != 200:
            raw.update({"status": "business_failure", "error": body})
            return raw
        raw["response"] = body
        raw["usage"] = body.get("usage") if isinstance(body, Mapping) else None
        request_id = str(body.get("request_id") or "") if isinstance(body, Mapping) else ""
        if request_id:
            try:
                trace_response = await self.client.get(f"{self.base_url}/trace/tool/{request_id}")
                if int(getattr(trace_response, "status_code", 0)) == 200:
                    raw["trace"] = trace_response.json()
                else:
                    raw["trace_error"] = f"trace status={getattr(trace_response, 'status_code', None)}"
            except Exception as exc:
                raw["trace_error"] = str(exc)
        return raw


def write_reports(report: Mapping[str, Any], output_root: str | Path = "evaluation/reports") -> Path:
    """写入 summary.json、cases.jsonl 和 report.md。"""

    run_dir = Path(output_root) / str(report["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {key: report[key] for key in ("run_id", "collection_name", "preflight", "summary")},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (run_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in report["cases"]:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    summary = report["summary"]
    markdown = "\n".join(
        [
            f"# ShopCare 评测报告 {report['run_id']}",
            "",
            f"- 评测 collection：`{report['collection_name']}`",
            f"- 案例数：{summary['total_cases']}；基础设施失败：{summary['infrastructure_failures']}；业务失败：{summary['business_failures']}",
            f"- 端点成功率：{summary['endpoint_success_rate']}",
            f"- 意图识别准确率（含领域子意图兼容）：{summary['intent_accuracy']}",
            f"- 路由准确率：{summary['route_accuracy']}",
            f"- 知识库引用率：{summary['knowledge_citation_rate']}（仅统计 knowledge_required 案例）",
            f"- 知识库 Precision@1/@3/@5：{summary['retrieval_precision_at_1']} / {summary['retrieval_precision_at_3']} / {summary['retrieval_precision_at_5']}",
            f"- 知识库 Recall@5 / F1 / MRR：{summary['retrieval_recall_at_5']} / {summary['retrieval_f1']} / {summary['retrieval_mrr']}",
            f"- 复合请求路由准确率：{summary['compound_routing_accuracy']}；明确跨域路由准确率：{summary['explicit_cross_domain_routing_accuracy']}；事实合成诊断值：{summary['compound_resolution_rate']}；复合分母={summary['compound_denominator']}，明确跨域分母={summary['explicit_cross_domain_denominator']}",
            f"- 安全规则筛查通过率（需人工复核）：{summary['safety_pass_rate']}",
            f"- 客户端往返延迟（ms）：P50={summary['client_round_trip_ms']['p50']}；P95={summary['client_round_trip_ms']['p95']}；P99={summary['client_round_trip_ms']['p99']}",
            f"- 成本：{summary['cost']}",
            "",
            "说明：本报告只做一次顺序运行；没有重试、A/B、LLM judge 或 prompt 优化。事实覆盖和安全字符串筛查仅作诊断，不能单独作为产品质量结论。",
        ]
    )
    (run_dir / "report.md").write_text(markdown + "\n", encoding="utf-8")
    return run_dir


async def run_from_files(
    base_url: str,
    cases_path: str,
    collection_name: str = EVALUATION_COLLECTION_NAME,
    output_root: str = "evaluation/reports",
) -> Path:
    """使用真实 HTTP client 执行一次文件型评测并写报告。"""

    cases = load_cases(cases_path)
    if len(cases) != 100:
        raise ValueError(f"正式评测必须使用完整 100 条案例，当前为 {len(cases)} 条")
    async with httpx.AsyncClient(timeout=60.0) as client:
        report = await EvaluationRunner(base_url, collection_name, client).run(cases)
    return write_reports(report, output_root)


def main() -> None:
    """解析命令行并执行一次评测。"""

    parser = argparse.ArgumentParser(description="执行 ShopCare 100 案例单次评测")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases-path", default="evaluation/datasets/shopcare_evaluation_cases.json")
    parser.add_argument("--collection-name", default=EVALUATION_COLLECTION_NAME)
    parser.add_argument("--output-root", default="evaluation/reports")
    args = parser.parse_args()
    output = asyncio.run(run_from_files(args.base_url, args.cases_path, args.collection_name, args.output_root))
    print(f"evaluation report written: {output}")


if __name__ == "__main__":
    main()
