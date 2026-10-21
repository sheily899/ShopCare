"""评测运行器的 fake HTTP 离线测试。"""

import asyncio
import json

from evaluation.run_evaluation import EvaluationRunner, write_reports


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, fail_chat=False):
        self.fail_chat = fail_chat
        self.calls = []

    async def get(self, url):
        self.calls.append(("GET", url))
        if url.endswith("/health"):
            return FakeResponse(200, {"status": "healthy", "knowledge_base": {"collection_name": "evaluation_benchmark"}})
        return FakeResponse(200, {"request_id": "abcd1234", "trace": {"tool_traces": []}})

    async def post(self, url, json):
        self.calls.append(("POST", url, json))
        if self.fail_chat:
            raise OSError("offline")
        return FakeResponse(200, {
            "request_id": "abcd1234",
            "response": "完成",
            "intent": "refund",
            "agent_type": "billing",
            "knowledge_used": True,
            "escalated": False,
            "supporting_agents": [],
        })


class FlakyHealthClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.health_attempts = 0

    async def get(self, url):
        if url.endswith("/health") and self.health_attempts == 0:
            self.health_attempts += 1
            self.calls.append(("GET", url))
            return FakeResponse(502, {"detail": "upstream starting"})
        return await super().get(url)


def _case():
    return {
        "case_id": "case-1",
        "message": "退款怎么申请",
        "user_id": "u1",
        "conv_id": "c1",
        "expected_intent": "refund",
        "acceptable_intents": ["refund"],
        "expected_agent": "billing",
        "knowledge_required": False,
        "compound": {"is_compound": False},
        "safety_constraints": {},
    }


def test_runner_preflights_and_sends_one_chat_without_retry():
    client = FakeClient()
    runner = EvaluationRunner("http://fake", "evaluation_benchmark", client)
    report = asyncio.run(runner.run([_case()]))
    assert report["summary"]["total_cases"] == 1
    assert [call[0] for call in client.calls] == ["GET", "POST", "GET"]


def test_runner_retries_transient_health_gateway_error():
    client = FlakyHealthClient()
    runner = EvaluationRunner("http://fake", "evaluation_benchmark", client, preflight_retries=1)
    report = asyncio.run(runner.run([_case()]))
    assert report["summary"]["total_cases"] == 1
    assert [call[0] for call in client.calls] == ["GET", "GET", "POST", "GET"]


def test_runner_separates_network_failure():
    client = FakeClient(fail_chat=True)
    runner = EvaluationRunner("http://fake", "evaluation_benchmark", client)
    report = asyncio.run(runner.run([_case()]))
    assert report["cases"][0]["status"] == "infrastructure_failure"
    assert report["summary"]["infrastructure_failures"] == 1


def test_runner_rejects_product_collection_before_cases():
    client = FakeClient()
    runner = EvaluationRunner("http://fake", "knowledge_base_zh", client)
    try:
        asyncio.run(runner.run([_case()]))
    except RuntimeError as exc:
        assert "不匹配" in str(exc)
    else:
        raise AssertionError("评测不应在 product collection 上运行")


def test_report_writer_creates_three_files(tmp_path):
    client = FakeClient()
    report = asyncio.run(EvaluationRunner("http://fake", "evaluation_benchmark", client).run([_case()]))
    output = write_reports(report, tmp_path)
    assert (output / "summary.json").exists()
    assert (output / "cases.jsonl").exists()
    assert (output / "report.md").exists()
    first_case = json.loads((output / "cases.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first_case["evaluation"]["e2e_contract_pass"] is True
