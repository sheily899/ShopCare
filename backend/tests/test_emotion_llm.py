import asyncio
from types import SimpleNamespace

from intent.emotion_llm import analyze_emotion, classify_answers


def test_emotion_category_uses_fixed_priority_not_model_selected_label() -> None:
    assert classify_answers({"threat": True, "anger": True, "dissatisfaction": True, "anxiety": True}) == "威胁升级"
    assert classify_answers({"anger": True, "dissatisfaction": True, "anxiety": True}) == "愤怒"
    assert classify_answers({"dissatisfaction": True, "anxiety": True}) == "不满"
    assert classify_answers({"anxiety": True}) == "焦虑"
    assert classify_answers({}) == "平静"


class _EmotionClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.messages = self
        self.call = None

    async def create(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(content=[SimpleNamespace(text=self.text)])


def test_emotion_llm_reads_recent_context_and_requires_literal_evidence() -> None:
    client = _EmotionClient(
        '{"threat":{"answer":false,"evidence":""},'
        '"anger":{"answer":false,"evidence":""},'
        '"dissatisfaction":{"answer":true,"evidence":"一直没处理"},'
        '"anxiety":{"answer":false,"evidence":""}}'
    )

    result = asyncio.run(analyze_emotion(client, "今天还是没处理", recent="消费者: 一直没处理"))

    assert result["emotion"] == "不满"
    assert result["source"] == "llm"
    assert result["evidence"] == {"dissatisfaction": "一直没处理"}
    assert "消费者: 一直没处理" in client.call["messages"][0]["content"]


def test_malformed_boolean_or_unverifiable_evidence_degrades_to_rule_emotion() -> None:
    client = _EmotionClient(
        '{"threat":{"answer":"false","evidence":""},'
        '"anger":{"answer":false,"evidence":""},'
        '"dissatisfaction":{"answer":false,"evidence":""},'
        '"anxiety":{"answer":false,"evidence":""}}'
    )

    result = asyncio.run(analyze_emotion(client, "这都第三次了", fallback_emotion="upset"))

    assert result["emotion"] == "不满"
    assert result["source"] == "fallback"
    assert result["degraded"] is True
    assert "字段格式错误" in result["degradation_reason"]
