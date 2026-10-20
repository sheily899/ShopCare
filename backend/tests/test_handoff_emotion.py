"""转人工情绪判断只做 Q1-Q4 是/否，类别由固定优先级归类。"""

import json
import asyncio
from types import SimpleNamespace

from handoff.config import load_handoff_emotion_config
from handoff.emotion import analyze_handoff_emotion, classify_handoff_answers
from handoff.urgency import score_final_urgency


def test_emotion_category_uses_fixed_q1_to_q4_order() -> None:
    assert classify_handoff_answers({"threat": True, "anger": True}) == "威胁升级"
    assert classify_handoff_answers({"anger": True, "dissatisfaction": True}) == "愤怒"
    assert classify_handoff_answers({"dissatisfaction": True, "anxiety": True}) == "不满"
    assert classify_handoff_answers({"anxiety": True}) == "焦虑"
    assert classify_handoff_answers({}) == "平静"


class _EmotionClient:
    def __init__(self, answers):
        self.answers = answers
        self.call = None

        async def create(**kwargs):
            self.call = kwargs
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(self.answers, ensure_ascii=False))])

        self.messages = SimpleNamespace(create=create)


def test_handoff_emotion_reads_questions_from_config_and_only_accepts_current_message_evidence() -> None:
    client = _EmotionClient({
        "threat": {"answer": True, "evidence": "我要投诉平台"},
        "anger": {"answer": True, "evidence": "一直没处理"},
        "dissatisfaction": {"answer": True, "evidence": "一直没处理"},
        "anxiety": {"answer": False, "evidence": ""},
    })

    result = asyncio.run(analyze_handoff_emotion(
        client,
        "一直没处理，我要投诉平台，请帮我转人工",
        recent="消费者前一轮表示很着急",
        config=load_handoff_emotion_config(),
    ))

    assert result["emotion"] == "威胁升级"
    assert result["source"] == "llm"
    assert result["degraded"] is False
    assert result["rule_version"] == load_handoff_emotion_config()["version"]
    assert client.call["temperature"] == 0
    assert "Q1" in client.call["system"] and "Q4" in client.call["system"]


def test_handoff_emotion_rejects_evidence_found_only_in_history_and_degrades() -> None:
    client = _EmotionClient({
        "threat": {"answer": True, "evidence": "我要投诉"},
        "anger": {"answer": False, "evidence": ""},
        "dissatisfaction": {"answer": False, "evidence": ""},
        "anxiety": {"answer": False, "evidence": ""},
    })

    result = asyncio.run(analyze_handoff_emotion(
        client,
        "谢谢，我现在没事了",
        recent="上一轮说：我要投诉",
        config=load_handoff_emotion_config(),
    ))

    assert result["source"] == "fallback_rules"
    assert result["degraded"] is True
    assert result["emotion"] == "平静"


def test_handoff_emotion_rejects_evidence_for_a_negative_answer() -> None:
    client = _EmotionClient({
        "threat": {"answer": False, "evidence": "谢谢"},
        "anger": {"answer": False, "evidence": ""},
        "dissatisfaction": {"answer": False, "evidence": ""},
        "anxiety": {"answer": False, "evidence": ""},
    })

    result = asyncio.run(analyze_handoff_emotion(
        client,
        "谢谢，我现在没事了",
        recent="",
        config=load_handoff_emotion_config(),
    ))

    assert result["degraded"] is True
    assert result["emotion"] == "平静"


def test_missing_emotion_model_uses_configured_rules_and_marks_degraded() -> None:
    result = asyncio.run(analyze_handoff_emotion(
        None,
        "明天没结果我就打12315",
        recent="",
        config=load_handoff_emotion_config(),
    ))

    assert result["emotion"] == "威胁升级"
    assert result["degraded"] is True
    assert result["evidence"]["threat"] == "打12315"


def test_final_urgency_upgrades_to_p0_with_llm_threat_evidence() -> None:
    from handoff.config import load_handoff_urgency_config

    result = score_final_urgency(
        "一直没处理，我要投诉平台，请帮我转人工",
        load_handoff_urgency_config(),
        {"emotion": "威胁升级", "evidence": {"threat": "我要投诉平台"}, "source": "llm"},
    )

    assert result["priority"] == "P0"
    assert result["reasons"][0]["rule_id"] == "emotion.threat_escalation"
    assert result["reasons"][0]["evidence"] == "我要投诉平台"


def test_final_urgency_keeps_abnormal_refund_transfer_separate_from_p1_priority() -> None:
    from handoff.config import load_handoff_urgency_config

    result = score_final_urgency(
        "这笔退款记录疑似重复",
        load_handoff_urgency_config(),
        {"emotion": "平静", "evidence": {}},
        intent="suspected_abnormal_refund",
    )

    assert result["priority"] == "P1"
    assert result["automatic_handoff"] is True


def test_final_urgency_preserves_secondary_abnormal_refund_subrequest() -> None:
    from handoff.config import load_handoff_urgency_config

    result = score_final_urgency(
        "申请退款，同时这笔退款疑似重复",
        load_handoff_urgency_config(),
        {"emotion": "平静", "evidence": {}},
        intent="return_refund",
        intents=["return_refund", "suspected_abnormal_refund"],
    )

    assert result["priority"] == "P1"
    assert result["automatic_handoff"] is True
