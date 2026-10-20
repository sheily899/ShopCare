from monitor.token_usage import TokenUsageLedger


def test_ledger_keeps_stage_breakdown_and_aggregates_known_tokens():
    ledger = TokenUsageLedger(max_items=2)
    ledger.record_request(
        "req-1",
        {
            "intent": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            "agent_general": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
        },
        latency_ms=123.4,
    )

    item = ledger.get("req-1")
    assert item["total"] == {"input_tokens": 30, "output_tokens": 7, "total_tokens": 37}
    assert item["stages"]["intent"]["total_tokens"] == 12
    assert ledger.summary()["by_stage"]["agent_general"]["total_tokens"] == 25


def test_ledger_does_not_turn_missing_usage_into_zero():
    ledger = TokenUsageLedger()
    ledger.record_request("req-missing", {}, latency_ms=10)

    item = ledger.get("req-missing")
    assert item["status"] == "unavailable"
    assert item["stages"] == {}
    assert ledger.summary()["missing_usage_requests"] == 1


def test_ledger_is_bounded_and_fail_open():
    ledger = TokenUsageLedger(max_items=1)
    ledger.record_request("old", {"intent": {"total_tokens": 1}})
    ledger.record_request("new", {"intent": {"total_tokens": 2}})

    assert ledger.get("old") is None
    assert ledger.get("new")["total"]["total_tokens"] == 2
    ledger.record_request("bad", {"intent": object()})
