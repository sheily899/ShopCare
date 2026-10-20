from backend.intent.beauty_intent_registry import (
    BEAUTY_INTENT_BY_ID,
    BEAUTY_INTENT_REGISTRY,
    EMBEDDING_TEMPLATES,
    LLM_INTENT_SCHEMA,
    PATTERN_MAPS,
)


def test_beauty_registry_has_nineteen_unique_intents() -> None:
    assert len(BEAUTY_INTENT_REGISTRY) == 19
    assert len(BEAUTY_INTENT_BY_ID) == 19
    assert {item["agent"] for item in BEAUTY_INTENT_REGISTRY} == {"consult", "order", "after_sales"}


def test_every_intent_has_recognition_contract() -> None:
    required = {"intent_id", "name_zh", "schema", "description", "embedding_examples", "exact_patterns", "generic_patterns", "distinction", "boundary"}
    assert all(required <= item.keys() for item in BEAUTY_INTENT_REGISTRY)


def test_runtime_contracts_are_generated_from_registry() -> None:
    ids = {item["intent_id"] for item in BEAUTY_INTENT_REGISTRY}
    assert set(EMBEDDING_TEMPLATES) == ids
    assert set(PATTERN_MAPS) == ids
    assert LLM_INTENT_SCHEMA["enum"] == [item["intent_id"] for item in BEAUTY_INTENT_REGISTRY]
    assert all(PATTERN_MAPS[item_id]["exact"] for item_id in ids)
    assert all(EMBEDDING_TEMPLATES[item_id] for item_id in ids)
