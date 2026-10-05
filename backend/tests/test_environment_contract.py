from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"


EXPECTED_REQUIREMENTS = {
    "anthropic": "0.40.0",
    "fastapi": "0.115.5",
    "uvicorn[standard]": "0.32.1",
    "pydantic": "2.10.3",
    "python-multipart": "0.0.12",
    "redis": "5.2.1",
    "chromadb": "0.5.23",
    "posthog": "2.5.0",
    "sentence-transformers": "3.3.1",
    "transformers": "4.46.3",
    "prometheus-client": "0.21.1",
    "httpx": "0.28.1",
    "python-dotenv": "1.0.1",
    "pytest": "8.3.4",
}


EXPECTED_ENV_KEYS = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_BASE_URL",
    "REDIS_URL",
    "REDIS_PASSWORD",
    "CHROMA_HOST",
    "CHROMA_PORT",
    "CHROMA_PERSIST_DIRECTORY",
    "ANONYMIZED_TELEMETRY",
    "CUSTOMEROPS_KB_COLLECTION",
    "CUSTOMEROPS_EMBEDDING_MODEL",
    "CUSTOMEROPS_SKILLS_DIR",
    "CUSTOMEROPS_SKILLS_MAX_PROMPT_CHARS",
    "LOG_LEVEL",
    "PROMETHEUS_PORT",
    "MONITOR_INTERVAL",
    "ALERT_WEBHOOK_URL",
    "CUSTOMEROPS_TOOL_TRACE_MAX",
    "CUSTOMEROPS_COMPOSER_MAX_TOKENS",
    "CUSTOMEROPS_COMPOSER_TEMPERATURE",
    "EVAL_BASELINE_PATH",
}


def _parse_requirements() -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", maxsplit=1)
        parsed[name.strip()] = version.strip()
    return parsed


def _parse_env_keys() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", maxsplit=1)
        values[key.strip()] = value.strip()
    return values


def test_requirements_lock_the_exact_fourteen_core_dependencies() -> None:
    assert REQUIREMENTS.is_file()
    assert _parse_requirements() == EXPECTED_REQUIREMENTS


def test_requirements_do_not_introduce_gpu_or_cuda_dependencies() -> None:
    content = REQUIREMENTS.read_text(encoding="utf-8").lower()
    assert not re.search(r"cuda|cudnn|nvidia|\bgpu\b|torch", content)


def test_env_example_has_all_core_model_monitoring_and_evaluation_keys() -> None:
    assert ENV_EXAMPLE.is_file()
    env_values = _parse_env_keys()
    assert EXPECTED_ENV_KEYS <= env_values.keys()


def test_env_example_contains_only_safe_placeholders() -> None:
    content = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert "sk-ant-" not in content
    assert "-----BEGIN" not in content
    assert "customerops123" not in content
    assert not re.search(r"AKIA[0-9A-Z]{16}", content)
    assert not re.search(r"gh[pousr]_[A-Za-z0-9_]{20,}", content)
