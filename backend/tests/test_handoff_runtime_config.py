"""交接运行配置的路径、完整性与 Docker 部署契约。"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from handoff import config as handoff_config
from handoff import generation as handoff_generation


ROOT = Path(__file__).resolve().parents[2]
CONFIG_LOADERS = (
    handoff_config.load_handoff_copy_config,
    handoff_config.load_handoff_urgency_config,
    handoff_config.load_handoff_emotion_config,
    handoff_generation.load_handoff_action_library,
    handoff_generation.load_handoff_action_version,
    handoff_generation.load_handoff_generation_config,
)


@pytest.fixture(autouse=True)
def clear_handoff_config_caches():
    for loader in CONFIG_LOADERS:
        loader.cache_clear()
    yield
    for loader in CONFIG_LOADERS:
        loader.cache_clear()


def test_handoff_configuration_preflight_uses_explicit_config_directory(monkeypatch, tmp_path):
    config_dir = tmp_path / "runtime-config"
    shutil.copytree(ROOT / "config", config_dir)
    monkeypatch.setenv("SHOPCARE_CONFIG_DIR", str(config_dir))

    versions = handoff_generation.validate_handoff_configuration()

    assert versions == {
        "copy": handoff_config.load_handoff_copy_config()["version"],
        "urgency": handoff_config.load_handoff_urgency_config()["version"],
        "emotion": handoff_config.load_handoff_emotion_config()["version"],
        "actions": handoff_generation.load_handoff_action_version(),
        "generation": handoff_generation.load_handoff_generation_config()["version"],
    }
    assert set(versions) == {"copy", "urgency", "emotion", "actions", "generation"}


def test_handoff_preflight_fails_on_missing_config_before_request_time(monkeypatch, tmp_path):
    config_dir = tmp_path / "incomplete-config"
    shutil.copytree(ROOT / "config", config_dir)
    (config_dir / "handoff_actions.yaml").unlink()
    monkeypatch.setenv("SHOPCARE_CONFIG_DIR", str(config_dir))

    with pytest.raises(FileNotFoundError, match="handoff_actions.yaml"):
        handoff_generation.validate_handoff_configuration()


def test_compose_mounts_the_same_read_only_config_directory_used_by_backend():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    backend = compose["services"]["backend"]
    environment = backend["environment"]

    assert environment["SHOPCARE_CONFIG_DIR"] == "/config"
    assert "./config:/config:ro" in backend["volumes"]
