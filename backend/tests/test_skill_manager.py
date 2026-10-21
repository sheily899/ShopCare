"""本地领域 Skill 元数据、范围过滤与 Prompt 注入测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from agents.agent_orchestrator import Request, TechnicalAgent
from skills.skill_manager import SkillManager


SKILLS_ROOT = Path(__file__).parents[1] / "skills"


def test_skill_manager_loads_three_domain_skills_with_precise_metadata() -> None:
    manager = SkillManager(str(SKILLS_ROOT))

    assert manager.skills == ["after_sales", "consult", "order"]
    assert manager.specs["consult"].agents == ("consult",)
    assert manager.specs["order"].description
    assert manager.specs["after_sales"].enabled is True


def test_relative_skill_path_resolves_from_repository_root(monkeypatch) -> None:
    monkeypatch.chdir(Path(__file__).parents[2])

    assert "order" in SkillManager("skills").skills


def test_skill_prompt_is_scoped_and_includes_its_description_and_keywords() -> None:
    manager = SkillManager(str(SKILLS_ROOT))

    consult_prompt = manager.prompt_for_agent("consult")
    order_prompt = manager.prompt_for_agent("order")
    after_sales_prompt = manager.prompt_for_agent("after_sales")

    assert "美妆商品咨询" in consult_prompt and "成分" in consult_prompt
    assert "美妆订单履约" in order_prompt
    assert "退款未到账" not in order_prompt
    assert "美妆售后与风险处理" in after_sales_prompt
    assert "运行时意图策略注册表" in after_sales_prompt


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="已核对订单状态。")])


def test_agent_injects_matching_skill_into_system_prompt() -> None:
    client = _FakeClient()
    agent = TechnicalAgent(client=client, skill_manager=SkillManager(str(SKILLS_ROOT)))

    result = asyncio.run(agent.handle(Request(message="接口返回 401")))

    assert result.success is True
    assert "美妆订单履约" in client.calls[0]["system"]
    assert "美妆售后与风险处理" not in client.calls[0]["system"]


def test_skill_prompt_respects_configured_character_limit() -> None:
    manager = SkillManager(str(SKILLS_ROOT), max_prompt_chars=80)

    assert len(manager.prompt_for_agent("consult")) <= 80
